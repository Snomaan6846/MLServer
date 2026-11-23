/* Flat numeric JSON span -> preallocated NumPy buffer.
 *
 * Grammar: strict JSON numbers with commas/whitespace as separators —
 * a consumed comma must be followed by a value (no trailing commas),
 * no leading zeros, no bare dots/exponents. Anything outside the
 * grammar returns -1 and the Python caller takes the standard parse,
 * so acceptance/rejection behaviour matches the standard path by
 * construction (enforced by differential fuzzing against orjson).
 *
 * Value contract: outputs are bit-identical to the standard path's
 * decimal -> correctly-rounded-double ( -> narrowed float32 )
 * sequence. Three tiers:
 *   - mant <= 2^53: the mantissa and the POW10 doubles are exact, so
 *     a single multiply/divide is the one IEEE rounding of the exact
 *     decimal value.
 *   - mant <= 2^63: 128-bit integer arithmetic computes the exact
 *     product (or a >= 54-bit fixed-point quotient for division),
 *     rounded once to the nearest double, ties-to-even, with the
 *     division's discarded fraction breaking exact ties upward.
 *   - anything else (mant > 2^63, extreme exponents, quotients too
 *     small to round): strtod on a copy — the C library's correctly-
 *     rounded conversion, which is what the fallback exists for.
 *
 * The strtod fallback token INCLUDES the sign, so the sign flag is
 * cleared before the common tail re-applies it (a double negation
 * produces wrong signs for 20+-digit negative tokens).
 *
 * Compile: gcc -O3 -shared -fPIC -o libfastspan.so fastspan.c
 */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static const double POW10[23] = {
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11,
    1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22};

static const unsigned __int128 POW10_128[23] = {
    1u,  10u,  100u,  1000u,  10000u,  100000u,  1000000u,
    10000000u,  100000000u,  1000000000u,  10000000000u,
    100000000000u,  1000000000000u,  10000000000000u,
    100000000000000u,  1000000000000000u,  10000000000000000u,
    100000000000000000u,  1000000000000000000u,
    10000000000000000000u,
    ((unsigned __int128)10000000000000000000u) * 10u,
    ((unsigned __int128)10000000000000000000u) * 100u,
    ((unsigned __int128)10000000000000000000u) * 1000u};

/* Bit length of a nonzero 128-bit value. */
static int u128_bits(unsigned __int128 v)
{
    unsigned long long hi = (unsigned long long)(v >> 64);
    unsigned long long lo = (unsigned long long)v;
    if (hi != 0)
        return 128 - __builtin_clzll(hi);
    return 64 - __builtin_clzll(lo);
}

/* Round an exact non-negative 128-bit integer to the nearest double,
 * ties-to-even. ``has_fraction`` marks that the true value is
 * strictly greater than v (the division path's discarded remainder):
 * an exact tie then rounds up, not to even.
 * Callers keep the result in normal-double range. */
static double u128_to_double(unsigned __int128 v, int has_fraction)
{
    if (v == 0)
        return 0.0;
    int bits = u128_bits(v);
    if (bits <= 53)
        return (double)v; /* exact */
    int shift = bits - 53;
    unsigned long long sig = (unsigned long long)(v >> shift);
    unsigned __int128 rest = v & ((((unsigned __int128)1) << shift) - 1);
    unsigned __int128 halfway = ((unsigned __int128)1) << (shift - 1);
    if (rest > halfway) {
        sig++;
    } else if (rest == halfway) {
        if (has_fraction || (sig & 1))
            sig++;
    }
    /* sig is in [2^52, 2^53] and exactly representable; scaling by a
     * power of two is exact within the normal range. */
    return (double)sig * ldexp(1.0, shift);
}

#define DEFINE_SPAN_PARSER(SUFFIX, OUT_T)                                  \
    long fastspan_parse_##SUFFIX(const char *buf, size_t len, OUT_T *out,  \
                                 size_t max_count)                         \
    {                                                                      \
        size_t i = 0;                                                      \
        size_t count = 0;                                                  \
        while (i < len) {                                                  \
            while (i < len && (buf[i] == ' ' || buf[i] == '\t' ||          \
                               buf[i] == '\n' || buf[i] == '\r'))          \
                i++;                                                       \
            if (i >= len)                                                  \
                break;                                                     \
            if (count >= max_count)                                        \
                return -1;                                                 \
            const char *start = buf + i;                                   \
            int negative = 0;                                              \
            if (buf[i] == '-') {                                           \
                negative = 1;                                              \
                i++;                                                       \
            }                                                              \
            if (i >= len || buf[i] < '0' || buf[i] > '9')                  \
                return -1;                                                 \
            unsigned long long mant = 0;                                   \
            int digits = 0;                                                \
            int exp10 = 0;                                                 \
            int has_fraction_or_exp = 0;                                   \
            int overflow = 0;                                              \
            if (buf[i] == '0') {                                           \
                i++;                                                       \
                digits++; /* the zero IS a digit: "0"/"-0" take the    \
                           * tier path (not strtod, whose "-0" is     \
                           * -0.0 where the standard path gives +0)   \
                           */                                         \
                if (i < len && buf[i] >= '0' && buf[i] <= '9')             \
                    return -1; /* leading zero */                          \
            } else {                                                       \
                while (i < len && buf[i] >= '0' && buf[i] <= '9') {         \
                    {                                                       \
                        unsigned long long digit =                          \
                            (unsigned long long)(buf[i] - '0');            \
                        if (mant > (18446744073709551615ULL - digit) / 10) {  \
                            overflow = 1;                                   \
                        } else {                                            \
                            mant = mant * 10 + digit;                      \
                        }                                                   \
                    }                                                       \
                    digits++;                                              \
                    i++;                                                   \
                }                                                          \
            }                                                              \
            if (i < len && buf[i] == '.') {                                 \
                has_fraction_or_exp = 1;                                    \
                i++;                                                       \
                if (i >= len || buf[i] < '0' || buf[i] > '9')              \
                    return -1; /* bare dot */                              \
                while (i < len && buf[i] >= '0' && buf[i] <= '9') {         \
                    {                                                       \
                        unsigned long long digit =                          \
                            (unsigned long long)(buf[i] - '0');            \
                        if (mant > (18446744073709551615ULL - digit) / 10) {  \
                            overflow = 1;                                   \
                        } else {                                            \
                            mant = mant * 10 + digit;                      \
                            exp10--;                                       \
                        }                                                   \
                    }                                                       \
                    digits++;                                              \
                    i++;                                                   \
                }                                                          \
            }                                                              \
            if (i < len && (buf[i] == 'e' || buf[i] == 'E')) {             \
                has_fraction_or_exp = 1;                                    \
                i++;                                                       \
                int eneg = 0;                                              \
                if (i < len && (buf[i] == '+' || buf[i] == '-')) {         \
                    eneg = (buf[i] == '-');                                \
                    i++;                                                   \
                }                                                          \
                if (i >= len || buf[i] < '0' || buf[i] > '9')              \
                    return -1; /* bare exponent */                         \
                int eval = 0;                                              \
                while (i < len && buf[i] >= '0' && buf[i] <= '9') {         \
                    if (eval < 100000)                                     \
                        eval = eval * 10 + (buf[i] - '0');                 \
                    i++;                                                   \
                }                                                          \
                exp10 += eneg ? -eval : eval;                              \
            }                                                              \
            double value;                                                  \
            if (!overflow && digits > 0 && exp10 >= -22 && exp10 <= 22) {  \
                if (mant <= 9007199254740992ULL) {                         \
                    /* Tier 1: exact mantissa and exact power of ten —    \
                     * the single multiply/divide is the one IEEE        \
                     * rounding of the exact decimal. */                  \
                    value = (exp10 >= 0)                                   \
                                ? (double)mant * POW10[exp10]              \
                                : (double)mant / POW10[-exp10];            \
                } else if (mant <= 9223372036854775807ULL) {               \
                    /* Tier 2: 128-bit exact arithmetic, rounded once. */ \
                    if (exp10 >= 0) {                                      \
                        unsigned __int128 den = POW10_128[exp10];          \
                        if ((unsigned __int128)mant >                      \
                            ~((unsigned __int128)0) / den) {               \
                            goto fallback;                                 \
                        }                                                  \
                        value = u128_to_double(                            \
                            (unsigned __int128)mant * den, 0);             \
                    } else {                                              \
                        int k = -exp10;                                    \
                        /* Fixed-point quotient with >= 54 significant   \
                         * bits; the flooring remainder breaks exact     \
                         * ties upward. */                                \
                        unsigned __int128 num =                            \
                            ((unsigned __int128)mant) << 65;               \
                        unsigned __int128 den = POW10_128[k];              \
                        unsigned __int128 q = num / den;                   \
                        if (q < (((unsigned __int128)1) << 54)) {         \
                            goto fallback;                                 \
                        }                                                  \
                        value = u128_to_double(q, (num % den) != 0) *      \
                                ldexp(1.0, -65);                           \
                    }                                                     \
                } else {                                                  \
                    goto fallback;                                         \
                }                                                         \
            } else {                                                      \
            fallback:                                                     \
                /* Rare long/extreme token: the C library decides. The  \
                 * token starts at `start`, which INCLUDES the sign —   \
                 * strtod consumes it, so clear the flag the common     \
                 * tail would re-apply. */                               \
                {                                                         \
                    size_t tlen = (size_t)(buf + i - start);               \
                    char *heap = (char *)malloc(tlen + 1);                 \
                    if (!heap)                                             \
                        return -1;                                         \
                    memcpy(heap, start, tlen);                             \
                    heap[tlen] = '\0';                                     \
                    char *end;                                             \
                    value = strtod(heap, &end);                            \
                    /* Full-token consumption and a finite result, or   \
                     * reject: a partial consumption means strtod      \
                     * stopped early (e.g. a non-C LC_NUMERIC locale  \
                     * rejects our '.'), and inf/nan tokens defer to   \
                     * the standard path's semantics. */                \
                    if (end != heap + tlen || !isfinite(value)) {          \
                        free(heap);                                        \
                        return -1;                                         \
                    }                                                     \
                    free(heap);                                            \
                    negative = 0;                                          \
                }                                                         \
            }                                                             \
            /* Integer "-0" is Python int 0 on the standard path (+0):  \
             * the sign applies only to float-form zero ("-0.0",       \
             * "-0e0"), which orjson keeps as -0.0. */                   \
            if (negative && value == 0.0 && !has_fraction_or_exp)         \
                negative = 0;                                             \
            out[count++] = negative ? (OUT_T)-value : (OUT_T)value;        \
            while (i < len && (buf[i] == ' ' || buf[i] == '\t' ||          \
                               buf[i] == '\n' || buf[i] == '\r'))          \
                i++;                                                       \
            if (i < len) {                                                 \
                if (buf[i] != ',')                                         \
                    return -1;                                             \
                i++;                                                       \
                /* A consumed separator must be followed by a value:      \
                 * a trailing comma (with or without trailing whitespace) \
                 * is invalid JSON and the loop-top break would           \
                 * otherwise accept it (found by differential fuzzing     \
                 * against orjson). */                                    \
                while (i < len && (buf[i] == ' ' || buf[i] == '\t' ||     \
                                   buf[i] == '\n' || buf[i] == '\r'))     \
                    i++;                                                   \
                if (i >= len)                                             \
                    return -1;                                             \
            }                                                              \
        }                                                                  \
        return (long)count;                                                \
    }

DEFINE_SPAN_PARSER(f32, float)
DEFINE_SPAN_PARSER(f64, double)

/* Integer span parser: same grammar as the float parser, but the
 * output is a C integer type. Pure-integer tokens use the mantissa
 * directly (exact for INT64/UINT64 boundary values that lose
 * precision through double); tokens with fraction/exponent go
 * through the double and truncate toward zero (matching
 * np.fromiter's behaviour). Out-of-range values return -1 (the
 * Python caller falls back, where the standard path raises
 * OverflowError). */
#define DEFINE_INT_SPAN_PARSER(SUFFIX, OUT_T, MIN_VAL, MAX_VAL)             \
    long fastspan_parse_##SUFFIX(const char *buf, size_t len, OUT_T *out,   \
                                 size_t max_count)                          \
    {                                                                       \
        size_t i = 0;                                                       \
        size_t count = 0;                                                   \
        while (i < len) {                                                   \
            while (i < len && (buf[i] == ' ' || buf[i] == '\t' ||           \
                               buf[i] == '\n' || buf[i] == '\r'))           \
                i++;                                                        \
            if (i >= len)                                                   \
                break;                                                      \
            if (count >= max_count)                                         \
                return -1;                                                  \
            const char *start = buf + i;                                    \
            int negative = 0;                                              \
            if (buf[i] == '-') {                                            \
                negative = 1;                                              \
                i++;                                                        \
            }                                                              \
            if (i >= len || buf[i] < '0' || buf[i] > '9')                  \
                return -1;                                                 \
            unsigned long long mant = 0;                                   \
            int digits = 0;                                                \
            int exp10 = 0;                                                 \
            int overflow = 0;                                              \
            int has_fraction_or_exp = 0;                                   \
            if (buf[i] == '0') {                                           \
                i++;                                                        \
                if (i < len && buf[i] >= '0' && buf[i] <= '9')             \
                    return -1; /* leading zero */                           \
            } else {                                                        \
                while (i < len && buf[i] >= '0' && buf[i] <= '9') {          \
                    /* Dynamic overflow check: the threshold depends on    \
                     * the CURRENT digit — a fixed threshold either      \
                     * rejects valid values (UINT64_MAX with digit 5)   \
                     * or accepts wraps (2^64 with digit 6). */         \
                    unsigned long long digit =                              \
                        (unsigned long long)(buf[i] - '0');                \
                    if (mant > (18446744073709551615ULL - digit) / 10) {  \
                        overflow = 1;                                       \
                    } else {                                                \
                        mant = mant * 10 + digit;                          \
                    }                                                       \
                    digits++;                                              \
                    i++;                                                    \
                }                                                           \
            }                                                               \
            if (i < len && buf[i] == '.') {                                 \
                has_fraction_or_exp = 1;                                    \
                i++;                                                        \
                if (i >= len || buf[i] < '0' || buf[i] > '9')              \
                    return -1; /* bare dot */                               \
                while (i < len && buf[i] >= '0' && buf[i] <= '9') {          \
                    {                                                       \
                        unsigned long long digit =                          \
                            (unsigned long long)(buf[i] - '0');            \
                        if (mant > (18446744073709551615ULL - digit) / 10) { \
                            overflow = 1;                                   \
                        } else {                                            \
                            mant = mant * 10 + digit;                      \
                            exp10--;                                       \
                        }                                                   \
                    }                                                       \
                    digits++;                                              \
                    i++;                                                    \
                }                                                           \
            }                                                               \
            if (i < len && (buf[i] == 'e' || buf[i] == 'E')) {              \
                has_fraction_or_exp = 1;                                    \
                i++;                                                        \
                int eneg = 0;                                              \
                if (i < len && (buf[i] == '+' || buf[i] == '-')) {          \
                    eneg = (buf[i] == '-');                                 \
                    i++;                                                    \
                }                                                           \
                if (i >= len || buf[i] < '0' || buf[i] > '9')              \
                    return -1; /* bare exponent */                          \
                int eval = 0;                                              \
                while (i < len && buf[i] >= '0' && buf[i] <= '9') {          \
                    if (eval < 100000)                                      \
                        eval = eval * 10 + (buf[i] - '0');                  \
                    i++;                                                    \
                }                                                           \
                exp10 += eneg ? -eval : eval;                              \
            }                                                               \
            /* Compute the value */                                         \
            if (!has_fraction_or_exp && !overflow && digits > 0) {          \
                /* Pure integer: use the mantissa directly (exact). */      \
                if (negative) {                                            \
                    /* |MIN| = MAX + 1 in two's complement; the */         \
                    /* unsigned magnitude avoids the -MIN_VAL overflow  */ \
                    /* warning and the out-of-range signed negation.  */   \
                    /* Unsigned types have MIN_VAL == 0: only -0 is a  */  \
                    /* valid token (any other negative must reject, or  */ \
                    /* the two's-complement write below silently wraps  */ \
                    /* it into the unsigned range — feeding the model a  */ \
                    /* value the standard path rejects). */                \
                    if (MIN_VAL == 0 && mant != 0)                          \
                        return -1; /* negative in an unsigned type */       \
                    if (mant > (unsigned long long)MAX_VAL + 1ULL)         \
                        return -1; /* out of range */                       \
                    out[count++] = (OUT_T)(~mant + 1ULL);                   \
                } else {                                                    \
                    if (mant > (unsigned long long)MAX_VAL)                \
                        return -1; /* out of range */                       \
                    out[count++] = (OUT_T)mant;                             \
                }                                                           \
            } else {                                                        \
                /* Fraction/exponent: go through double, truncate. */       \
                double value;                                               \
                if (!overflow && digits > 0 &&                              \
                    exp10 >= -22 && exp10 <= 22) {                          \
                    if (mant <= 9007199254740992ULL) {                      \
                        value = (exp10 >= 0)                                \
                                    ? (double)mant * POW10[exp10]           \
                                    : (double)mant / POW10[-exp10];         \
                    } else {                                               \
                        goto int_fallback;                                  \
                    }                                                       \
                } else {                                                    \
                int_fallback:                                               \
                    {                                                       \
                        size_t tlen = (size_t)(buf + i - start);            \
                        char *heap = (char *)malloc(tlen + 1);              \
                        if (!heap)                                          \
                            return -1;                                      \
                        memcpy(heap, start, tlen);                          \
                        heap[tlen] = '\0';                                  \
                        char *end;                                          \
                        value = strtod(heap, &end);                         \
                        /* Full-token consumption and a finite result,    */ \
                        /* or reject (same contract as the float parser's */ \
                        /* fallback: partial consumption means a locale   */ \
                        /* or grammar surprise — the standard path decides. */ \
                        if (end != heap + tlen || !isfinite(value)) {       \
                            free(heap);                                     \
                            return -1;                                      \
                        }                                                  \
                        free(heap);                                         \
                        /* The heap copy starts at `start`, which INCLUDES */ \
                        /* the leading '-': strtod already applied the    */ \
                        /* sign. Clear the flag or the tail below would   */ \
                        /* negate again (the float parser's fallback does */ \
                        /* exactly this).                                  */ \
                        negative = 0;                                       \
                    }                                                       \
                }                                                           \
                if (negative)                                              \
                    value = -value; /* Bug fix: apply sign BEFORE trunc */ \
                if (isnan(value) || isinf(value))                           \
                    return -1;                                              \
                /* Only accept values representable exactly as integers  \
                 * in a double (<= 2^53). Larger values lose precision  \
                 * through the double route; the standard path handles  \
                 * them exactly via Python ints. */                       \
                if (fabs(value) > 9007199254740992.0)                     \
                    return -1;                                              \
                /* Truncate toward zero (matches np.fromiter). */           \
                double truncated = trunc(value);                            \
                /* Exact range check (safe: |truncated| <= 2^53 which  \
                 * is exactly representable, so comparisons are exact). */ \
                if (truncated < (double)MIN_VAL ||                          \
                    truncated > (double)MAX_VAL)                            \
                    return -1; /* out of range — std path wraps */         \
                out[count++] = (OUT_T)truncated;                            \
            }                                                               \
            while (i < len && (buf[i] == ' ' || buf[i] == '\t' ||           \
                               buf[i] == '\n' || buf[i] == '\r'))           \
                i++;                                                        \
            if (i < len) {                                                  \
                if (buf[i] != ',')                                          \
                    return -1;                                              \
                i++;                                                        \
                while (i < len && (buf[i] == ' ' || buf[i] == '\t' ||       \
                                   buf[i] == '\n' || buf[i] == '\r'))       \
                    i++;                                                    \
                if (i >= len)                                              \
                    return -1;                                              \
            }                                                               \
        }                                                                   \
        return (long)count;                                                 \
    }

DEFINE_INT_SPAN_PARSER(i8, int8_t, INT8_MIN, INT8_MAX)
DEFINE_INT_SPAN_PARSER(i16, int16_t, INT16_MIN, INT16_MAX)
DEFINE_INT_SPAN_PARSER(i32, int32_t, INT32_MIN, INT32_MAX)
DEFINE_INT_SPAN_PARSER(i64, int64_t, INT64_MIN, INT64_MAX)
DEFINE_INT_SPAN_PARSER(u8, uint8_t, 0, UINT8_MAX)
DEFINE_INT_SPAN_PARSER(u16, uint16_t, 0, UINT16_MAX)
DEFINE_INT_SPAN_PARSER(u32, uint32_t, 0, UINT32_MAX)
DEFINE_INT_SPAN_PARSER(u64, uint64_t, 0, UINT64_MAX)

/* BOOL span parser: accepts JSON true/false literals AND numeric
 * tokens (any non-zero → true, matching np.fromiter's behaviour).
 * String bools (["true"]) contain quotes and are skipped by the
 * Python span extractor upstream — they never reach this parser. */
static int is_ws_or_comma(char c) { return c == ' ' || c == '\t' ||        \
    c == '\n' || c == '\r' || c == ','; }

#define DEFINE_BOOL_SPAN_PARSER(SUFFIX, OUT_T)                              \
    long fastspan_parse_##SUFFIX(const char *buf, size_t len, OUT_T *out,   \
                                 size_t max_count)                          \
    {                                                                       \
        size_t i = 0;                                                       \
        size_t count = 0;                                                   \
        while (i < len) {                                                   \
            while (i < len && (buf[i] == ' ' || buf[i] == '\t' ||           \
                               buf[i] == '\n' || buf[i] == '\r'))           \
                i++;                                                        \
            if (i >= len)                                                   \
                break;                                                      \
            if (count >= max_count)                                         \
                return -1;                                                  \
            /* Try true/false literals first */                             \
            if (i + 4 <= len && memcmp(buf + i, "true", 4) == 0) {          \
                out[count++] = 1;                                           \
                i += 4;                                                     \
            } else if (i + 5 <= len && memcmp(buf + i, "false", 5) == 0) {  \
                out[count++] = 0;                                           \
                i += 5;                                                     \
            } else {                                                        \
                /* Numeric token: parse as double, non-zero → true. */     \
                const char *start = buf + i;                                \
                while (i < len && !is_ws_or_comma(buf[i]))                   \
                    i++;                                                    \
                /* Validate the FULL JSON number grammar before strtod:  \
                 * strtod accepts C-extensions JSON rejects (hex "0x10", \
                 * trailing "1.", leading "01") — any such token must   \
                 * reject so the standard path decides. */              \
                {                                                           \
                    size_t tlen = (size_t)(buf + i - start);                \
                    size_t j = 0;                                           \
                    if (j < tlen && start[j] == '-') j++;                   \
                    if (j >= tlen || start[j] < '0' || start[j] > '9')     \
                        return -1; /* not a number */                       \
                    if (start[j] == '0') {                                  \
                        j++; /* single zero or leading-zero reject */     \
                        if (j < tlen && start[j] >= '0' &&                 \
                            start[j] <= '9')                               \
                            return -1; /* leading zero */                   \
                    } else {                                                \
                        while (j < tlen && start[j] >= '0' &&              \
                               start[j] <= '9')                            \
                            j++;                                            \
                    }                                                       \
                    if (j < tlen && start[j] == '.') {                      \
                        j++;                                                \
                        if (j >= tlen || start[j] < '0' ||                 \
                            start[j] > '9')                                 \
                            return -1; /* bare dot */                       \
                        while (j < tlen && start[j] >= '0' &&              \
                               start[j] <= '9')                            \
                            j++;                                            \
                    }                                                       \
                    if (j < tlen &&                                         \
                        (start[j] == 'e' || start[j] == 'E')) {             \
                        j++;                                                \
                        if (j < tlen &&                                     \
                            (start[j] == '+' || start[j] == '-'))           \
                            j++;                                            \
                        if (j >= tlen || start[j] < '0' ||                 \
                            start[j] > '9')                                 \
                            return -1; /* bare exponent */                  \
                        while (j < tlen && start[j] >= '0' &&              \
                               start[j] <= '9')                            \
                            j++;                                            \
                    }                                                       \
                    if (j != tlen)                                          \
                        return -1; /* trailing non-number chars (hex, etc) */ \
                }                                                           \
                double value = 0;                                           \
                {                                                           \
                    size_t tlen = (size_t)(buf + i - start);                \
                    char *tok = (char *)malloc(tlen + 1);                   \
                    if (!tok)                                               \
                        return -1;                                          \
                    memcpy(tok, start, tlen);                               \
                    tok[tlen] = '\0';                                       \
                    char *end;                                              \
                    value = strtod(tok, &end);                              \
                    if (end != tok + tlen || end == tok) {                  \
                        free(tok);                                          \
                        return -1; /* not a valid number */                 \
                    }                                                       \
                    free(tok);                                              \
                }                                                           \
                if (isnan(value) || isinf(value))                           \
                    return -1;                                              \
                out[count++] = (value != 0.0) ? 1 : 0;                      \
            }                                                               \
            while (i < len && (buf[i] == ' ' || buf[i] == '\t' ||           \
                               buf[i] == '\n' || buf[i] == '\r'))           \
                i++;                                                        \
            if (i < len) {                                                  \
                if (buf[i] != ',')                                          \
                    return -1;                                              \
                i++;                                                        \
                while (i < len && (buf[i] == ' ' || buf[i] == '\t' ||       \
                                   buf[i] == '\n' || buf[i] == '\r'))       \
                    i++;                                                    \
                if (i >= len)                                              \
                    return -1;                                              \
            }                                                               \
        }                                                                   \
        return (long)count;                                                 \
    }

DEFINE_BOOL_SPAN_PARSER(bool, uint8_t)
