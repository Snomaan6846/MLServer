"""
MLServer Inference Request Flow — Animated Architecture Diagram.

Animates the step-by-step path of an inference request through MLServer:
Client → REST/gRPC → DataPlane → Registry → Cache → Model → Response

Generated with Manim Community Edition (MIT license).
Run: manim -qh --format=gif inference_flow_animation.py InferenceRequestFlow
"""

from manim import *

# ── Colour palette ──────────────────────────────────────────────────
BG_COLOR = "#1a1a2e"
BOX_BLUE = "#4A90D9"
BOX_GREEN = "#7ED321"
BOX_ORANGE = "#F5A623"
BOX_RED = "#D0021B"
BOX_PURPLE = "#9B59B6"
BOX_TEAL = "#16a085"
ARROW_COLOR = "#E0E0E0"
HIGHLIGHT = "#FFD700"
DATA_DOT = "#00FFAA"


def make_box(label: str, color: str, width: float = 2.2, height: float = 0.9) -> VGroup:
    """Create a labelled rounded rectangle component box."""
    rect = RoundedRectangle(
        corner_radius=0.15,
        width=width,
        height=height,
        stroke_color=color,
        fill_color=color,
        fill_opacity=0.15,
        stroke_width=2.5,
    )
    text = Text(label, font_size=18, color=WHITE, font="Monospace")
    text.move_to(rect.get_center())
    return VGroup(rect, text)


def make_arrow(start, end, color=ARROW_COLOR):
    """Create a styled arrow between two mobjects."""
    return Arrow(
        start.get_center(),
        end.get_center(),
        buff=0.45,
        stroke_width=2,
        color=color,
        max_tip_length_to_length_ratio=0.12,
    )


def pulse(mobject, color=HIGHLIGHT, scale=1.08):
    """Return an animation that pulses a box to indicate activity."""
    return AnimationGroup(
        mobject[0].animate.set_stroke(color=color, width=4),
        mobject.animate.scale(scale),
        rate_func=there_and_back,
        run_time=0.5,
    )


def flow_dot(start, end, color=DATA_DOT):
    """Return a dot that travels from start to end along a line."""
    dot = Dot(radius=0.08, color=color).move_to(start.get_center())
    path = Line(start.get_center(), end.get_center())
    return dot, MoveAlongPath(dot, path, run_time=0.6, rate_func=smooth)


class InferenceRequestFlow(Scene):
    """Animated inference request flow through MLServer components."""

    def construct(self):
        self.camera.background_color = BG_COLOR

        # ── Title ───────────────────────────────────────────────────
        title = Text(
            "MLServer — Inference Request Flow",
            font_size=28,
            color=WHITE,
            font="Monospace",
        ).to_edge(UP, buff=0.4)
        subtitle = Text(
            "V2 Inference Protocol  ·  REST + gRPC",
            font_size=16,
            color=GRAY_B,
            font="Monospace",
        ).next_to(title, DOWN, buff=0.15)
        self.play(Write(title), FadeIn(subtitle, shift=UP * 0.2), run_time=1)
        self.wait(0.3)

        # ── Build components ────────────────────────────────────────
        client = make_box("Client", BOX_ORANGE, width=1.8)
        rest = make_box("REST :8080", BOX_BLUE, width=2.0)
        grpc = make_box("gRPC :8081", BOX_BLUE, width=2.0)
        dataplane = make_box("DataPlane", BOX_GREEN, width=2.4)
        registry = make_box("Model\nRegistry", BOX_PURPLE, width=2.0, height=1.0)
        cache = make_box("Response\nCache", BOX_TEAL, width=2.0, height=1.0)
        model = make_box("MLModel\n(sklearn)", BOX_RED, width=2.0, height=1.0)
        metrics = make_box("Prometheus\nMetrics", BOX_ORANGE, width=2.0, height=1.0)

        # ── Layout ──────────────────────────────────────────────────
        client.move_to(LEFT * 5.5 + UP * 0.5)
        rest.move_to(LEFT * 2.8 + UP * 1.2)
        grpc.move_to(LEFT * 2.8 + DOWN * 0.3)
        dataplane.move_to(ORIGIN + UP * 0.5)
        registry.move_to(RIGHT * 2.8 + UP * 1.5)
        cache.move_to(RIGHT * 2.8 + DOWN * 0.0)
        model.move_to(RIGHT * 5.5 + UP * 0.5)
        metrics.move_to(ORIGIN + DOWN * 1.8)

        components = VGroup(
            client, rest, grpc, dataplane, registry, cache, model, metrics
        )

        # ── Arrows ──────────────────────────────────────────────────
        a_client_rest = make_arrow(client, rest, BOX_BLUE)
        a_client_grpc = make_arrow(client, grpc, BOX_BLUE)
        a_rest_dp = make_arrow(rest, dataplane, BOX_GREEN)
        a_grpc_dp = make_arrow(grpc, dataplane, BOX_GREEN)
        a_dp_reg = make_arrow(dataplane, registry, BOX_PURPLE)
        a_dp_cache = make_arrow(dataplane, cache, BOX_TEAL)
        a_cache_model = make_arrow(cache, model, BOX_RED)
        a_dp_metrics = make_arrow(dataplane, metrics, BOX_ORANGE)

        arrows = VGroup(
            a_client_rest,
            a_client_grpc,
            a_rest_dp,
            a_grpc_dp,
            a_dp_reg,
            a_dp_cache,
            a_cache_model,
            a_dp_metrics,
        )

        # ── Fade in all components ──────────────────────────────────
        self.play(
            *[FadeIn(c, shift=DOWN * 0.3) for c in components],
            run_time=1.2,
        )
        self.play(*[GrowArrow(a) for a in arrows], run_time=0.8)
        self.wait(0.5)

        # ── Step labels ─────────────────────────────────────────────
        def step_label(text, position):
            lbl = Text(text, font_size=14, color=HIGHLIGHT, font="Monospace")
            lbl.move_to(position)
            return lbl

        # ═══════════════════════════════════════════════════════════
        # STEP 1 — Client sends request
        # ═══════════════════════════════════════════════════════════
        s1 = step_label("① POST /v2/models/iris/infer", client.get_top() + UP * 0.5)
        self.play(FadeIn(s1, shift=UP * 0.2))
        self.play(pulse(client, HIGHLIGHT))

        dot1, anim1 = flow_dot(client, rest)
        self.play(FadeIn(dot1), anim1)
        self.play(pulse(rest, BOX_BLUE))
        self.play(FadeOut(dot1))
        self.wait(0.2)

        # ═══════════════════════════════════════════════════════════
        # STEP 2 — REST forwards to DataPlane
        # ═══════════════════════════════════════════════════════════
        s2 = step_label("② Parse request → DataPlane", rest.get_top() + UP * 0.4)
        self.play(FadeOut(s1), FadeIn(s2, shift=UP * 0.2))

        dot2, anim2 = flow_dot(rest, dataplane)
        self.play(FadeIn(dot2), anim2)
        self.play(pulse(dataplane, BOX_GREEN))
        self.play(FadeOut(dot2))
        self.wait(0.2)

        # ═══════════════════════════════════════════════════════════
        # STEP 3 — DataPlane queries Registry
        # ═══════════════════════════════════════════════════════════
        s3 = step_label("③ Resolve model from registry", dataplane.get_top() + UP * 0.4)
        self.play(FadeOut(s2), FadeIn(s3, shift=UP * 0.2))

        dot3, anim3 = flow_dot(dataplane, registry)
        self.play(FadeIn(dot3), anim3)
        self.play(pulse(registry, BOX_PURPLE))

        # Return dot
        dot3r, anim3r = flow_dot(registry, dataplane, color=BOX_PURPLE)
        self.play(FadeOut(dot3), FadeIn(dot3r), anim3r)
        self.play(FadeOut(dot3r))
        self.wait(0.2)

        # ═══════════════════════════════════════════════════════════
        # STEP 4 — Prometheus timer starts
        # ═══════════════════════════════════════════════════════════
        s4 = step_label("④ Start Prometheus timer", metrics.get_bottom() + DOWN * 0.35)
        self.play(FadeOut(s3), FadeIn(s4, shift=UP * 0.2))

        dot4, anim4 = flow_dot(dataplane, metrics, color=BOX_ORANGE)
        self.play(FadeIn(dot4), anim4)
        self.play(pulse(metrics, BOX_ORANGE))
        self.play(FadeOut(dot4))
        self.wait(0.2)

        # ═══════════════════════════════════════════════════════════
        # STEP 5 — Cache lookup (miss)
        # ═══════════════════════════════════════════════════════════
        s5 = step_label("⑤ Cache lookup → MISS", cache.get_bottom() + DOWN * 0.4)
        self.play(FadeOut(s4), FadeIn(s5, shift=UP * 0.2))

        dot5, anim5 = flow_dot(dataplane, cache)
        self.play(FadeIn(dot5), anim5)
        self.play(pulse(cache, BOX_TEAL))

        miss_label = Text("MISS", font_size=16, color=BOX_RED, font="Monospace")
        miss_label.next_to(cache, DOWN, buff=0.15)
        self.play(FadeIn(miss_label, scale=1.5), FadeOut(dot5))
        self.wait(0.3)

        # ═══════════════════════════════════════════════════════════
        # STEP 6 — Forward to Model for inference
        # ═══════════════════════════════════════════════════════════
        s6 = step_label("⑥ model.predict(payload)", model.get_top() + UP * 0.4)
        self.play(FadeOut(s5), FadeOut(miss_label), FadeIn(s6, shift=UP * 0.2))

        dot6, anim6 = flow_dot(cache, model)
        self.play(FadeIn(dot6), anim6)

        # Model processing — pulsing glow
        self.play(
            model[0].animate.set_fill(BOX_RED, opacity=0.4),
            rate_func=there_and_back,
            run_time=0.8,
        )
        self.play(pulse(model, BOX_RED))
        self.play(FadeOut(dot6))
        self.wait(0.2)

        # ═══════════════════════════════════════════════════════════
        # STEP 7 — Response flows back
        # ═══════════════════════════════════════════════════════════
        s7 = step_label("⑦ InferenceResponse → Client", UP * 2.8)
        self.play(FadeOut(s6), FadeIn(s7, shift=UP * 0.2))

        # Model → DataPlane (via cache for insert)
        dot7a, anim7a = flow_dot(model, cache, color=BOX_GREEN)
        self.play(FadeIn(dot7a), anim7a)

        insert_label = Text("INSERT", font_size=16, color=BOX_GREEN, font="Monospace")
        insert_label.next_to(cache, DOWN, buff=0.15)
        self.play(FadeIn(insert_label, scale=1.2), FadeOut(dot7a))
        self.wait(0.2)

        # Cache → DataPlane
        dot7b, anim7b = flow_dot(cache, dataplane, color=BOX_GREEN)
        self.play(FadeOut(insert_label), FadeIn(dot7b), anim7b)
        self.play(FadeOut(dot7b))

        # DataPlane → REST → Client
        dot7c, anim7c = flow_dot(dataplane, rest, color=BOX_GREEN)
        self.play(FadeIn(dot7c), anim7c)
        self.play(FadeOut(dot7c))

        dot7d, anim7d = flow_dot(rest, client, color=BOX_GREEN)
        self.play(FadeIn(dot7d), anim7d)
        self.play(pulse(client, BOX_GREEN))
        self.play(FadeOut(dot7d))

        # ═══════════════════════════════════════════════════════════
        # Final — 200 OK
        # ═══════════════════════════════════════════════════════════
        result = Text(
            "200 OK  ·  {model_name: iris, outputs: [...]}",
            font_size=16,
            color=BOX_GREEN,
            font="Monospace",
        ).next_to(client, DOWN, buff=0.5)
        self.play(FadeOut(s7), FadeIn(result, shift=UP * 0.2))

        # Highlight full path
        self.play(
            a_client_rest.animate.set_color(BOX_GREEN),
            a_rest_dp.animate.set_color(BOX_GREEN),
            a_dp_cache.animate.set_color(BOX_GREEN),
            a_cache_model.animate.set_color(BOX_GREEN),
            run_time=0.8,
        )
        self.wait(1.5)

        # ── Fade out ────────────────────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1)
