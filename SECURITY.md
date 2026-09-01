# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in MLServer, please report it
responsibly. **Do not open a public GitHub issue.**

Instead, please report vulnerabilities through one of the following channels:

- **Red Hat Product Security:** If you are using MLServer as part of
  Red Hat OpenShift AI, report via
  [Red Hat's vulnerability reporting process](https://access.redhat.com/security/team/contact/).

- **GitHub Security Advisories:** Use the
  [private vulnerability reporting](https://github.com/opendatahub-io/MLServer/security/advisories/new)
  feature on this repository.

Please include:

- A description of the vulnerability
- Steps to reproduce the issue
- The potential impact
- Any suggested remediation

We will acknowledge receipt within 3 business days and aim to provide an
initial assessment within 10 business days.

## Supported Versions

Security fixes are applied to the latest release branch. Older versions
may receive backports on a case-by-case basis.

## Security Model

MLServer operates in one of two security modes:

- **PRODUCTION** — enforced when a trusted runtimes allowlist file exists
  at `/etc/mlserver/trusted-runtimes.json`. Only explicitly allowlisted
  model implementations can be loaded.

- **DEVELOPMENT** — active when no allowlist file exists. Allows dynamic
  loading of arbitrary model implementations. Should never be used in
  production environments.

For full details, see the [Security Guide](./docs/engineering/security.md).
