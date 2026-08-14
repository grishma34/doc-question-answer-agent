# Aurora Security Policy

This document describes the (fictional) security requirements for services
running on the Aurora platform.

## Secrets management

Services must read secrets from Aurora Vault, which is backed by AWS
Secrets Manager. Hard-coding secrets in container images, environment
files, or source code is prohibited and blocked by the pre-deploy scanner.
Secrets are rotated automatically every 90 days; database credentials are
rotated every 30 days.

## Network policy

All service-to-service traffic inside Aurora is encrypted with mutual TLS,
issued by the platform's internal certificate authority. Certificates have
a 7-day lifetime and are renewed automatically. Services may not expose
public endpoints directly; all external traffic must enter through the
Aurora edge gateway, which enforces WAF rules and rate limiting of 1000
requests per minute per client IP by default.

## Vulnerability management

Container images are scanned on every build. Deployments are blocked if
the image contains any critical CVE older than 7 days or any high CVE
older than 30 days. Teams receive a weekly digest of open findings in
their `#team-security` Slack channel.

## Access control

Production access follows least privilege: engineers get read-only access
by default, and write or exec access requires a time-boxed elevation
through the Access Portal, capped at 4 hours per grant. All elevations are
logged and reviewed weekly by the security team.

## Incident reporting

Suspected security incidents must be reported within 1 hour of discovery
to the security on-call via the `#security-incidents` channel or the
emergency pager. The security team runs a blameless postmortem for every
confirmed incident within 5 business days.

## Data classification

Aurora services may process data classified as Public, Internal, or
Confidential. Restricted data (payment card numbers, government IDs) is
not permitted on the shared platform and requires the dedicated isolated
tier, which is provisioned by request only.
