# Aurora Platform Overview

Aurora is a fictional internal developer platform used as the sample corpus
for this project. It lets teams deploy containerized services to shared
Kubernetes clusters without writing any Kubernetes manifests themselves.

## Deployments

Services are deployed with the `aurora deploy` command. Every deployment is
immutable: a new release is created each time, and the previous ten releases
are retained for instant rollback. Rollback is performed with
`aurora rollback --release <id>` and completes in under 30 seconds because
the old container images are kept warm in the regional registry cache.

Deployments to the production environment require two approvals in the
Aurora web console. Staging and development environments deploy immediately
with no approval step.

## Environments

Aurora provides three environments per team: development, staging, and
production. Development environments are scaled to zero overnight (between
20:00 and 06:00 UTC) to save cluster costs. Staging mirrors production
configuration but runs at 25 percent of production capacity.

## Service limits

Each service may use at most 4 vCPUs and 8 GiB of memory per replica.
Teams can run up to 20 replicas per service in production. Requests for
higher limits go through the platform team via a QUOTA ticket in Jira.

## Observability

Every Aurora service automatically ships logs to the central Loki cluster
and metrics to Prometheus. Traces are sampled at 10 percent by default;
teams can raise the sampling rate to 100 percent for debugging windows of
up to 24 hours using `aurora trace --sample 1.0 --ttl 24h`.

## Support

The platform team operates an on-call rotation reachable in the
`#aurora-support` Slack channel. Incident response targets are 15 minutes
for production-impacting issues and one business day for everything else.
