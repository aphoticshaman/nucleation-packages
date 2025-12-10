# DEVSECOPS_MLOPS.skill.md

## Unified Operations: DevSecOps + MLOps for Production AI Systems

**Version**: 1.0
**Domain**: DevSecOps, MLOps, CI/CD, Security, Model Operations
**Prerequisites**: AUTOMATION_AND_WRAPPERS skill, CODING_BEST_PRACTICES skill
**Output**: Secure, automated, observable production systems

---

## 1. EXECUTIVE SUMMARY

Modern systems require unified operations across development, security, and machine learning. This skill covers patterns for building secure CI/CD pipelines, managing ML models in production, and creating observable, maintainable systems at scale.

**Core Principle**: Security is not a gate—it's woven into every stage. ML is not special—it needs the same rigor as any production system.

---

## 2. DEVSECOPS FUNDAMENTALS

### 2.1 Shift-Left Security

```
TRADITIONAL (security at end):
Dev → Test → Build → SECURITY REVIEW → Deploy → Maintain

SHIFT-LEFT (security throughout):
[Security] Dev → [Security] Test → [Security] Build → [Security] Deploy

SHIFT-LEFT PRACTICES:
├── Pre-commit hooks for secret scanning
├── SAST in every PR
├── Dependency scanning on every build
├── Container scanning before registry push
├── Policy-as-code in deployment
└── Runtime protection in production
```

### 2.2 Security Pipeline Integration

```yaml
# .github/workflows/security-pipeline.yml
name: Security Checks

on: [push, pull_request]

jobs:
  secret-scanning:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Gitleaks scan
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

  sast:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run Semgrep
        uses: returntocorp/semgrep-action@v1
        with:
          config: auto

  dependency-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: npm audit
        run: npm audit --audit-level=moderate

      - name: Snyk scan
        uses: snyk/actions/node@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}

  container-scan:
    runs-on: ubuntu-latest
    needs: [build]
    steps:
      - name: Trivy scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: '${{ env.IMAGE }}:${{ github.sha }}'
          format: 'sarif'
          output: 'trivy-results.sarif'

  license-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: License scan
        run: |
          npx license-checker --onlyAllow 'MIT;Apache-2.0;BSD-2-Clause;BSD-3-Clause;ISC'
```

### 2.3 Security Boundaries

```
SECURITY ZONES:
├── PUBLIC: Internet-facing, highest scrutiny
├── DMZ: Load balancers, CDN, API gateways
├── PRIVATE: Internal services, databases
├── SENSITIVE: Secrets management, key storage
└── AIR-GAPPED: Backup systems, audit logs

ZONE RULES:
├── Traffic flows inward only (public → private)
├── Secrets never leave sensitive zone
├── Logs always flow to air-gapped
├── Database never touches public zone
└── API gateway terminates all external connections
```

---

## 3. CI/CD PIPELINE ARCHITECTURE

### 3.1 Pipeline Stages

```yaml
# Complete CI/CD pipeline
stages:
  - validate:      # Syntax, lint, format
  - test:          # Unit, integration tests
  - security:      # SAST, dependency scan
  - build:         # Compile, bundle
  - scan:          # Container/artifact scan
  - staging:       # Deploy to staging
  - verify:        # E2E tests, smoke tests
  - approve:       # Manual approval (if required)
  - production:    # Deploy to production
  - monitor:       # Post-deploy verification

# STAGE REQUIREMENTS
validate:
  required: true
  blocks: test
  timeout: 5m

test:
  required: true
  blocks: build
  coverage_threshold: 80%

security:
  required: true
  blocks: build
  fail_on: [critical, high]
  warn_on: [medium]

production:
  requires: [staging.verify.passed, security.passed]
  approval: [tech-lead, security-team]
  rollback: automatic_on_failure
```

### 3.2 Deployment Strategies

```python
# DEPLOYMENT STRATEGIES

class BlueGreenDeployment:
    """Zero-downtime deployment with instant rollback."""

    def deploy(self, new_version: str) -> DeployResult:
        # 1. Deploy to inactive environment (green)
        green = self.deploy_to_green(new_version)

        # 2. Health check green
        if not self.health_check(green):
            return DeployResult(success=False, reason="Health check failed")

        # 3. Switch traffic
        self.switch_traffic(from_env='blue', to_env='green')

        # 4. Monitor
        if not self.monitor_period(duration=300):
            self.rollback()
            return DeployResult(success=False, reason="Rollback triggered")

        # 5. Update blue for next deploy
        self.update_blue(new_version)

        return DeployResult(success=True)

class CanaryDeployment:
    """Gradual rollout with automatic rollback."""

    def deploy(self, new_version: str) -> DeployResult:
        stages = [
            {'weight': 1, 'duration': 300},   # 1% for 5 min
            {'weight': 10, 'duration': 600},  # 10% for 10 min
            {'weight': 50, 'duration': 900},  # 50% for 15 min
            {'weight': 100, 'duration': 0},   # 100% final
        ]

        for stage in stages:
            self.set_traffic_weight(new_version, stage['weight'])

            if not self.monitor_period(stage['duration']):
                self.rollback_to_stable()
                return DeployResult(
                    success=False,
                    reason=f"Failed at {stage['weight']}% rollout"
                )

        return DeployResult(success=True)
```

### 3.3 Infrastructure as Code

```hcl
# Terraform module for secure infrastructure

module "vpc" {
  source = "./modules/vpc"

  name = "production"
  cidr = "10.0.0.0/16"

  # Security: Private subnets for workloads
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]

  # Security: Enable flow logs
  enable_flow_logs = true
  flow_log_destination = aws_s3_bucket.logs.arn
}

module "eks" {
  source = "./modules/eks"

  cluster_name = "production"
  vpc_id       = module.vpc.vpc_id
  subnet_ids   = module.vpc.private_subnet_ids

  # Security: Enable audit logging
  cluster_enabled_log_types = ["audit", "api", "authenticator"]

  # Security: Encrypt secrets at rest
  cluster_encryption_config = {
    provider_key_arn = aws_kms_key.eks.arn
    resources        = ["secrets"]
  }

  # Security: Managed node groups in private subnets
  node_groups = {
    main = {
      desired_capacity = 3
      min_capacity     = 2
      max_capacity     = 10
      instance_types   = ["t3.large"]
    }
  }
}

# Security: Secrets management
resource "aws_secretsmanager_secret" "api_keys" {
  name = "production/api-keys"

  # Security: Automatic rotation
  rotation_lambda_arn = aws_lambda_function.secret_rotation.arn
  rotation_rules {
    automatically_after_days = 30
  }
}
```

---

## 4. MLOPS FUNDAMENTALS

### 4.1 ML Pipeline Architecture

```python
# ML PIPELINE STAGES

class MLPipeline:
    """End-to-end ML pipeline."""

    stages = [
        DataIngestion,      # Collect and validate data
        DataValidation,     # Schema, distribution checks
        DataPreprocessing,  # Feature engineering
        ModelTraining,      # Train with tracking
        ModelEvaluation,    # Metrics, comparison
        ModelValidation,    # Bias, fairness, performance
        ModelRegistry,      # Version and store
        ModelDeployment,    # Serve with monitoring
    ]

    def run(self, config: PipelineConfig) -> PipelineResult:
        context = {}

        for stage_class in self.stages:
            stage = stage_class(config)

            # Execute with tracking
            with mlflow.start_run(nested=True):
                result = stage.execute(context)
                mlflow.log_params(result.params)
                mlflow.log_metrics(result.metrics)

            if not result.success:
                return PipelineResult(
                    success=False,
                    stage=stage_class.__name__,
                    error=result.error
                )

            context.update(result.output)

        return PipelineResult(success=True, context=context)
```

### 4.2 Model Registry

```python
class ModelRegistry:
    """Central registry for ML models."""

    def register(self, model: Model, metadata: ModelMetadata) -> ModelVersion:
        """Register a new model version."""
        version = ModelVersion(
            model_id=model.id,
            version=self._next_version(model.id),
            artifact_path=self._store_artifact(model),
            metadata=metadata,
            status=ModelStatus.STAGED,
        )

        # Required checks before registration
        self._validate_model(model)
        self._check_performance(model, metadata)
        self._check_bias(model, metadata)
        self._check_compliance(model, metadata)

        self.db.save(version)
        return version

    def promote(self, version: ModelVersion, to_stage: str) -> None:
        """Promote model to new stage."""
        valid_transitions = {
            'STAGED': ['CANARY', 'ARCHIVED'],
            'CANARY': ['PRODUCTION', 'STAGED', 'ARCHIVED'],
            'PRODUCTION': ['ARCHIVED'],
        }

        if to_stage not in valid_transitions.get(version.status, []):
            raise InvalidTransition(f"{version.status} -> {to_stage}")

        # Archive current production before promoting new
        if to_stage == 'PRODUCTION':
            current = self.get_production(version.model_id)
            if current:
                self.promote(current, 'ARCHIVED')

        version.status = to_stage
        version.promoted_at = datetime.utcnow()
        self.db.save(version)

    def get_production(self, model_id: str) -> Optional[ModelVersion]:
        """Get current production model."""
        return self.db.find_one(
            model_id=model_id,
            status=ModelStatus.PRODUCTION
        )
```

### 4.3 Model Monitoring

```python
class ModelMonitor:
    """Monitor model performance in production."""

    def __init__(self, model_id: str, config: MonitorConfig):
        self.model_id = model_id
        self.config = config
        self.baseline = self._load_baseline()

    def log_prediction(self, input: dict, output: dict, latency_ms: float):
        """Log prediction for monitoring."""
        self.metrics.log({
            'model_id': self.model_id,
            'timestamp': datetime.utcnow(),
            'input_hash': hash_input(input),
            'output': output,
            'latency_ms': latency_ms,
        })

    def check_drift(self) -> DriftReport:
        """Check for data and model drift."""
        recent = self._get_recent_predictions(hours=24)

        return DriftReport(
            data_drift=self._check_data_drift(recent),
            prediction_drift=self._check_prediction_drift(recent),
            performance_drift=self._check_performance_drift(recent),
        )

    def _check_data_drift(self, predictions: list) -> DriftMetric:
        """Check if input distribution has shifted."""
        current_dist = compute_distribution(predictions)
        baseline_dist = self.baseline.input_distribution

        # KL divergence for continuous, chi-square for categorical
        drift_score = compute_drift_score(current_dist, baseline_dist)

        return DriftMetric(
            score=drift_score,
            threshold=self.config.drift_threshold,
            is_drifted=drift_score > self.config.drift_threshold,
        )

    def auto_retrain(self) -> Optional[str]:
        """Trigger automatic retraining if needed."""
        drift = self.check_drift()

        if drift.requires_action:
            job_id = self.training_service.trigger_retrain(
                model_id=self.model_id,
                reason=f"Drift detected: {drift.summary}",
                priority='high' if drift.is_critical else 'normal',
            )
            return job_id

        return None
```

---

## 5. SECURITY PATTERNS

### 5.1 Secret Management

```python
class SecretManager:
    """Unified secret management."""

    def __init__(self, provider: SecretProvider):
        self.provider = provider
        self.cache = TTLCache(maxsize=100, ttl=300)

    def get(self, secret_name: str) -> str:
        """Get secret value with caching."""
        if secret_name in self.cache:
            return self.cache[secret_name]

        value = self.provider.get_secret(secret_name)
        self.cache[secret_name] = value
        return value

    def rotate(self, secret_name: str) -> None:
        """Rotate a secret."""
        # Generate new value
        new_value = self.provider.generate_secret()

        # Update in provider
        self.provider.update_secret(secret_name, new_value)

        # Invalidate cache
        del self.cache[secret_name]

        # Notify dependent services
        self.notify_rotation(secret_name)

# USAGE PATTERN
secrets = SecretManager(provider=AWSSecretsManager())

# In application code - never hardcode
api_key = secrets.get('production/api-key')
db_password = secrets.get('production/db-password')

# NEVER DO THIS:
# api_key = "sk-12345"  # Hardcoded secret!
# db_password = os.environ.get('DB_PASS')  # Env vars less secure
```

### 5.2 Policy as Code

```python
# OPA/Rego style policy definitions

# policies/deployment.rego
package deployment

# Deny deployments without security scan
deny[msg] {
    input.kind == "Deployment"
    not input.metadata.annotations["security.scan/passed"]
    msg = "Deployment must have security scan passed"
}

# Deny containers running as root
deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    not container.securityContext.runAsNonRoot
    msg = sprintf("Container %v must not run as root", [container.name])
}

# Require resource limits
deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    not container.resources.limits
    msg = sprintf("Container %v must have resource limits", [container.name])
}

# Python policy enforcement
class PolicyEnforcer:
    def __init__(self, policies_path: str):
        self.policies = self._load_policies(policies_path)

    def check(self, resource: dict) -> PolicyResult:
        """Check resource against all policies."""
        violations = []

        for policy in self.policies:
            result = policy.evaluate(resource)
            if result.violated:
                violations.append(result)

        return PolicyResult(
            allowed=len(violations) == 0,
            violations=violations,
        )

    def enforce(self, resource: dict) -> dict:
        """Enforce policies, blocking on violation."""
        result = self.check(resource)

        if not result.allowed:
            raise PolicyViolation(result.violations)

        return resource
```

### 5.3 Zero Trust Architecture

```python
# ZERO TRUST IMPLEMENTATION

class ZeroTrustAuthenticator:
    """Never trust, always verify."""

    def authenticate(self, request: Request) -> AuthResult:
        """Multi-layer authentication."""

        # Layer 1: Token validation
        token = self._extract_token(request)
        if not self._validate_token(token):
            return AuthResult(allowed=False, reason="Invalid token")

        # Layer 2: Device validation
        device = self._extract_device_info(request)
        if not self._validate_device(device):
            return AuthResult(allowed=False, reason="Untrusted device")

        # Layer 3: Location/IP validation
        location = self._extract_location(request)
        if not self._validate_location(location, token.user):
            return AuthResult(allowed=False, reason="Suspicious location")

        # Layer 4: Behavior analysis
        if self._is_anomalous_behavior(token.user, request):
            return AuthResult(allowed=False, reason="Anomalous behavior")

        # Layer 5: Just-in-time access
        if not self._check_jit_access(token.user, request.resource):
            return AuthResult(allowed=False, reason="No active access grant")

        return AuthResult(
            allowed=True,
            user=token.user,
            permissions=self._get_scoped_permissions(token.user, request),
        )

# NETWORK ZERO TRUST
class ServiceMesh:
    """mTLS for all service-to-service communication."""

    def configure_sidecar(self, service: str) -> SidecarConfig:
        return SidecarConfig(
            mtls_mode='STRICT',
            allowed_peers=[
                # Only explicitly allowed services
                'frontend' if service == 'api' else None,
                'api' if service in ['user-service', 'order-service'] else None,
            ],
            # Encrypt all traffic
            tls_config=TLSConfig(
                min_version='TLS1.3',
                cipher_suites=['TLS_AES_256_GCM_SHA384'],
            ),
        )
```

---

## 6. OBSERVABILITY

### 6.1 The Three Pillars

```python
class ObservabilityStack:
    """Unified observability: logs, metrics, traces."""

    def __init__(self):
        self.logger = StructuredLogger()
        self.metrics = MetricsCollector()
        self.tracer = DistributedTracer()

    def log(self, level: str, message: str, **context):
        """Structured logging with trace context."""
        trace_id = self.tracer.current_trace_id()
        self.logger.log(level, message, trace_id=trace_id, **context)

    def metric(self, name: str, value: float, tags: dict = None):
        """Emit metric with context."""
        trace_id = self.tracer.current_trace_id()
        self.metrics.emit(name, value, tags={**(tags or {}), 'trace_id': trace_id})

    def span(self, name: str):
        """Create traced span."""
        return self.tracer.start_span(name)

# USAGE
obs = ObservabilityStack()

with obs.span("process_order") as span:
    obs.log("info", "Processing order", order_id=order.id)
    obs.metric("orders.processed", 1, tags={"status": "started"})

    result = process(order)

    obs.metric("orders.processing_time", result.duration_ms)
    obs.log("info", "Order processed", order_id=order.id, status=result.status)
```

### 6.2 SLO/SLI Management

```python
@dataclass
class SLO:
    """Service Level Objective definition."""
    name: str
    description: str
    target: float  # e.g., 0.999 for 99.9%
    window: timedelta  # e.g., 30 days
    sli: 'SLI'

@dataclass
class SLI:
    """Service Level Indicator definition."""
    name: str
    query: str  # Prometheus query
    good_threshold: float

class SLOMonitor:
    """Monitor and report on SLOs."""

    def __init__(self, slos: list[SLO]):
        self.slos = slos

    def get_current_status(self) -> list[SLOStatus]:
        """Get current status of all SLOs."""
        return [self._evaluate_slo(slo) for slo in self.slos]

    def _evaluate_slo(self, slo: SLO) -> SLOStatus:
        """Evaluate single SLO."""
        # Query metrics
        good_events = self._query(slo.sli.query + "[good]")
        total_events = self._query(slo.sli.query + "[total]")

        current = good_events / total_events if total_events > 0 else 1.0

        # Calculate error budget
        error_budget_total = 1 - slo.target
        error_budget_used = 1 - current
        error_budget_remaining = error_budget_total - error_budget_used

        return SLOStatus(
            slo=slo,
            current=current,
            target=slo.target,
            error_budget_remaining=max(0, error_budget_remaining),
            is_meeting=current >= slo.target,
        )

# EXAMPLE SLOs
slos = [
    SLO(
        name="API Availability",
        description="API responds successfully",
        target=0.999,
        window=timedelta(days=30),
        sli=SLI(
            name="success_rate",
            query="sum(rate(http_requests_total{status!~'5..'}[5m])) / sum(rate(http_requests_total[5m]))",
            good_threshold=0.999,
        ),
    ),
    SLO(
        name="API Latency",
        description="P99 latency under 500ms",
        target=0.99,
        window=timedelta(days=30),
        sli=SLI(
            name="latency_p99",
            query="histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))",
            good_threshold=0.5,
        ),
    ),
]
```

---

## 7. INCIDENT MANAGEMENT

### 7.1 Incident Response

```python
class IncidentManager:
    """Structured incident response."""

    def create_incident(self, alert: Alert) -> Incident:
        """Create incident from alert."""
        incident = Incident(
            id=generate_id(),
            severity=self._determine_severity(alert),
            status=IncidentStatus.DETECTED,
            timeline=[
                TimelineEntry(
                    timestamp=datetime.utcnow(),
                    event="Incident detected",
                    details=alert.to_dict(),
                ),
            ],
        )

        # Auto-assign based on runbook
        if runbook := self._find_runbook(alert):
            incident.runbook = runbook
            incident.assigned_to = runbook.default_assignee

        # Notify on-call
        self._notify_oncall(incident)

        return incident

    def update_status(self, incident_id: str, status: IncidentStatus,
                      notes: str = None) -> None:
        """Update incident status."""
        incident = self.get(incident_id)
        incident.status = status
        incident.timeline.append(
            TimelineEntry(
                timestamp=datetime.utcnow(),
                event=f"Status changed to {status}",
                details=notes,
            )
        )

        if status == IncidentStatus.RESOLVED:
            self._schedule_postmortem(incident)

    def _determine_severity(self, alert: Alert) -> Severity:
        """Determine incident severity."""
        if alert.affects_revenue or alert.affects_data:
            return Severity.CRITICAL
        if alert.affects_users:
            return Severity.HIGH
        if alert.is_degradation:
            return Severity.MEDIUM
        return Severity.LOW
```

### 7.2 Postmortem Process

```markdown
# POSTMORTEM TEMPLATE

## Incident: [Title]
**Date**: [Date]
**Duration**: [Total time]
**Severity**: [P0/P1/P2/P3]
**Author**: [Name]

## Summary
One paragraph describing what happened.

## Impact
- Users affected: [number]
- Revenue impact: [estimate]
- Data impact: [description]

## Timeline (all times UTC)
| Time | Event |
|------|-------|
| HH:MM | [Event] |

## Root Cause
[Technical explanation of why this happened]

## Resolution
[What was done to fix it]

## Lessons Learned
### What went well
- [Point]

### What went poorly
- [Point]

### Where we got lucky
- [Point]

## Action Items
| Action | Owner | Due Date | Status |
|--------|-------|----------|--------|
| [Action] | [Name] | [Date] | [Open/Done] |

## Detection
How was this detected? How could we detect it faster?

## Prevention
What would prevent this class of incident?
```

---

## 8. IMPLEMENTATION CHECKLIST

### DevSecOps:
- [ ] Secret scanning in pre-commit
- [ ] SAST in every PR
- [ ] Dependency scanning daily
- [ ] Container scanning before deploy
- [ ] Policy enforcement at deploy
- [ ] Runtime protection enabled

### MLOps:
- [ ] Data validation in pipeline
- [ ] Model versioning and registry
- [ ] A/B testing infrastructure
- [ ] Model monitoring for drift
- [ ] Automated retraining triggers
- [ ] Model rollback capability

### Observability:
- [ ] Structured logging everywhere
- [ ] Metrics for business and technical
- [ ] Distributed tracing
- [ ] SLOs defined and monitored
- [ ] Alerting based on SLOs
- [ ] Dashboards for key metrics

---

**Remember**: Security and reliability are not features—they're properties of how you build. Automate the boring parts, monitor the critical parts, and always have a rollback plan.

Secure by default. Observable by design. Reliable through automation.
