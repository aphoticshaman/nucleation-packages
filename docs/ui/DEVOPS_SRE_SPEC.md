# LatticeForge DevOps/SRE Specification

## Document Purpose

This specification provides comprehensive guidance for DevOps engineers and Site Reliability Engineers (SREs) working on LatticeForge. It covers CI/CD pipelines, infrastructure as code, deployment strategies, monitoring, alerting, incident management, and operational excellence. Use this as the authoritative reference for building and operating reliable, scalable infrastructure.

---

## 1. Infrastructure Overview

### 1.1 Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GLOBAL EDGE                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │ Cloudflare  │  │ Cloudflare  │  │ Cloudflare  │  │ Cloudflare  │       │
│  │   (SFO)     │  │   (NYC)     │  │   (LHR)     │  │   (SIN)     │       │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘       │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
┌─────────────────────────────────────┴───────────────────────────────────────┐
│                           PRIMARY REGION (us-west-2)                        │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Kubernetes Cluster                          │   │
│  │                                                                      │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │   │
│  │  │  API Pods   │  │  Web Pods   │  │ Worker Pods │                 │   │
│  │  │  (3-10)     │  │  (2-6)      │  │  (2-20)     │                 │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                 │   │
│  │                                                                      │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │   │
│  │  │ Inference   │  │  Redis      │  │  RabbitMQ   │                 │   │
│  │  │ Service     │  │  (Cluster)  │  │  (Cluster)  │                 │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                 │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       Managed Services                              │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │   │
│  │  │ PostgreSQL  │  │     S3      │  │  External   │                 │   │
│  │  │ (RDS)       │  │  (Storage)  │  │  LLM APIs   │                 │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ Async Replication
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DR REGION (us-east-1)                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  PostgreSQL Replica (read-only) │ S3 Cross-Region Replication      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| CDN/WAF | Cloudflare | Edge caching, DDoS protection, WAF |
| Container Orchestration | Kubernetes (EKS) | Container scheduling, scaling |
| Service Mesh | Linkerd | mTLS, observability, traffic management |
| Database | PostgreSQL (RDS) | Primary data store |
| Cache | Redis (ElastiCache) | Session cache, rate limiting, pub/sub |
| Queue | RabbitMQ | Background job processing |
| Object Storage | S3 | PDFs, exports, static assets |
| Secrets | HashiCorp Vault | Secret management |
| IaC | Terraform + Pulumi | Infrastructure provisioning |
| CI/CD | GitHub Actions + ArgoCD | Build, test, deploy |
| Monitoring | Prometheus + Grafana | Metrics, dashboards |
| Logging | Loki + Promtail | Log aggregation |
| Tracing | Jaeger | Distributed tracing |
| Alerting | Alertmanager + PagerDuty | Alert routing, on-call |

### 1.3 Environment Matrix

| Environment | Purpose | Cluster | Database | Deployment |
|-------------|---------|---------|----------|------------|
| Development | Local development | Minikube/k3d | Local PostgreSQL | Manual |
| Staging | Pre-production testing | EKS (shared) | RDS (small) | On PR merge |
| Production | Live traffic | EKS (dedicated) | RDS (Multi-AZ) | On release |
| DR | Disaster recovery | EKS (standby) | RDS (replica) | Failover only |

---

## 2. CI/CD Pipeline

### 2.1 Pipeline Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                           CI/CD Pipeline                                  │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐ │
│  │  Push   │───▶│  Build  │───▶│  Test   │───▶│  Scan   │───▶│ Publish │ │
│  │         │    │         │    │         │    │         │    │         │ │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘ │
│       │                                                            │      │
│       │                                                            ▼      │
│       │         ┌─────────────────────────────────────────────────────┐  │
│       │         │                    ArgoCD                          │  │
│       │         │  ┌─────────┐    ┌─────────┐    ┌─────────┐        │  │
│       │         │  │ Staging │───▶│  Prod   │───▶│   DR    │        │  │
│       │         │  │  Sync   │    │  Sync   │    │  Sync   │        │  │
│       │         │  └─────────┘    └─────────┘    └─────────┘        │  │
│       │         └─────────────────────────────────────────────────────┘  │
│       │                                                                   │
│       ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                      PR Preview Environments                        │ │
│  │        latticeforge-pr-123.preview.latticeforge.io                  │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Build Pipeline

**GitHub Actions Workflow:**
```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

env:
  CARGO_TERM_COLOR: always
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt, clippy

      - name: Cache cargo
        uses: Swatinem/rust-cache@v2

      - name: Format check
        run: cargo fmt --all -- --check

      - name: Clippy
        run: cargo clippy --all-targets --all-features -- -D warnings

  test:
    runs-on: ubuntu-latest
    needs: lint
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_PASSWORD: test
          POSTGRES_DB: latticeforge_test
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

      redis:
        image: redis:7
        ports:
          - 6379:6379

    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Cache cargo
        uses: Swatinem/rust-cache@v2

      - name: Run tests
        run: cargo test --all-features
        env:
          DATABASE_URL: postgres://postgres:test@localhost/latticeforge_test
          REDIS_URL: redis://localhost:6379

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: ./target/coverage/lcov.info

  security:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4

      - name: Dependency audit
        run: cargo audit

      - name: SAST scan
        uses: github/codeql-action/analyze@v3

  build:
    runs-on: ubuntu-latest
    needs: [test, security]
    permissions:
      contents: read
      packages: write

    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to GHCR
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=sha,prefix=
            type=ref,event=branch
            type=ref,event=pr

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Scan image
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-results.sarif'

      - name: Upload scan results
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: 'trivy-results.sarif'

  deploy-staging:
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/main'
    environment: staging

    steps:
      - uses: actions/checkout@v4

      - name: Update image tag
        run: |
          cd deploy/overlays/staging
          kustomize edit set image api=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}

      - name: Commit and push
        run: |
          git config user.name "github-actions"
          git config user.email "github-actions@github.com"
          git add .
          git commit -m "Deploy ${{ github.sha }} to staging"
          git push
```

### 2.3 Deployment Pipeline

**ArgoCD Application:**
```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: latticeforge-api
  namespace: argocd
spec:
  project: latticeforge

  source:
    repoURL: https://github.com/latticeforge/platform
    targetRevision: main
    path: deploy/overlays/production

  destination:
    server: https://kubernetes.default.svc
    namespace: latticeforge

  syncPolicy:
    automated:
      prune: true
      selfHeal: true
      allowEmpty: false
    syncOptions:
      - CreateNamespace=true
      - PrunePropagationPolicy=foreground
      - PruneLast=true

    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m

  ignoreDifferences:
    - group: apps
      kind: Deployment
      jsonPointers:
        - /spec/replicas  # HPA manages this
```

### 2.4 Rollback Procedures

**Automatic Rollback (ArgoCD):**
```yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: api
spec:
  replicas: 5
  strategy:
    canary:
      steps:
        - setWeight: 20
        - pause: {duration: 5m}
        - setWeight: 40
        - pause: {duration: 5m}
        - setWeight: 60
        - pause: {duration: 5m}
        - setWeight: 80
        - pause: {duration: 5m}

      analysis:
        templates:
          - templateName: success-rate
        startingStep: 2
        args:
          - name: service-name
            value: api

      # Automatic rollback on failure
      rollbackWindow:
        revisions: 3

---
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: success-rate
spec:
  args:
    - name: service-name
  metrics:
    - name: success-rate
      interval: 1m
      successCondition: result[0] >= 0.99
      failureLimit: 3
      provider:
        prometheus:
          address: http://prometheus:9090
          query: |
            sum(rate(http_requests_total{service="{{args.service-name}}",status=~"2.."}[5m]))
            /
            sum(rate(http_requests_total{service="{{args.service-name}}"}[5m]))
```

**Manual Rollback:**
```bash
# ArgoCD rollback
argocd app rollback latticeforge-api --revision <previous-revision>

# Or via kubectl
kubectl rollout undo deployment/api -n latticeforge

# Database rollback (if needed)
sqlx migrate revert --target-version <version>
```

---

## 3. Infrastructure as Code

### 3.1 Terraform Structure

```
infrastructure/
├── modules/
│   ├── vpc/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   ├── eks/
│   ├── rds/
│   ├── elasticache/
│   ├── s3/
│   └── cloudflare/
├── environments/
│   ├── staging/
│   │   ├── main.tf
│   │   ├── terraform.tfvars
│   │   └── backend.tf
│   └── production/
│       ├── main.tf
│       ├── terraform.tfvars
│       └── backend.tf
└── global/
    ├── iam/
    └── dns/
```

### 3.2 Core Infrastructure

**VPC Module:**
```hcl
# modules/vpc/main.tf

resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name        = "${var.environment}-vpc"
    Environment = var.environment
    Terraform   = "true"
  }
}

resource "aws_subnet" "public" {
  count = length(var.availability_zones)

  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 4, count.index)
  availability_zone       = var.availability_zones[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name                                        = "${var.environment}-public-${count.index}"
    "kubernetes.io/role/elb"                    = "1"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }
}

resource "aws_subnet" "private" {
  count = length(var.availability_zones)

  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 4, count.index + 4)
  availability_zone = var.availability_zones[count.index]

  tags = {
    Name                                        = "${var.environment}-private-${count.index}"
    "kubernetes.io/role/internal-elb"           = "1"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }
}

resource "aws_subnet" "database" {
  count = length(var.availability_zones)

  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 4, count.index + 8)
  availability_zone = var.availability_zones[count.index]

  tags = {
    Name = "${var.environment}-database-${count.index}"
  }
}

# NAT Gateways for private subnet internet access
resource "aws_nat_gateway" "main" {
  count = length(var.availability_zones)

  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id

  tags = {
    Name = "${var.environment}-nat-${count.index}"
  }
}
```

**EKS Module:**
```hcl
# modules/eks/main.tf

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = var.cluster_name
  cluster_version = "1.29"

  vpc_id     = var.vpc_id
  subnet_ids = var.private_subnet_ids

  cluster_endpoint_public_access  = true
  cluster_endpoint_private_access = true

  eks_managed_node_groups = {
    general = {
      name           = "general"
      instance_types = ["m6i.large", "m6i.xlarge"]

      min_size     = 2
      max_size     = 10
      desired_size = 3

      labels = {
        role = "general"
      }

      taints = []
    }

    workers = {
      name           = "workers"
      instance_types = ["c6i.2xlarge", "c6i.4xlarge"]

      min_size     = 2
      max_size     = 20
      desired_size = 3

      labels = {
        role = "worker"
      }

      taints = [{
        key    = "dedicated"
        value  = "worker"
        effect = "NO_SCHEDULE"
      }]
    }

    inference = {
      name           = "inference"
      instance_types = ["g5.xlarge", "g5.2xlarge"]
      ami_type       = "AL2_x86_64_GPU"

      min_size     = 0
      max_size     = 4
      desired_size = 1

      labels = {
        role                          = "inference"
        "nvidia.com/gpu.present"      = "true"
      }

      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }
  }

  # Enable IRSA
  enable_irsa = true

  # Cluster add-ons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent              = true
      service_account_role_arn = module.ebs_csi_irsa.iam_role_arn
    }
  }

  tags = {
    Environment = var.environment
    Terraform   = "true"
  }
}
```

**RDS Module:**
```hcl
# modules/rds/main.tf

resource "aws_db_instance" "main" {
  identifier = "${var.environment}-postgres"

  engine               = "postgres"
  engine_version       = "16.1"
  instance_class       = var.instance_class
  allocated_storage    = var.allocated_storage
  max_allocated_storage = var.max_allocated_storage

  db_name  = "latticeforge"
  username = "latticeforge"
  password = var.db_password

  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name

  # High availability
  multi_az = var.environment == "production"

  # Backup configuration
  backup_retention_period = var.environment == "production" ? 30 : 7
  backup_window           = "03:00-04:00"
  maintenance_window      = "Mon:04:00-Mon:05:00"

  # Encryption
  storage_encrypted = true
  kms_key_id        = var.kms_key_arn

  # Performance
  performance_insights_enabled = true
  monitoring_interval          = 60
  monitoring_role_arn          = aws_iam_role.rds_monitoring.arn

  # Security
  publicly_accessible    = false
  deletion_protection    = var.environment == "production"
  skip_final_snapshot    = var.environment != "production"
  final_snapshot_identifier = var.environment == "production" ? "${var.environment}-final-snapshot" : null

  # Parameters
  parameter_group_name = aws_db_parameter_group.main.name

  tags = {
    Environment = var.environment
    Terraform   = "true"
  }
}

resource "aws_db_parameter_group" "main" {
  name   = "${var.environment}-postgres16"
  family = "postgres16"

  parameter {
    name  = "log_statement"
    value = "ddl"
  }

  parameter {
    name  = "log_min_duration_statement"
    value = "1000"  # Log queries > 1s
  }

  parameter {
    name  = "shared_preload_libraries"
    value = "pg_stat_statements"
  }
}
```

### 3.3 Kubernetes Manifests

**Base Deployment:**
```yaml
# deploy/base/api/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
  labels:
    app: api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api
  template:
    metadata:
      labels:
        app: api
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: api
      securityContext:
        runAsNonRoot: true
        runAsUser: 65534
        fsGroup: 65534
        seccompProfile:
          type: RuntimeDefault

      containers:
        - name: api
          image: ghcr.io/latticeforge/api:latest
          ports:
            - containerPort: 8080
              name: http
            - containerPort: 9090
              name: metrics

          env:
            - name: RUST_LOG
              value: "info,tower_http=debug"
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: database-credentials
                  key: url
            - name: REDIS_URL
              valueFrom:
                secretKeyRef:
                  name: redis-credentials
                  key: url

          resources:
            requests:
              cpu: 500m
              memory: 512Mi
            limits:
              cpu: 2000m
              memory: 2Gi

          livenessProbe:
            httpGet:
              path: /health/live
              port: http
            initialDelaySeconds: 10
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3

          readinessProbe:
            httpGet:
              path: /health/ready
              port: http
            initialDelaySeconds: 5
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 3

          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop:
                - ALL

          volumeMounts:
            - name: tmp
              mountPath: /tmp

      volumes:
        - name: tmp
          emptyDir: {}

      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchLabels:
                    app: api
                topologyKey: kubernetes.io/hostname

      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: topology.kubernetes.io/zone
          whenUnsatisfiable: ScheduleAnyway
          labelSelector:
            matchLabels:
              app: api
```

**HPA Configuration:**
```yaml
# deploy/base/api/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second
        target:
          type: AverageValue
          averageValue: "1000"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Percent
          value: 100
          periodSeconds: 60
        - type: Pods
          value: 4
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 25
          periodSeconds: 60
```

---

## 4. Monitoring and Observability

### 4.1 Metrics Collection

**Prometheus Configuration:**
```yaml
# monitoring/prometheus/values.yaml
prometheus:
  prometheusSpec:
    retention: 30d
    retentionSize: 50GB

    storageSpec:
      volumeClaimTemplate:
        spec:
          storageClassName: gp3
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 100Gi

    serviceMonitorSelector:
      matchLabels:
        release: prometheus

    additionalScrapeConfigs:
      - job_name: 'kubernetes-pods'
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: true
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
            action: replace
            target_label: __metrics_path__
            regex: (.+)
          - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
            action: replace
            regex: ([^:]+)(?::\d+)?;(\d+)
            replacement: $1:$2
            target_label: __address__

    resources:
      requests:
        cpu: 500m
        memory: 2Gi
      limits:
        cpu: 2000m
        memory: 4Gi
```

**Service Monitor:**
```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: api
  labels:
    release: prometheus
spec:
  selector:
    matchLabels:
      app: api
  endpoints:
    - port: metrics
      interval: 15s
      path: /metrics
  namespaceSelector:
    matchNames:
      - latticeforge
```

### 4.2 Key Metrics

**Application Metrics (RED Method):**
```rust
// Instrument handler
#[derive(Clone)]
struct MetricsLayer {
    request_count: Counter,
    request_duration: Histogram,
    active_requests: Gauge,
}

impl MetricsLayer {
    fn new(registry: &Registry) -> Self {
        Self {
            request_count: Counter::new(
                "http_requests_total",
                "Total HTTP requests",
            ).with_label_names(&["method", "path", "status"]),

            request_duration: Histogram::new(
                "http_request_duration_seconds",
                "HTTP request duration in seconds",
            ).with_label_names(&["method", "path"]),

            active_requests: Gauge::new(
                "http_requests_active",
                "Currently active HTTP requests",
            ),
        }
    }
}
```

**Infrastructure Metrics:**
| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| container_cpu_usage_seconds_total | CPU usage | >80% for 5m |
| container_memory_working_set_bytes | Memory usage | >85% for 5m |
| kube_pod_status_phase | Pod phase | phase != Running |
| pg_stat_activity_count | DB connections | >80% of max |
| redis_connected_clients | Redis connections | >1000 |

**Business Metrics:**
| Metric | Description | Purpose |
|--------|-------------|---------|
| sources_processed_total | Sources processed | Throughput |
| synthesis_generation_duration_seconds | Synthesis time | Performance |
| insights_generated_total | Insights generated | Feature usage |
| active_users | DAU/WAU/MAU | Engagement |

### 4.3 Dashboards

**Service Overview Dashboard:**
```json
{
  "dashboard": {
    "title": "LatticeForge - Service Overview",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total[5m])) by (service)",
            "legendFormat": "{{service}}"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{status=~\"5..\"}[5m])) / sum(rate(http_requests_total[5m])) * 100",
            "legendFormat": "Error %"
          }
        ]
      },
      {
        "title": "P99 Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service))",
            "legendFormat": "{{service}} p99"
          }
        ]
      },
      {
        "title": "Pod Status",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(kube_pod_status_phase{namespace=\"latticeforge\", phase=\"Running\"})",
            "legendFormat": "Running"
          }
        ]
      }
    ]
  }
}
```

### 4.4 Distributed Tracing

**Jaeger Configuration:**
```yaml
# monitoring/jaeger/values.yaml
jaeger:
  collector:
    service:
      type: ClusterIP
    resources:
      requests:
        cpu: 200m
        memory: 256Mi

  query:
    service:
      type: ClusterIP
    ingress:
      enabled: true
      hosts:
        - jaeger.internal.latticeforge.io

  storage:
    type: elasticsearch
    elasticsearch:
      host: elasticsearch
      port: 9200

  sampling:
    default_strategy:
      service_strategies:
        - service: api
          type: probabilistic
          param: 0.1  # Sample 10%
        - service: worker
          type: probabilistic
          param: 0.05  # Sample 5%
```

**Trace Instrumentation:**
```rust
use tracing::{instrument, info_span, Span};
use tracing_opentelemetry::OpenTelemetrySpanExt;

#[instrument(skip(db, auth), fields(user_id = %auth.user_id))]
async fn get_stream(
    db: &Database,
    auth: &AuthContext,
    stream_id: &str,
) -> Result<Stream> {
    let span = Span::current();
    span.set_attribute("stream_id", stream_id);

    let stream = db.get_stream(stream_id).await?;

    span.set_attribute("source_count", stream.source_count);

    Ok(stream)
}
```

---

## 5. Alerting

### 5.1 Alert Rules

```yaml
# monitoring/alertmanager/rules.yaml
groups:
  - name: latticeforge.availability
    rules:
      - alert: ServiceDown
        expr: up{job="api"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ $labels.service }} is down"
          description: "{{ $labels.service }} has been down for more than 1 minute"
          runbook_url: "https://runbooks.latticeforge.io/service-down"

      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m])) by (service)
          /
          sum(rate(http_requests_total[5m])) by (service)
          > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate for {{ $labels.service }}"
          description: "Error rate is {{ $value | humanizePercentage }} (>5%)"

      - alert: HighErrorRateCritical
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m])) by (service)
          /
          sum(rate(http_requests_total[5m])) by (service)
          > 0.10
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Critical error rate for {{ $labels.service }}"
          description: "Error rate is {{ $value | humanizePercentage }} (>10%)"

  - name: latticeforge.latency
    rules:
      - alert: HighLatency
        expr: |
          histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service))
          > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High P99 latency for {{ $labels.service }}"
          description: "P99 latency is {{ $value | humanizeDuration }}"

  - name: latticeforge.resources
    rules:
      - alert: HighCPUUsage
        expr: |
          sum(rate(container_cpu_usage_seconds_total{namespace="latticeforge"}[5m])) by (pod)
          /
          sum(container_spec_cpu_quota{namespace="latticeforge"}/container_spec_cpu_period{namespace="latticeforge"}) by (pod)
          > 0.8
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage for {{ $labels.pod }}"

      - alert: HighMemoryUsage
        expr: |
          sum(container_memory_working_set_bytes{namespace="latticeforge"}) by (pod)
          /
          sum(container_spec_memory_limit_bytes{namespace="latticeforge"}) by (pod)
          > 0.85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage for {{ $labels.pod }}"

      - alert: PodCrashLooping
        expr: |
          rate(kube_pod_container_status_restarts_total{namespace="latticeforge"}[15m]) > 0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Pod {{ $labels.pod }} is crash looping"

  - name: latticeforge.database
    rules:
      - alert: DatabaseConnectionPoolExhausted
        expr: |
          pg_stat_activity_count{datname="latticeforge"}
          /
          pg_settings_max_connections
          > 0.8
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Database connection pool near exhaustion"
          description: "{{ $value | humanizePercentage }} of connections used"

      - alert: SlowQueries
        expr: |
          rate(pg_stat_statements_seconds_total[5m])
          /
          rate(pg_stat_statements_calls_total[5m])
          > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Slow database queries detected"
```

### 5.2 Alertmanager Configuration

```yaml
# monitoring/alertmanager/config.yaml
global:
  resolve_timeout: 5m
  slack_api_url: '${SLACK_WEBHOOK_URL}'

route:
  receiver: 'default'
  group_by: ['alertname', 'service']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h

  routes:
    - receiver: 'critical-pagerduty'
      match:
        severity: critical
      continue: true

    - receiver: 'slack-alerts'
      match_re:
        severity: warning|critical

receivers:
  - name: 'default'
    webhook_configs:
      - url: 'http://alertmanager-webhook:9093/webhook'

  - name: 'critical-pagerduty'
    pagerduty_configs:
      - service_key: '${PAGERDUTY_SERVICE_KEY}'
        severity: critical
        description: '{{ .CommonAnnotations.summary }}'
        details:
          firing: '{{ .Alerts.Firing | len }}'
          resolved: '{{ .Alerts.Resolved | len }}'

  - name: 'slack-alerts'
    slack_configs:
      - channel: '#alerts'
        send_resolved: true
        title: '{{ if eq .Status "firing" }}🔥{{ else }}✅{{ end }} {{ .CommonAnnotations.summary }}'
        text: '{{ .CommonAnnotations.description }}'
        actions:
          - type: button
            text: 'Runbook'
            url: '{{ .CommonAnnotations.runbook_url }}'
          - type: button
            text: 'Dashboard'
            url: 'https://grafana.latticeforge.io/d/service-overview'

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'service']
```

### 5.3 On-Call Schedule

**PagerDuty Configuration:**
| Tier | Response Time | Escalation | Coverage |
|------|---------------|------------|----------|
| Primary | 5 minutes | 15 min to secondary | 24/7 |
| Secondary | 15 minutes | 30 min to manager | 24/7 |
| Manager | 30 minutes | - | Business hours |

**Escalation Policy:**
1. Primary on-call: 5 min ACK required
2. Secondary on-call: 15 min if no ACK
3. Engineering manager: 30 min if still unresolved
4. VP Engineering: 1 hour for P1 incidents

---

## 6. Disaster Recovery

### 6.1 Backup Strategy

**Database Backups:**
```yaml
# Automated RDS snapshots
resource "aws_db_instance" "main" {
  # ... other config

  backup_retention_period = 30
  backup_window          = "03:00-04:00"

  # Enable automated backups
  copy_tags_to_snapshot = true
}

# Point-in-time recovery enabled by default
# Manual snapshot before major changes
```

**S3 Backups:**
```yaml
resource "aws_s3_bucket_replication_configuration" "main" {
  bucket = aws_s3_bucket.main.id

  rule {
    id     = "dr-replication"
    status = "Enabled"

    destination {
      bucket        = "arn:aws:s3:::${var.dr_bucket_name}"
      storage_class = "STANDARD_IA"

      encryption_configuration {
        replica_kms_key_id = var.dr_kms_key_arn
      }
    }
  }
}
```

### 6.2 Recovery Procedures

**RTO/RPO Targets:**
| Component | RTO | RPO | Strategy |
|-----------|-----|-----|----------|
| API | 15 min | 0 | Multi-AZ, auto-failover |
| Database | 30 min | 5 min | Multi-AZ RDS, PITR |
| Object Storage | 1 hour | 0 | Cross-region replication |
| Full DR | 4 hours | 1 hour | Warm standby region |

**Database Recovery Runbook:**
```bash
#!/bin/bash
# recover-database.sh

# 1. Identify recovery point
aws rds describe-db-cluster-snapshots \
  --db-cluster-identifier latticeforge-production \
  --query 'DBClusterSnapshots[*].[DBClusterSnapshotIdentifier,SnapshotCreateTime]'

# 2. Restore from snapshot
aws rds restore-db-cluster-from-snapshot \
  --db-cluster-identifier latticeforge-recovery \
  --snapshot-identifier <snapshot-id> \
  --vpc-security-group-ids <security-group-id> \
  --db-subnet-group-name latticeforge-db-subnet

# OR Point-in-time recovery
aws rds restore-db-cluster-to-point-in-time \
  --source-db-cluster-identifier latticeforge-production \
  --target-db-cluster-identifier latticeforge-recovery \
  --restore-to-time "2024-03-15T10:30:00Z" \
  --use-latest-restorable-time

# 3. Update application config
kubectl set env deployment/api DATABASE_URL=<new-connection-string>

# 4. Verify data integrity
psql $NEW_DATABASE_URL -c "SELECT COUNT(*) FROM streams;"
```

### 6.3 DR Drills

**Quarterly DR Drill Checklist:**
```markdown
## Disaster Recovery Drill

### Pre-Drill
- [ ] Schedule drill window (off-peak)
- [ ] Notify stakeholders
- [ ] Prepare rollback plan
- [ ] Verify DR region readiness

### Drill Execution
- [ ] Simulate primary region failure
- [ ] Trigger DNS failover to DR region
- [ ] Verify database replica promotion
- [ ] Test all critical user flows
- [ ] Measure actual RTO achieved

### Post-Drill
- [ ] Document issues encountered
- [ ] Calculate actual RTO/RPO
- [ ] Update runbooks with learnings
- [ ] Schedule remediation for gaps
- [ ] Report to leadership

### Success Criteria
- [ ] RTO < 4 hours
- [ ] RPO < 1 hour
- [ ] All critical flows functional
- [ ] No data loss detected
```

---

## 7. Operational Runbooks

### 7.1 Common Incidents

**High Error Rate Runbook:**
```markdown
## Incident: High Error Rate

### Detection
Alert: HighErrorRate or HighErrorRateCritical

### Initial Assessment (2 min)
1. Check error rate graph: `sum(rate(http_requests_total{status=~"5.."}[5m]))`
2. Identify affected endpoints: `topk(10, sum by (path) (rate(http_requests_total{status=~"5.."}[5m])))`
3. Check recent deployments: `kubectl rollout history deployment/api`

### Diagnosis (5 min)
1. Check application logs:
   ```bash
   kubectl logs -l app=api --tail=500 | grep -i error
   ```
2. Check external dependencies:
   - Database: `pg_isready -h $DB_HOST`
   - Redis: `redis-cli ping`
   - External APIs: Check dashboard

3. Check resource exhaustion:
   - CPU/Memory: Grafana → Resource dashboard
   - Connection pools: `pg_stat_activity` count

### Remediation
**If recent deployment:**
```bash
kubectl rollout undo deployment/api
```

**If database issue:**
- Scale up RDS if connection exhaustion
- Check for slow queries
- Failover to replica if primary unhealthy

**If external API issue:**
- Enable circuit breaker
- Fail gracefully with cached data

### Escalation
- After 15 min: Page secondary on-call
- After 30 min: Page engineering manager
- Customer impact: Notify support team
```

### 7.2 Scaling Runbook

**Manual Scaling:**
```bash
# Scale API deployment
kubectl scale deployment/api --replicas=10

# Scale worker deployment
kubectl scale deployment/worker --replicas=20

# Scale database (RDS)
# Note: This causes brief downtime
aws rds modify-db-instance \
  --db-instance-identifier latticeforge-production \
  --db-instance-class db.r6g.2xlarge \
  --apply-immediately
```

**Capacity Planning Thresholds:**
| Resource | Scale Up Threshold | Scale Down Threshold |
|----------|-------------------|---------------------|
| API Pods | 70% CPU sustained 5m | 30% CPU sustained 30m |
| Worker Pods | Queue depth > 1000 | Queue depth < 100 |
| Database | 80% connections | 40% connections |

### 7.3 Database Maintenance

**Index Maintenance:**
```sql
-- Find bloated indexes
SELECT
    schemaname || '.' || indexrelname AS index,
    pg_size_pretty(pg_relation_size(indexrelid)) AS size,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE idx_scan = 0
AND pg_relation_size(indexrelid) > 10000000
ORDER BY pg_relation_size(indexrelid) DESC;

-- Reindex without locking (PostgreSQL 12+)
REINDEX INDEX CONCURRENTLY idx_name;

-- Vacuum analyze
VACUUM ANALYZE table_name;
```

---

## 8. Cost Management

### 8.1 Cost Monitoring

**AWS Cost Tags:**
```hcl
locals {
  common_tags = {
    Environment = var.environment
    Team        = "platform"
    Service     = "latticeforge"
    CostCenter  = "engineering"
    Terraform   = "true"
  }
}
```

**Cost Alerts:**
```yaml
resource "aws_budgets_budget" "monthly" {
  name              = "latticeforge-monthly"
  budget_type       = "COST"
  limit_amount      = "10000"
  limit_unit        = "USD"
  time_unit         = "MONTHLY"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = ["alerts@latticeforge.io"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = ["alerts@latticeforge.io"]
  }
}
```

### 8.2 Cost Optimization

**Recommendations:**
| Area | Current | Optimization | Savings |
|------|---------|--------------|---------|
| EC2 | On-demand | Savings Plans | 30-40% |
| RDS | On-demand | Reserved Instance | 40% |
| S3 | Standard | Intelligent Tiering | 20% |
| NAT | Per-GB pricing | NAT Instance (staging) | 50% |

**Right-sizing Schedule:**
- Weekly: Review underutilized pods
- Monthly: Review RDS instance size
- Quarterly: Review reserved capacity

---

## 9. Security Operations

### 9.1 Security Patching

**Automated Patching:**
```yaml
# kube-system/kured.yaml (Kubernetes node reboot daemon)
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: kured
  namespace: kube-system
spec:
  template:
    spec:
      containers:
        - name: kured
          image: weaveworks/kured:1.13.1
          args:
            - "--reboot-sentinel=/var/run/reboot-required"
            - "--start-time=01:00"
            - "--end-time=05:00"
            - "--time-zone=UTC"
            - "--period=1h"
            - "--slack-hook-url=${SLACK_WEBHOOK_URL}"
```

**Container Image Updates:**
```yaml
# Use Dependabot for automated PRs
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
    reviewers:
      - "platform-team"
```

### 9.2 Access Management

**kubectl Access:**
```yaml
# RBAC for developers
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: latticeforge
  name: developer
rules:
  - apiGroups: [""]
    resources: ["pods", "pods/log", "pods/exec"]
    verbs: ["get", "list", "watch"]
  - apiGroups: [""]
    resources: ["services", "configmaps"]
    verbs: ["get", "list"]
  - apiGroups: ["apps"]
    resources: ["deployments", "replicasets"]
    verbs: ["get", "list", "watch"]
```

**Database Access:**
- No direct production access
- Read replica for debugging
- Query audit logging enabled
- Time-limited access via Vault

---

## 10. SLOs and Error Budgets

### 10.1 Service Level Objectives

| Service | SLI | SLO | Error Budget (30 day) |
|---------|-----|-----|----------------------|
| API | Availability | 99.9% | 43.2 minutes |
| API | Latency (p99) | < 500ms | - |
| Synthesis | Success rate | 99.5% | 3.6 hours |
| Processing | Queue latency | < 5 min | - |

### 10.2 Error Budget Policy

**Budget Consumption Actions:**
| Remaining Budget | Action |
|-----------------|--------|
| >50% | Normal development velocity |
| 25-50% | Prioritize reliability work |
| 10-25% | Freeze non-critical deploys |
| <10% | Emergency reliability focus |

**Error Budget Burn Rate Alerts:**
```yaml
- alert: HighErrorBudgetBurn
  expr: |
    (
      1 - (
        sum(rate(http_requests_total{status!~"5.."}[1h]))
        /
        sum(rate(http_requests_total[1h]))
      )
    ) > (1 - 0.999) * 14.4
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Error budget burning fast"
    description: "At current rate, error budget exhausted in {{ $value }} days"
```

---

*Operational excellence is a continuous journey. This document should evolve as we learn from incidents, optimize performance, and scale our infrastructure. Review and update quarterly.*
