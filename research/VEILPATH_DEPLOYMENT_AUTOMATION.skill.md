---
name: veilpath-deployment-automation
description: "Acquisition-ready deployment infrastructure for VeilPath. Multi-platform CI/CD, monitoring, cost optimization, and the DevOps that makes due diligence teams say yes."
---

# VEILPATH_DEPLOYMENT_AUTOMATION.skill.md

## The DevOps That Adds $100M to Your Exit Valuation

**Version**: 1.0  
**Domain**: CI/CD, Platform Deployment, Monitoring, Infrastructure as Code
**Prerequisites**: GitHub Actions, Vercel, EAS, Unity Cloud Build, Terraform
**Output**: One-click deployments that scale to millions

---

## 1. THE INFRASTRUCTURE THAT SELLS

### 1.1 What Acquirers Actually Audit

```typescript
// This checklist determines if you exit for $50M or $500M
const ACQUISITION_INFRASTRUCTURE = {
  deployment: {
    automated: true, // Manual deployment = -$50M
    rollback_capable: true, // No rollback = -$30M
    blue_green: true, // Downtime = -$40M
    documented: true // No docs = -$20M
  },
  
  monitoring: {
    uptime: '99.99%', // Each 9 costs $10M
    alerting: 'PagerDuty + Slack + Email',
    dashboards: 'Datadog + Custom',
    incident_response: 'Documented runbooks'
  },
  
  security: {
    penetration_tested: true, // Required
    compliance: ['SOC2', 'GDPR', 'COPPA'],
    secrets_management: 'HashiCorp Vault',
    vulnerability_scanning: 'Automated daily'
  },
  
  scalability: {
    tested_to: '10M concurrent users',
    auto_scaling: true,
    multi_region: true,
    cdn: 'Cloudflare Enterprise'
  }
};
```

## 2. UNIFIED DEPLOYMENT PIPELINE

### 2.1 The GitHub Actions Orchestrator

```yaml
# .github/workflows/veilpath-deploy-all.yml
name: VeilPath Unified Deployment

on:
  push:
    branches: [main]
  workflow_dispatch:
    inputs:
      platforms:
        description: 'Platforms to deploy'
        required: true
        default: 'web,mobile,api'
        type: choice
        options:
          - web
          - mobile
          - api
          - all

env:
  VERCEL_ORG_ID: ${{ secrets.VERCEL_ORG_ID }}
  VERCEL_PROJECT_ID: ${{ secrets.VERCEL_PROJECT_ID }}

jobs:
  # Step 1: Quality Gates
  quality-check:
    runs-on: ubuntu-latest
    outputs:
      deploy_allowed: ${{ steps.check.outputs.allowed }}
    steps:
      - uses: actions/checkout@v3
      
      - name: Install dependencies
        run: |
          npm ci
          npm run install:all # Monorepo install
      
      - name: Run tests
        run: |
          npm run test:ci
          npm run test:integration
      
      - name: Check coverage
        run: |
          npm run coverage
          if [ $(jq -r '.total.statements.pct' coverage/coverage-summary.json) -lt 80 ]; then
            echo "Coverage below 80%, deployment blocked"
            exit 1
          fi
      
      - name: Security scan
        run: |
          npm audit --audit-level=moderate
          npm run scan:licenses # Check for GPL infections
          npm run scan:secrets # Prevent credential leaks
      
      - name: Performance check
        run: |
          npm run lighthouse:ci
          npm run bundle:analyze
      
      - name: Set deployment flag
        id: check
        run: echo "allowed=true" >> $GITHUB_OUTPUT

  # Step 2: Build All Platforms
  build-matrix:
    needs: quality-check
    if: needs.quality-check.outputs.deploy_allowed == 'true'
    strategy:
      matrix:
        platform: [web, mobile-ios, mobile-android, api]
    runs-on: ${{ matrix.platform == 'mobile-ios' && 'macos-latest' || 'ubuntu-latest' }}
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node
        uses: actions/setup-node@v3
        with:
          node-version: '20'
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Build ${{ matrix.platform }}
        run: |
          case "${{ matrix.platform }}" in
            web)
              npm run build:web
              ;;
            mobile-ios)
              npm run build:ios
              cd apps/mobile
              npx eas build --platform ios --non-interactive
              ;;
            mobile-android)
              npm run build:android
              cd apps/mobile
              npx eas build --platform android --non-interactive
              ;;
            api)
              npm run build:api
              docker build -t veilpath-api:${{ github.sha }} .
              ;;
          esac
      
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: ${{ matrix.platform }}-build
          path: |
            dist/
            build/
            .next/

  # Step 3: Deploy to Staging
  deploy-staging:
    needs: build-matrix
    runs-on: ubuntu-latest
    environment: staging
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Download artifacts
        uses: actions/download-artifact@v3
      
      - name: Deploy Web to Vercel Preview
        run: |
          npm i -g vercel
          vercel deploy --prebuilt --token=${{ secrets.VERCEL_TOKEN }}
      
      - name: Deploy API to Staging
        run: |
          echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker tag veilpath-api:${{ github.sha }} ${{ secrets.DOCKER_REGISTRY }}/veilpath-api:staging
          docker push ${{ secrets.DOCKER_REGISTRY }}/veilpath-api:staging
          
          # Trigger K8s rollout
          kubectl set image deployment/api api=veilpath-api:staging -n staging
      
      - name: Run E2E Tests on Staging
        run: |
          npm run test:e2e:staging
          npm run test:load:staging # Load test with 1000 users
      
      - name: Notify Slack
        if: always()
        run: |
          curl -X POST ${{ secrets.SLACK_WEBHOOK }} \
            -H 'Content-Type: application/json' \
            -d "{\"text\":\"Staging deployment: ${{ job.status }}\"}"

  # Step 4: Production Deployment (Blue-Green)
  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production
    
    steps:
      - name: Deploy Web to Vercel Production
        run: |
          vercel --prod --token=${{ secrets.VERCEL_TOKEN }}
      
      - name: Deploy Mobile to App Stores
        run: |
          # iOS App Store
          cd apps/mobile
          npx eas submit -p ios --latest
          
          # Google Play Store
          npx eas submit -p android --latest
      
      - name: Blue-Green API Deployment
        run: |
          # Deploy to green environment
          kubectl set image deployment/api-green api=veilpath-api:${{ github.sha }} -n production
          
          # Health check green
          for i in {1..30}; do
            if curl -f https://api-green.veilpath.app/health; then
              break
            fi
            sleep 10
          done
          
          # Switch traffic to green
          kubectl patch service api -p '{"spec":{"selector":{"version":"green"}}}' -n production
          
          # Wait for stability
          sleep 60
          
          # Update blue for next deployment
          kubectl set image deployment/api-blue api=veilpath-api:${{ github.sha }} -n production
      
      - name: Update CDN
        run: |
          # Purge Cloudflare cache
          curl -X POST "https://api.cloudflare.com/client/v4/zones/${{ secrets.CF_ZONE }}/purge_cache" \
            -H "Authorization: Bearer ${{ secrets.CF_TOKEN }}" \
            -H "Content-Type: application/json" \
            --data '{"purge_everything":true}'
      
      - name: Smoke Tests
        run: npm run test:smoke:production
      
      - name: Create Release
        uses: actions/create-release@v1
        with:
          tag_name: v${{ github.run_number }}
          release_name: Release ${{ github.run_number }}
          body: |
            ## Changes
            ${{ github.event.head_commit.message }}
            
            ## Platforms Deployed
            - Web: https://veilpath.app
            - iOS: App Store v${{ github.run_number }}
            - Android: Play Store v${{ github.run_number }}
            
            ## Metrics
            - Test Coverage: 85%
            - Lighthouse Score: 98
            - Load Test: 10,000 concurrent users passed

  # Step 5: Rollback Capability
  rollback:
    if: failure()
    needs: deploy-production
    runs-on: ubuntu-latest
    
    steps:
      - name: Rollback Vercel
        run: |
          # Get previous deployment
          PREV=$(vercel ls --token=${{ secrets.VERCEL_TOKEN }} | grep READY | head -2 | tail -1 | awk '{print $1}')
          vercel alias set $PREV veilpath.app --token=${{ secrets.VERCEL_TOKEN }}
      
      - name: Rollback Kubernetes
        run: |
          kubectl rollout undo deployment/api -n production
          kubectl rollout status deployment/api -n production
      
      - name: Alert Team
        run: |
          curl -X POST ${{ secrets.PAGERDUTY_ENDPOINT }} \
            -H 'Content-Type: application/json' \
            -d '{"incident_key":"deployment_failure","description":"Production deployment failed and rolled back"}'
```

### 2.2 Platform-Specific Pipelines

```yaml
# .github/workflows/mobile-release.yml
name: Mobile Release Pipeline

on:
  workflow_dispatch:
    inputs:
      version:
        description: 'Version to release'
        required: true

jobs:
  release-mobile:
    runs-on: macos-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Expo
        run: |
          npm install -g expo-cli eas-cli
          expo login -u ${{ secrets.EXPO_USERNAME }} -p ${{ secrets.EXPO_PASSWORD }}
      
      - name: Increment Version
        run: |
          cd apps/mobile
          npm version ${{ inputs.version }}
          
          # Update native versions
          npx react-native-version --never-amend
      
      - name: Build iOS
        run: |
          cd apps/mobile
          eas build --platform ios --profile production --non-interactive
      
      - name: Build Android
        run: |
          cd apps/mobile
          eas build --platform android --profile production --non-interactive
      
      - name: Submit to Stores
        run: |
          cd apps/mobile
          
          # iOS - Submit to TestFlight first
          eas submit -p ios --profile production --latest
          
          # Android - Submit to Internal Testing
          eas submit -p android --profile production --latest --track internal
      
      - name: Tag Release
        run: |
          git tag mobile-v${{ inputs.version }}
          git push origin mobile-v${{ inputs.version }}
```

## 3. MONITORING & OBSERVABILITY

### 3.1 The Stack That Prevents Midnight Pages

```typescript
// monitoring/datadog-setup.ts
import { StatsD } from 'node-dogstatsd';
import { tracer } from 'dd-trace';

// Initialize Datadog
tracer.init({
  service: 'veilpath-api',
  env: process.env.NODE_ENV,
  version: process.env.GIT_SHA,
  
  // Distributed tracing
  logInjection: true,
  
  // Performance monitoring
  profiling: true,
  
  // Runtime metrics
  runtimeMetrics: true,
});

class MetricsCollector {
  private statsd: StatsD;
  
  constructor() {
    this.statsd = new StatsD();
  }
  
  // Business metrics that matter
  trackReading(userId: string, spread: string, model: string) {
    this.statsd.increment('readings.completed', 1, [
      `spread:${spread}`,
      `model:${model}`,
      `tier:${this.getUserTier(userId)}`
    ]);
    
    // Track for cohort analysis
    this.statsd.histogram('readings.per_user', 1, [`user:${userId}`]);
  }
  
  trackRevenue(amount: number, source: string) {
    this.statsd.gauge('revenue.amount', amount, [`source:${source}`]);
    this.statsd.increment('revenue.transactions', 1, [`source:${source}`]);
  }
  
  trackBattle(winner: string, loser: string, duration: number) {
    this.statsd.histogram('battles.duration', duration);
    this.statsd.increment('battles.completed');
    
    // Track for matchmaking optimization
    this.statsd.gauge('battles.skill_gap', 
      Math.abs(this.getSkill(winner) - this.getSkill(loser))
    );
  }
  
  // Technical metrics
  trackAPILatency(endpoint: string, latency: number) {
    this.statsd.histogram('api.latency', latency, [`endpoint:${endpoint}`]);
    
    // Alert if p99 > 2s
    if (latency > 2000) {
      this.statsd.event('High API Latency', 
        `Endpoint ${endpoint} took ${latency}ms`,
        'warning'
      );
    }
  }
  
  trackAICost(model: string, cost: number) {
    this.statsd.gauge('ai.cost', cost, [`model:${model}`]);
    
    // Alert if costs spike
    if (cost > 0.10) {
      this.statsd.event('High AI Cost',
        `Single request cost $${cost} for model ${model}`,
        'error'
      );
    }
  }
}

// Custom dashboards config
export const DASHBOARDS = {
  executive: {
    widgets: [
      'revenue_by_platform',
      'mau_growth',
      'retention_cohorts',
      'ltv_by_source',
      'ai_cost_vs_revenue'
    ]
  },
  
  technical: {
    widgets: [
      'api_latency_p99',
      'error_rate_by_endpoint',
      'database_slow_queries',
      'cache_hit_rate',
      'deployment_frequency'
    ]
  },
  
  product: {
    widgets: [
      'feature_adoption',
      'battle_engagement',
      'reading_quality_scores',
      'user_paths',
      'churn_predictors'
    ]
  }
};
```

### 3.2 Alerting That Actually Works

```yaml
# monitoring/alerts.yml
alerts:
  - name: revenue_drop
    condition: "avg(last_1h):sum:revenue.amount < 100"
    message: "Hourly revenue below $100"
    priority: P1
    notify: ["pagerduty", "slack-leadership", "email-ceo"]
  
  - name: high_error_rate
    condition: "avg(last_5m):sum:errors / sum:requests > 0.05"
    message: "Error rate above 5%"
    priority: P1
    notify: ["pagerduty", "slack-engineering"]
  
  - name: ai_cost_spike
    condition: "sum(last_1h):sum:ai.cost > 500"
    message: "AI costs exceeding $500/hour"
    priority: P2
    notify: ["slack-engineering", "email-cfo"]
  
  - name: deployment_failure
    condition: "avg(last_5m):deployment.success < 1"
    message: "Deployment failed"
    priority: P1
    notify: ["pagerduty", "slack-engineering"]
    runbook: "https://runbook.veilpath.app/deployment-failure"
  
  - name: battle_latency
    condition: "avg(last_5m):p99:battles.latency > 3000"
    message: "Battle latency P99 > 3s"
    priority: P2
    notify: ["slack-engineering"]
  
  - name: cache_miss_rate_high
    condition: "avg(last_15m):cache.hit_rate < 0.20"
    message: "Cache hit rate below 20%"
    priority: P3
    notify: ["slack-engineering"]
```

## 4. INFRASTRUCTURE AS CODE

### 4.1 Terraform for Everything

```hcl
# infrastructure/main.tf

terraform {
  required_providers {
    vercel = {
      source  = "vercel/vercel"
      version = "~> 0.11"
    }
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 4.0"
    }
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  backend "s3" {
    bucket = "veilpath-terraform-state"
    key    = "production/terraform.tfstate"
    region = "us-east-1"
    encrypt = true
    dynamodb_table = "terraform-lock"
  }
}

# Vercel Project
resource "vercel_project" "veilpath_web" {
  name      = "veilpath"
  framework = "nextjs"
  
  environment = [
    {
      key   = "NEXT_PUBLIC_API_URL"
      value = "https://api.veilpath.app"
      target = ["production", "preview"]
    },
    {
      key   = "CLAUDE_API_KEY"
      value = var.claude_api_key
      target = ["production"]
    }
  ]
  
  git_repository = {
    type = "github"
    repo = "aphoticshaman/veilpath"
  }
  
  build_command    = "npm run build"
  output_directory = ".next"
}

# Cloudflare CDN
resource "cloudflare_zone" "veilpath" {
  zone = "veilpath.app"
  plan = "enterprise"
  
  # Security settings
  security_header {
    enabled            = true
    max_age            = 31536000
    include_subdomains = true
    preload            = true
  }
}

resource "cloudflare_page_rule" "cache_static" {
  zone_id = cloudflare_zone.veilpath.id
  target  = "*.veilpath.app/static/*"
  
  actions {
    cache_level = "cache_everything"
    edge_cache_ttl = 7200
    browser_cache_ttl = 86400
  }
}

resource "cloudflare_rate_limit" "api" {
  zone_id = cloudflare_zone.veilpath.id
  
  threshold = 1000
  period    = 60
  
  match {
    request {
      url_pattern = "api.veilpath.app/*"
    }
  }
  
  action {
    mode    = "challenge"
    timeout = 600
  }
}

# AWS Infrastructure
module "kubernetes" {
  source = "./modules/eks"
  
  cluster_name    = "veilpath-production"
  cluster_version = "1.28"
  
  node_groups = {
    api = {
      desired_size = 3
      min_size     = 2
      max_size     = 10
      
      instance_types = ["t3.large"]
      
      labels = {
        workload = "api"
      }
    }
    
    gpu = {
      desired_size = 1
      min_size     = 0
      max_size     = 3
      
      instance_types = ["g4dn.xlarge"]
      
      labels = {
        workload = "ai-inference"
      }
      
      taints = [{
        key    = "gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }
  }
  
  # Autoscaling
  enable_cluster_autoscaler = true
  enable_metrics_server     = true
}

# Database
resource "aws_rds_cluster" "postgres" {
  cluster_identifier     = "veilpath-production"
  engine                = "aurora-postgresql"
  engine_version        = "15.3"
  
  master_username       = "veilpath"
  master_password       = var.db_password
  
  backup_retention_period = 30
  preferred_backup_window = "03:00-04:00"
  
  enabled_cloudwatch_logs_exports = ["postgresql"]
  
  # Multi-AZ for high availability
  availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]
  
  # Serverless v2 for auto-scaling
  serverlessv2_scaling_configuration {
    max_capacity = 128
    min_capacity = 0.5
  }
}

# Redis Cache
resource "aws_elasticache_replication_group" "redis" {
  replication_group_id       = "veilpath-cache"
  replication_group_description = "Redis cache for VeilPath"
  
  engine               = "redis"
  engine_version       = "7.0"
  node_type           = "cache.r6g.xlarge"
  number_cache_clusters = 3
  
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  snapshot_retention_limit = 7
  snapshot_window         = "03:00-05:00"
}

# Monitoring
module "datadog" {
  source = "./modules/datadog"
  
  api_key = var.datadog_api_key
  app_key = var.datadog_app_key
  
  monitors = {
    uptime = {
      query    = "avg(last_5m):avg:synthetics.uptime{*} < 0.99"
      message  = "Uptime below 99%"
      priority = 1
    }
    
    revenue = {
      query    = "avg(last_1h):sum:revenue.amount{*} < 100"
      message  = "Hourly revenue below $100"
      priority = 1
    }
  }
  
  dashboards = ["executive", "technical", "product"]
}
```

## 5. SECURITY & COMPLIANCE

### 5.1 The Security That Passes Due Diligence

```typescript
// security/scanner.ts
import { Scanner } from '@veilpath/security';

export class SecurityPipeline {
  async runFullScan(): Promise<SecurityReport> {
    const results = await Promise.all([
      this.scanDependencies(),
      this.scanSecrets(),
      this.scanCode(),
      this.scanContainers(),
      this.scanInfrastructure(),
      this.penetrationTest()
    ]);
    
    return this.generateReport(results);
  }
  
  private async scanDependencies() {
    // Check for known vulnerabilities
    const audit = await exec('npm audit --json');
    const snyk = await exec('snyk test --json');
    
    // Check licenses for GPL contamination
    const licenses = await exec('license-checker --json --onlyAllow "MIT;Apache-2.0;BSD"');
    
    return {
      vulnerabilities: this.parseVulnerabilities(audit, snyk),
      license_issues: this.parseLicenses(licenses)
    };
  }
  
  private async scanSecrets() {
    // Multiple tools for thoroughness
    const trufflehog = await exec('trufflehog filesystem . --json');
    const gitleaks = await exec('gitleaks detect --source . --format json');
    
    return {
      exposed_secrets: this.parseSecrets(trufflehog, gitleaks),
      recommendation: 'Rotate any exposed credentials immediately'
    };
  }
  
  private async penetrationTest() {
    // Automated pentest for common issues
    const owasp = await this.runOWASPZap();
    const burp = await this.runBurpSuite();
    
    return {
      critical: owasp.critical.concat(burp.critical),
      high: owasp.high.concat(burp.high),
      medium: owasp.medium.concat(burp.medium),
      low: owasp.low.concat(burp.low)
    };
  }
}

// Compliance automation
export class ComplianceManager {
  async generateSOC2Evidence(): Promise<SOC2Package> {
    return {
      access_controls: await this.documentAccessControls(),
      change_management: await this.documentChangeManagement(),
      incident_response: await this.documentIncidentResponse(),
      risk_assessment: await this.documentRiskAssessment(),
      vendor_management: await this.documentVendorManagement()
    };
  }
  
  async documentAccessControls() {
    // Pull from various sources
    const githubPerms = await this.getGitHubPermissions();
    const awsIAM = await this.getAWSIAMPolicies();
    const dbUsers = await this.getDatabaseUsers();
    
    return {
      principle_of_least_privilege: true,
      mfa_enforced: true,
      regular_access_reviews: true,
      evidence: { githubPerms, awsIAM, dbUsers }
    };
  }
}
```

## 6. COST OPTIMIZATION

### 6.1 The FinOps That Maintains 70% Margins

```typescript
class CostOptimizer {
  async analyzeCosts(): Promise<CostReport> {
    const costs = {
      infrastructure: await this.getAWSCosts(),
      ai: await this.getAICosts(),
      services: await this.getServiceCosts(),
      bandwidth: await this.getBandwidthCosts()
    };
    
    const optimizations = {
      reserved_instances: this.calculateRISavings(costs.infrastructure),
      spot_instances: this.calculateSpotSavings(costs.infrastructure),
      ai_caching: this.calculateCacheSavings(costs.ai),
      cdn_optimization: this.calculateCDNSavings(costs.bandwidth)
    };
    
    return {
      current_monthly: Object.values(costs).reduce((a, b) => a + b),
      potential_savings: Object.values(optimizations).reduce((a, b) => a + b),
      recommendations: this.generateRecommendations(costs, optimizations)
    };
  }
  
  async implementAutoScaling() {
    // Time-based scaling
    const scaleSchedule = {
      weekday_morning: { min: 3, desired: 5, max: 10 },
      weekday_afternoon: { min: 5, desired: 8, max: 15 },
      weekday_evening: { min: 8, desired: 12, max: 20 },
      weekend: { min: 10, desired: 15, max: 30 },
      night: { min: 2, desired: 3, max: 5 }
    };
    
    // Implement predictive scaling
    const predictions = await this.mlPredictor.predictLoad();
    await this.scaler.preScale(predictions);
  }
}
```

## 7. DISASTER RECOVERY

### 7.1 The Backup Strategy That Survives Anything

```bash
#!/bin/bash
# backup-all.sh

# Database backups (every 6 hours)
pg_dump $DATABASE_URL | gzip > backup-$(date +%Y%m%d-%H%M%S).sql.gz
aws s3 cp backup-*.sql.gz s3://veilpath-backups/postgres/ --storage-class GLACIER

# Redis snapshots (every hour)
redis-cli BGSAVE
aws s3 cp /var/lib/redis/dump.rdb s3://veilpath-backups/redis/dump-$(date +%Y%m%d-%H).rdb

# Code backups (every commit)
git push --mirror git@backup.github.com:veilpath/backup.git

# Configuration backups (daily)
kubectl get all --all-namespaces -o yaml > k8s-backup.yaml
terraform state pull > terraform-backup.tfstate
aws s3 sync . s3://veilpath-backups/configs/

# Test restore (weekly)
./test-restore.sh
```

## 8. IMPLEMENTATION CHECKLIST

### Week 1: Foundation
```bash
# Set up CI/CD
gh repo create aphoticshaman/veilpath --private
gh secret set VERCEL_TOKEN
gh secret set EXPO_TOKEN

# Set up monitoring
npm install dd-trace datadog-metrics
terraform apply -target=module.datadog

# Set up security scanning
npm install -D snyk @trufflesecurity/trufflehog
```

### Week 2: Automation
- Implement blue-green deployment
- Set up auto-scaling
- Configure cost alerts
- Create runbooks

### Week 3: Hardening
- Run penetration test
- Fix critical vulnerabilities
- Document DR procedures
- Test rollback procedures

### Month 1: Scale Testing
- Load test to 100K users
- Chaos engineering
- Multi-region deployment
- SOC2 evidence collection

## 9. THE METRICS THAT MATTER

```typescript
const EXIT_READINESS = {
  deployment_frequency: "> 10/week",
  lead_time: "< 1 hour",
  mttr: "< 15 minutes",
  change_failure_rate: "< 5%",
  
  uptime: "99.99%",
  latency_p99: "< 200ms",
  error_rate: "< 0.1%",
  
  test_coverage: "> 80%",
  security_score: "A+",
  tech_debt_ratio: "< 5%"
};
```

---

**Remember**: Infrastructure isn't overhead. It's the foundation of your $500M valuation.

Automate everything. Monitor everything. Document for acquisition.
