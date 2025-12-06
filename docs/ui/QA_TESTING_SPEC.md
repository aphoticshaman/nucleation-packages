# LatticeForge QA Engineer Testing Specification

## Document Purpose

This specification provides comprehensive quality assurance guidance for LatticeForge. It covers testing strategy, test automation, test case management, quality metrics, and release processes. QA engineers should use this as the authoritative reference for ensuring product quality across all components.

---

## 1. Quality Philosophy

### 1.1 Quality Principles

**Shift Left**
Find defects as early as possible in the development lifecycle. Prevention is cheaper than detection; detection is cheaper than correction in production.

**Test Pyramid**
Maintain a healthy ratio of unit tests (many), integration tests (some), and E2E tests (few). Unit tests run fast and catch most issues; E2E tests validate critical user journeys.

**Continuous Quality**
Quality is built in, not tested in. Every PR must pass automated checks. Quality gates prevent regressions from reaching production.

**Risk-Based Testing**
Prioritize testing based on business impact and technical risk. Not all features need the same level of coverage.

**User-Centric Quality**
The ultimate measure of quality is user satisfaction. Test what users care about, not just what code does.

### 1.2 Quality Responsibilities

| Role | Responsibility |
|------|---------------|
| Developer | Unit tests, code review, fix own defects |
| QA Engineer | Test strategy, automation, exploratory testing |
| Product Manager | Acceptance criteria, priority guidance |
| SRE | Performance testing, production monitoring |
| Security | Security testing, vulnerability assessment |

### 1.3 Definition of Done

A feature is "Done" when:
- [ ] Code is complete and reviewed
- [ ] Unit tests written and passing (>80% coverage)
- [ ] Integration tests written for API changes
- [ ] E2E tests updated for UI changes
- [ ] Manual exploratory testing completed
- [ ] Performance impact assessed
- [ ] Security implications reviewed
- [ ] Documentation updated
- [ ] Accessibility verified
- [ ] Product acceptance received

---

## 2. Test Strategy

### 2.1 Test Pyramid

```
                    /\
                   /  \
                  / E2E\        5-10% of tests
                 / Tests \      ~50 critical paths
                /----------\
               /            \
              / Integration  \  20-30% of tests
             /    Tests       \ ~500 API/service tests
            /------------------\
           /                    \
          /     Unit Tests       \ 60-70% of tests
         /                        \ ~5000+ unit tests
        /---------------------------\
```

### 2.2 Test Types by Component

**Frontend (Leptos/WASM):**
| Type | Scope | Tools | Coverage Target |
|------|-------|-------|-----------------|
| Unit | Component logic, utils | wasm-bindgen-test | 80% |
| Component | Isolated components | Leptos test utilities | Key components |
| Integration | Component interactions | Browser automation | Critical flows |
| E2E | Full user journeys | Playwright | Top 20 journeys |
| Visual | UI regression | Percy/Chromatic | All pages |
| A11y | Accessibility | axe-core, manual | All pages |

**Backend (Rust/Axum):**
| Type | Scope | Tools | Coverage Target |
|------|-------|-------|-----------------|
| Unit | Functions, modules | cargo test | 85% |
| Integration | API endpoints | cargo test + testcontainers | All endpoints |
| Contract | API contracts | Pact or similar | All public APIs |
| Load | Performance | k6, Locust | Key endpoints |
| Security | Vulnerabilities | OWASP ZAP, sqlmap | All endpoints |

**AI/ML Pipeline:**
| Type | Scope | Tools | Coverage Target |
|------|-------|-------|-----------------|
| Unit | Processing functions | pytest | 80% |
| Integration | Pipeline stages | pytest + fixtures | All pipelines |
| Quality | Output quality | Custom metrics | All model outputs |
| Regression | Model behavior | Golden datasets | On model changes |

### 2.3 Testing Environments

| Environment | Purpose | Data | Access |
|-------------|---------|------|--------|
| Local | Developer testing | Fixtures | Developers |
| CI | Automated tests | Generated | CI system |
| Staging | Pre-production | Sanitized prod | QA, Dev |
| Production | Live monitoring | Real | Limited |

**Environment Parity:**
- Staging mirrors production architecture
- Same container images, different configs
- Sanitized production data refreshed weekly
- Feature flags control environment differences

---

## 3. Test Automation

### 3.1 Unit Testing

**Rust Backend Example:**
```rust
// tests/unit/services/source_service_test.rs

use latticeforge::services::source_service::SourceService;
use latticeforge::models::Source;
use mockall::predicate::*;

#[tokio::test]
async fn test_source_creation_validates_url() {
    // Arrange
    let mut mock_repo = MockSourceRepository::new();
    mock_repo.expect_create().never();

    let service = SourceService::new(mock_repo);

    // Act
    let result = service.create_source(CreateSourceRequest {
        url: "not-a-valid-url".into(),
        stream_id: "stream_123".into(),
    }).await;

    // Assert
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err().code,
        "validation_error"
    );
}

#[tokio::test]
async fn test_source_processing_extracts_entities() {
    // Arrange
    let source = Source {
        id: "source_123".into(),
        content: "CRISPR-Cas9 is a gene editing technology...".into(),
        ..Default::default()
    };

    let processor = SourceProcessor::new(MockEntityExtractor::with_response(vec![
        Entity::new("CRISPR-Cas9", EntityType::Technology),
    ]));

    // Act
    let result = processor.process(&source).await.unwrap();

    // Assert
    assert_eq!(result.entities.len(), 1);
    assert_eq!(result.entities[0].name, "CRISPR-Cas9");
}
```

**Leptos Frontend Example:**
```rust
// tests/unit/components/source_card_test.rs

use leptos::*;
use wasm_bindgen_test::*;
use crate::components::SourceCard;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn test_source_card_displays_title() {
    create_scope(create_runtime(), |cx| {
        let source = Source {
            id: "source_123".into(),
            title: "Test Paper Title".into(),
            authors: vec!["Author A".into()],
            ..Default::default()
        };

        let view = view! { cx,
            <SourceCard source=source />
        };

        // Mount and query
        let container = mount_to_document(view);
        let title = container.query_selector(".source-title").unwrap().unwrap();

        assert_eq!(title.text_content().unwrap(), "Test Paper Title");
    });
}

#[wasm_bindgen_test]
fn test_source_card_click_selects() {
    create_scope(create_runtime(), |cx| {
        let (selected, set_selected) = create_signal(cx, None::<String>);

        let source = Source {
            id: "source_123".into(),
            ..Default::default()
        };

        let view = view! { cx,
            <SourceCard
                source=source
                on_select=move |id| set_selected(Some(id))
            />
        };

        let container = mount_to_document(view);
        let card = container.query_selector(".source-card").unwrap().unwrap();

        // Simulate click
        card.dispatch_event(&Event::new("click").unwrap()).unwrap();

        assert_eq!(selected.get(), Some("source_123".into()));
    });
}
```

### 3.2 Integration Testing

**API Integration Tests:**
```rust
// tests/integration/api/streams_test.rs

use axum_test::TestServer;
use sqlx::PgPool;

#[sqlx::test]
async fn test_create_stream_returns_201(pool: PgPool) {
    // Arrange
    let app = create_test_app(pool).await;
    let server = TestServer::new(app).unwrap();

    let token = create_test_user_token(&server).await;

    // Act
    let response = server
        .post("/v1/streams")
        .authorization_bearer(&token)
        .json(&json!({
            "name": "Test Stream",
            "visibility": "private"
        }))
        .await;

    // Assert
    response.assert_status_created();

    let body: Value = response.json();
    assert!(body["data"]["id"].as_str().unwrap().starts_with("stream_"));
    assert_eq!(body["data"]["name"], "Test Stream");
}

#[sqlx::test]
async fn test_get_stream_requires_authorization(pool: PgPool) {
    // Arrange
    let app = create_test_app(pool).await;
    let server = TestServer::new(app).unwrap();

    let (_, stream) = create_test_user_with_stream(&server).await;
    let other_token = create_test_user_token(&server).await;

    // Act - try to access stream with different user
    let response = server
        .get(&format!("/v1/streams/{}", stream.id))
        .authorization_bearer(&other_token)
        .await;

    // Assert
    response.assert_status_not_found(); // Don't reveal existence
}

#[sqlx::test]
async fn test_source_processing_flow(pool: PgPool) {
    // Arrange
    let app = create_test_app(pool).await;
    let server = TestServer::new(app).unwrap();

    let (token, stream) = create_test_user_with_stream(&server).await;

    // Act - Add source
    let response = server
        .post(&format!("/v1/streams/{}/sources", stream.id))
        .authorization_bearer(&token)
        .json(&json!({
            "url": "https://arxiv.org/abs/2301.00000"
        }))
        .await;

    response.assert_status_accepted();
    let source_id = response.json::<Value>()["data"]["id"].as_str().unwrap();

    // Wait for processing
    tokio::time::sleep(Duration::from_secs(5)).await;

    // Assert - Source is processed
    let response = server
        .get(&format!("/v1/sources/{}", source_id))
        .authorization_bearer(&token)
        .await;

    response.assert_status_ok();
    assert_eq!(response.json::<Value>()["data"]["status"], "processed");
}
```

### 3.3 End-to-End Testing

**Playwright E2E Tests:**
```typescript
// tests/e2e/flows/onboarding.spec.ts

import { test, expect } from '@playwright/test';

test.describe('User Onboarding', () => {
  test('new user can sign up and add first source', async ({ page }) => {
    // Navigate to sign up
    await page.goto('/signup');

    // Fill sign up form
    await page.fill('[data-testid="email-input"]', 'test@example.com');
    await page.fill('[data-testid="password-input"]', 'SecurePassword123!');
    await page.click('[data-testid="signup-button"]');

    // Should redirect to onboarding
    await expect(page).toHaveURL('/onboarding');

    // Complete research focus step
    await page.fill('[data-testid="research-focus"]', 'Machine Learning');
    await page.click('[data-testid="continue-button"]');

    // Add first source
    await page.fill('[data-testid="source-url"]', 'https://arxiv.org/abs/2301.00000');
    await page.click('[data-testid="add-source-button"]');

    // Wait for processing
    await expect(page.locator('[data-testid="processing-indicator"]')).toBeVisible();
    await expect(page.locator('[data-testid="source-ready"]')).toBeVisible({ timeout: 30000 });

    // Should see first insight
    await expect(page.locator('[data-testid="insight-card"]')).toBeVisible();

    // Should be on dashboard
    await expect(page).toHaveURL('/dashboard');
  });

  test('existing user can log in', async ({ page }) => {
    await page.goto('/login');

    await page.fill('[data-testid="email-input"]', 'existing@example.com');
    await page.fill('[data-testid="password-input"]', 'ExistingPassword123!');
    await page.click('[data-testid="login-button"]');

    await expect(page).toHaveURL('/dashboard');
    await expect(page.locator('[data-testid="user-avatar"]')).toBeVisible();
  });
});

test.describe('Research Stream Management', () => {
  test.beforeEach(async ({ page }) => {
    await loginAsTestUser(page);
  });

  test('user can create new stream', async ({ page }) => {
    await page.click('[data-testid="new-stream-button"]');

    await page.fill('[data-testid="stream-name"]', 'My Research Project');
    await page.fill('[data-testid="stream-description"]', 'Testing stream creation');
    await page.click('[data-testid="create-stream-button"]');

    await expect(page).toHaveURL(/\/streams\/stream_/);
    await expect(page.locator('h1')).toContainText('My Research Project');
  });

  test('user can add source to stream', async ({ page }) => {
    await createTestStream(page);

    // Drag and drop PDF
    const fileInput = page.locator('[data-testid="file-input"]');
    await fileInput.setInputFiles('tests/fixtures/sample-paper.pdf');

    // Wait for processing
    await expect(page.locator('[data-testid="source-card"]')).toBeVisible({ timeout: 30000 });
    await expect(page.locator('[data-testid="entity-count"]')).toContainText(/\d+ entities/);
  });

  test('user can generate synthesis', async ({ page }) => {
    await createStreamWithSources(page, 3);

    await page.click('[data-testid="generate-synthesis-button"]');

    // Should see streaming generation
    await expect(page.locator('[data-testid="synthesis-content"]')).toBeVisible();
    await expect(page.locator('[data-testid="generating-indicator"]')).toBeVisible();

    // Wait for completion
    await expect(page.locator('[data-testid="generating-indicator"]')).not.toBeVisible({ timeout: 60000 });

    // Should have citations
    await expect(page.locator('[data-testid="citation-link"]').first()).toBeVisible();
  });
});

test.describe('Accessibility', () => {
  test('dashboard is accessible', async ({ page }) => {
    await loginAsTestUser(page);
    await page.goto('/dashboard');

    const accessibilityScanResults = await new AxeBuilder({ page }).analyze();
    expect(accessibilityScanResults.violations).toEqual([]);
  });

  test('keyboard navigation works', async ({ page }) => {
    await loginAsTestUser(page);
    await page.goto('/dashboard');

    // Tab through main navigation
    await page.keyboard.press('Tab');
    await expect(page.locator('[data-testid="nav-streams"]')).toBeFocused();

    await page.keyboard.press('Tab');
    await expect(page.locator('[data-testid="nav-library"]')).toBeFocused();

    // Enter activates focused element
    await page.keyboard.press('Enter');
    await expect(page).toHaveURL('/library');
  });
});
```

### 3.4 Visual Regression Testing

**Percy Integration:**
```typescript
// tests/visual/pages.spec.ts

import { test } from '@playwright/test';
import percySnapshot from '@percy/playwright';

test.describe('Visual Regression', () => {
  test('dashboard matches snapshot', async ({ page }) => {
    await loginAsTestUser(page);
    await page.goto('/dashboard');

    // Wait for content to load
    await page.waitForSelector('[data-testid="stream-card"]');

    await percySnapshot(page, 'Dashboard');
  });

  test('stream detail matches snapshot', async ({ page }) => {
    await loginAsTestUser(page);
    await page.goto('/streams/stream_test123');

    await page.waitForSelector('[data-testid="source-list"]');

    await percySnapshot(page, 'Stream Detail');
  });

  test('dark mode matches snapshot', async ({ page }) => {
    await loginAsTestUser(page);
    await page.goto('/dashboard');

    // Toggle dark mode
    await page.click('[data-testid="theme-toggle"]');
    await page.waitForTimeout(500); // Wait for transition

    await percySnapshot(page, 'Dashboard - Dark Mode');
  });

  test('responsive snapshots', async ({ page }) => {
    await loginAsTestUser(page);
    await page.goto('/dashboard');

    // Desktop
    await page.setViewportSize({ width: 1440, height: 900 });
    await percySnapshot(page, 'Dashboard - Desktop');

    // Tablet
    await page.setViewportSize({ width: 768, height: 1024 });
    await percySnapshot(page, 'Dashboard - Tablet');

    // Mobile
    await page.setViewportSize({ width: 375, height: 667 });
    await percySnapshot(page, 'Dashboard - Mobile');
  });
});
```

### 3.5 Performance Testing

**k6 Load Test:**
```javascript
// tests/load/api-load.js

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

const errorRate = new Rate('errors');
const latency = new Trend('latency');

export const options = {
  stages: [
    { duration: '2m', target: 50 },   // Ramp up
    { duration: '5m', target: 50 },   // Stay at 50 users
    { duration: '2m', target: 100 },  // Spike to 100
    { duration: '5m', target: 100 },  // Stay at 100
    { duration: '2m', target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<500', 'p(99)<1000'],
    errors: ['rate<0.01'],
  },
};

const BASE_URL = __ENV.BASE_URL || 'https://api.staging.latticeforge.io';

export function setup() {
  // Create test user and get token
  const loginRes = http.post(`${BASE_URL}/v1/auth/login`, JSON.stringify({
    email: 'loadtest@example.com',
    password: 'loadtest123',
  }), { headers: { 'Content-Type': 'application/json' } });

  return {
    token: loginRes.json('access_token'),
  };
}

export default function(data) {
  const headers = {
    'Authorization': `Bearer ${data.token}`,
    'Content-Type': 'application/json',
  };

  // List streams
  const streamsRes = http.get(`${BASE_URL}/v1/streams`, { headers });
  check(streamsRes, { 'list streams status 200': (r) => r.status === 200 });
  latency.add(streamsRes.timings.duration);
  errorRate.add(streamsRes.status !== 200);

  sleep(1);

  // Get specific stream
  const streamId = streamsRes.json('data.0.id');
  if (streamId) {
    const streamRes = http.get(`${BASE_URL}/v1/streams/${streamId}`, { headers });
    check(streamRes, { 'get stream status 200': (r) => r.status === 200 });
    latency.add(streamRes.timings.duration);
  }

  sleep(1);

  // Generate synthesis (expensive operation)
  if (Math.random() < 0.1) { // Only 10% of virtual users
    const synthRes = http.post(`${BASE_URL}/v1/streams/${streamId}/syntheses`, JSON.stringify({
      source_ids: ['source_1', 'source_2'],
      type: 'overview',
    }), { headers });
    check(synthRes, { 'synthesis accepted': (r) => r.status === 202 });
  }

  sleep(2);
}

export function handleSummary(data) {
  return {
    'results/load-test-summary.json': JSON.stringify(data),
  };
}
```

---

## 4. Test Case Management

### 4.1 Test Case Structure

```yaml
# test-cases/TC-001-user-authentication.yaml
id: TC-001
title: User Authentication - Email/Password Login
priority: P0
type: functional
component: authentication
automated: true
automation_id: tests/e2e/auth.spec.ts::test_email_login

preconditions:
  - User account exists with email "test@example.com"
  - User is not logged in

steps:
  - step: Navigate to login page
    expected: Login form is displayed
  - step: Enter valid email "test@example.com"
    expected: Email field accepts input
  - step: Enter valid password
    expected: Password field shows masked characters
  - step: Click Login button
    expected: |
      - Loading state shown
      - Redirected to dashboard
      - User avatar visible in header

postconditions:
  - User session is active
  - Refresh token is stored

tags:
  - auth
  - critical-path
  - regression

history:
  - date: 2024-03-01
    author: QA Team
    change: Created
  - date: 2024-03-15
    author: QA Team
    change: Added automation reference
```

### 4.2 Test Suite Organization

```
test-suites/
├── smoke/                    # Quick sanity checks (~5 min)
│   ├── auth-smoke.yaml
│   ├── streams-smoke.yaml
│   └── sources-smoke.yaml
│
├── regression/               # Full regression (~2 hours)
│   ├── authentication/
│   ├── streams/
│   ├── sources/
│   ├── synthesis/
│   ├── insights/
│   └── export/
│
├── integration/              # Cross-feature tests
│   ├── user-journey-onboarding.yaml
│   ├── user-journey-synthesis.yaml
│   └── user-journey-collaboration.yaml
│
├── performance/              # Performance test scenarios
│   ├── load-baseline.yaml
│   ├── stress-test.yaml
│   └── soak-test.yaml
│
└── exploratory/              # Exploratory testing charters
    ├── new-feature-exploration.md
    └── edge-case-hunting.md
```

### 4.3 Test Coverage Matrix

| Feature | Unit | Integration | E2E | Manual | Coverage |
|---------|------|-------------|-----|--------|----------|
| Authentication | 85% | 100% | 80% | 100% | Complete |
| Stream CRUD | 90% | 100% | 70% | 100% | Complete |
| Source Import | 80% | 90% | 60% | 100% | Complete |
| Processing | 85% | 80% | 40% | 80% | Adequate |
| Synthesis | 75% | 70% | 50% | 100% | Adequate |
| Insights | 70% | 60% | 40% | 80% | Needs work |
| Graph | 60% | 50% | 30% | 70% | Needs work |
| Export | 80% | 90% | 60% | 100% | Complete |
| Collaboration | 50% | 40% | 20% | 60% | Insufficient |

### 4.4 Exploratory Testing Charters

```markdown
# Charter: Synthesis Generation Edge Cases

## Mission
Explore synthesis generation with unusual inputs to find edge cases and failure modes.

## Areas to Explore
- Very long sources (>100 pages)
- Non-English sources
- Sources with heavy formatting (tables, equations)
- Mixed source types (papers + articles + code)
- Sources with conflicting information
- Single-source synthesis
- Empty or minimal content sources

## Time Box
2 hours

## Notes Template
| Time | Area | Observation | Severity | Bug # |
|------|------|-------------|----------|-------|
| 10:00 | Long sources | Synthesis truncated at 5k tokens | Medium | BUG-123 |

## Debrief Questions
1. What surprised you?
2. What areas need more testing?
3. What risks did you discover?
```

---

## 5. Bug Management

### 5.1 Bug Severity Classification

| Severity | Definition | Response Time | Examples |
|----------|------------|---------------|----------|
| S1 - Critical | System unusable, data loss | 4 hours | Auth broken, data corruption |
| S2 - High | Major feature broken | 24 hours | Synthesis fails, can't add sources |
| S3 - Medium | Feature degraded | 1 week | Slow performance, UI glitch |
| S4 - Low | Minor issue | Next sprint | Typo, cosmetic issue |

### 5.2 Bug Report Template

```markdown
## Bug Report: [Title]

**ID:** BUG-XXX
**Severity:** S2 - High
**Priority:** P1
**Component:** Synthesis
**Environment:** Production
**Reporter:** [Name]
**Date:** 2024-03-15

### Description
[Clear description of the bug]

### Steps to Reproduce
1. Login as any user
2. Navigate to stream with 5+ sources
3. Click "Generate Synthesis"
4. Select "Comparison" type
5. Click "Generate"

### Expected Result
Synthesis should generate comparing all selected sources.

### Actual Result
Error message "Failed to generate synthesis" after 30 seconds.

### Environment Details
- Browser: Chrome 122.0.6261.69
- OS: macOS 14.3
- Account: test@example.com
- Stream ID: stream_abc123

### Screenshots/Logs
[Attach screenshots, console logs, network requests]

### Additional Context
- Works for "Overview" type
- Fails consistently with >3 sources
- Error in console: `TimeoutError: Request exceeded 30000ms`
```

### 5.3 Bug Triage Process

**Daily Bug Triage:**
1. Review new bugs (15 min)
2. Assign severity and priority
3. Route to appropriate team
4. Update stakeholders on critical bugs

**Triage Checklist:**
- [ ] Is severity correctly assigned?
- [ ] Can it be reproduced?
- [ ] Is there a workaround?
- [ ] What's the user impact?
- [ ] Which component is affected?
- [ ] Who should fix it?

---

## 6. Quality Metrics

### 6.1 Key Quality Indicators

**Defect Metrics:**
| Metric | Target | Current | Trend |
|--------|--------|---------|-------|
| Escaped defects/release | <3 | 2 | Stable |
| S1/S2 bugs in prod | 0 | 0 | Stable |
| Mean time to detect (MTTD) | <24h | 18h | Improving |
| Mean time to resolve (MTTR) | <48h | 36h | Improving |

**Test Metrics:**
| Metric | Target | Current | Trend |
|--------|--------|---------|-------|
| Unit test coverage | >80% | 78% | Improving |
| Integration test coverage | >70% | 65% | Stable |
| E2E test coverage (critical paths) | 100% | 90% | Improving |
| Test pass rate (CI) | >98% | 97.5% | Stable |
| Flaky test rate | <2% | 3% | Needs work |

**Process Metrics:**
| Metric | Target | Current | Trend |
|--------|--------|---------|-------|
| Test cycle time | <4h | 3.5h | Stable |
| Automation ratio | >80% | 75% | Improving |
| Regression test time | <2h | 1.5h | Stable |

### 6.2 Quality Dashboard

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LatticeForge Quality Dashboard                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Build Status: ● PASSING                 Last Updated: 2024-03-15 14:30    │
│                                                                             │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐               │
│  │ Unit Tests      │ │ Integration     │ │ E2E Tests       │               │
│  │ 4,892 / 4,950   │ │ 487 / 502       │ │ 48 / 52         │               │
│  │ 98.8% ✓         │ │ 97.0% ✓         │ │ 92.3% ⚠         │               │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘               │
│                                                                             │
│  Open Bugs by Severity:                                                     │
│  S1: 0   S2: 2   S3: 8   S4: 15                                            │
│                                                                             │
│  Test Coverage Trend:                                                       │
│  [========================================] 78% (+2% this week)            │
│                                                                             │
│  Recent Failures:                                                           │
│  • e2e/synthesis.spec.ts - Timeout on large source (INVESTIGATING)         │
│  • integration/auth.rs - Token refresh race (FIXED, awaiting deploy)       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Quality Gates

**PR Quality Gate:**
- [ ] All unit tests pass
- [ ] No decrease in coverage
- [ ] No new lint warnings
- [ ] No high/critical vulnerabilities
- [ ] Code review approved

**Merge Quality Gate:**
- [ ] All integration tests pass
- [ ] E2E smoke tests pass
- [ ] Performance benchmarks within 10% of baseline

**Release Quality Gate:**
- [ ] Full regression suite passes
- [ ] No S1/S2 bugs open
- [ ] Load test passes
- [ ] Security scan clean
- [ ] Staging sign-off from QA

---

## 7. Release Testing

### 7.1 Release Types

| Type | Testing Scope | Duration | Approval |
|------|---------------|----------|----------|
| Hotfix | Targeted fix + smoke | 2 hours | QA Lead |
| Patch | Bug fixes + regression | 4 hours | QA Lead |
| Minor | New features + full regression | 1 day | QA Lead + PM |
| Major | Full testing + beta | 1 week | QA Lead + PM + Eng Lead |

### 7.2 Release Checklist

**Pre-Release:**
```markdown
## Release v1.2.0 Checklist

### Testing
- [ ] Full regression suite executed
- [ ] All P0/P1 tests passing
- [ ] Performance benchmarks acceptable
- [ ] Security scan completed
- [ ] Accessibility audit completed
- [ ] Cross-browser testing done
- [ ] Mobile testing done

### Documentation
- [ ] Release notes drafted
- [ ] API changes documented
- [ ] Known issues documented
- [ ] Runbook updated

### Approval
- [ ] QA sign-off: ________________
- [ ] PM sign-off: ________________
- [ ] Eng Lead sign-off: ________________

### Deployment
- [ ] Staging deployment verified
- [ ] Monitoring alerts configured
- [ ] Rollback plan documented
- [ ] On-call notified
```

### 7.3 Rollback Criteria

Initiate rollback if:
- Error rate increases >5% from baseline
- P99 latency increases >50%
- Any S1 bug discovered
- >10 user-reported issues in first hour
- Critical business metric drops >20%

---

## 8. Specialized Testing

### 8.1 Security Testing

**OWASP Top 10 Checklist:**
| Vulnerability | Test Approach | Status |
|--------------|---------------|--------|
| Injection | SQLMap, manual testing | Automated |
| Broken Auth | Session testing, brute force | Automated |
| Sensitive Data | SSL scan, data exposure | Automated |
| XXE | XML parsing tests | Automated |
| Broken Access | IDOR testing, privilege escalation | Manual |
| Misconfig | Config scan, header analysis | Automated |
| XSS | ZAP, manual payloads | Automated |
| Insecure Deserialization | Payload testing | Manual |
| Known Vulnerabilities | Dependency scan | Automated |
| Logging | Log injection, log review | Manual |

### 8.2 Accessibility Testing

**WCAG 2.1 AA Checklist:**
| Criterion | Test Method | Tool |
|-----------|-------------|------|
| Color contrast | Automated + manual | axe-core |
| Keyboard navigation | Manual | - |
| Screen reader | Manual | NVDA, VoiceOver |
| Focus indicators | Automated + manual | axe-core |
| Alt text | Automated | axe-core |
| Form labels | Automated | axe-core |
| Error identification | Manual | - |
| Zoom/resize | Manual | - |

### 8.3 Localization Testing

**L10n Checklist:**
- [ ] All strings externalized
- [ ] Date/time formats correct per locale
- [ ] Number formats correct per locale
- [ ] Currency formats correct per locale
- [ ] RTL layout works (if applicable)
- [ ] String expansion doesn't break layout
- [ ] Character encoding correct
- [ ] Cultural appropriateness reviewed

---

## 9. Test Data Management

### 9.1 Test Data Strategy

**Types of Test Data:**
| Type | Source | Usage | Refresh |
|------|--------|-------|---------|
| Fixtures | Static files | Unit tests | On change |
| Generated | Factories/builders | Integration | Per test |
| Sanitized Prod | Production copy | E2E, manual | Weekly |
| Synthetic | Generated to spec | Load tests | On demand |

### 9.2 Test Data Factories

```rust
// tests/factories/source_factory.rs

use fake::{Fake, Faker};
use crate::models::Source;

pub struct SourceFactory {
    title: Option<String>,
    content: Option<String>,
    entity_count: Option<i32>,
}

impl SourceFactory {
    pub fn new() -> Self {
        Self {
            title: None,
            content: None,
            entity_count: None,
        }
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub fn with_entities(mut self, count: i32) -> Self {
        self.entity_count = Some(count);
        self
    }

    pub fn build(self) -> Source {
        Source {
            id: format!("source_{}", Faker.fake::<String>()[..8].to_string()),
            title: self.title.unwrap_or_else(|| Faker.fake()),
            content: self.content.unwrap_or_else(|| generate_paper_content()),
            entity_count: self.entity_count.unwrap_or(10),
            ..Default::default()
        }
    }
}

// Usage
let source = SourceFactory::new()
    .with_title("Test Paper")
    .with_entities(25)
    .build();
```

### 9.3 Data Sanitization

```sql
-- Sanitize production data for staging

-- Replace emails
UPDATE users SET
    email = 'user_' || id || '@test.latticeforge.io',
    name = 'Test User ' || id;

-- Truncate sensitive tables
TRUNCATE TABLE audit_logs;
TRUNCATE TABLE api_keys;
TRUNCATE TABLE password_reset_tokens;

-- Anonymize content (optional)
UPDATE sources SET
    content = regexp_replace(content, '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}', '[EMAIL]', 'g');
```

---

## 10. Continuous Improvement

### 10.1 Retrospective Topics

**Monthly QA Retrospective:**
1. What defects escaped to production?
2. What tests saved us?
3. What's causing flaky tests?
4. What areas need more coverage?
5. What processes are slowing us down?

### 10.2 Test Debt Tracking

| Item | Impact | Effort | Priority |
|------|--------|--------|----------|
| Fix flaky synthesis tests | High | Medium | P1 |
| Add graph visualization E2E | Medium | High | P2 |
| Improve test data setup | Medium | Medium | P2 |
| Add contract tests | High | High | P2 |
| Performance test baseline | High | Medium | P1 |

### 10.3 Skills Development

**QA Engineer Growth Areas:**
- Rust testing proficiency
- WASM testing techniques
- AI/ML output quality testing
- Performance engineering
- Security testing expertise
- Accessibility expertise

---

*Quality is not a phase—it's a mindset. This document should evolve with our product and processes. Review and update quarterly.*
