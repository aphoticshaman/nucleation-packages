# AUTOMATION_AND_WRAPPERS.skill.md

## Automation Engineering: Building Systems That Work While You Sleep

**Version**: 1.0
**Domain**: Automation, API Wrappers, Scripting, Workflow Orchestration
**Prerequisites**: CODING_BEST_PRACTICES skill, basic DevOps knowledge
**Output**: Reliable automated systems, clean API wrappers, efficient workflows

---

## 1. EXECUTIVE SUMMARY

Automation is leverage—doing work once that pays dividends forever. This skill covers patterns for building reliable automated systems, creating clean API wrappers, and orchestrating complex workflows. The goal: eliminate toil, reduce errors, and free humans for creative work.

**Core Principle**: Automate the repeatable. Improve what you automate. Monitor what you deploy.

---

## 2. AUTOMATION PHILOSOPHY

### 2.1 The Automation Decision Matrix

```
              LOW FREQUENCY    HIGH FREQUENCY
LOW EFFORT    |  Maybe        |  Definitely    |
              |  automate     |  automate      |
HIGH EFFORT   |  Don't        |  Maybe         |
              |  automate     |  automate      |

DECISION FACTORS:
├── Frequency: How often is this done?
├── Effort: How hard is automation vs manual?
├── Error rate: How often does manual fail?
├── Speed: How much faster is automation?
└── Learning: Does manual teach something valuable?

RULE: Automate if (frequency × manual_effort) > automation_effort
```

### 2.2 The Three Stages of Automation

```
STAGE 1: SCRIPT IT
├── One-off script that does the job
├── Hardcoded values, minimal error handling
├── Gets the job done, but fragile

STAGE 2: PARAMETERIZE IT
├── Configurable inputs
├── Error handling and logging
├── Reusable across contexts

STAGE 3: PRODUCTIONIZE IT
├── Monitoring and alerting
├── Retry logic and recovery
├── Documentation and tests
├── Deployment automation

MATCH STAGE TO NEED: Not everything needs Stage 3
```

### 2.3 Toil Identification

Recognize what should be automated:

```
TOIL CHARACTERISTICS:
├── Manual: Requires human to do it
├── Repetitive: Done multiple times
├── Automatable: Could be done by machine
├── Tactical: Keeps things running, no new value
├── Scales linearly: More load = more work

EXAMPLES:
├── Deploying code manually
├── Rotating logs/secrets
├── Provisioning environments
├── Responding to alerts with known fixes
├── Data entry from one system to another
```

---

## 3. API WRAPPER DESIGN

### 3.1 Wrapper Architecture

```typescript
// THREE-LAYER WRAPPER PATTERN

// Layer 1: HTTP Client (handles transport)
class HttpClient {
  constructor(baseUrl: string, defaultHeaders: Headers) {}
  async request(method: string, path: string, body?: unknown): Promise<Response>;
}

// Layer 2: API Client (handles API specifics)
class ApiClient {
  constructor(private http: HttpClient, private config: ApiConfig) {}

  private async request<T>(method: string, path: string, body?: unknown): Promise<T> {
    const response = await this.http.request(method, path, body);
    if (!response.ok) {
      throw new ApiError(response.status, await response.json());
    }
    return response.json() as T;
  }
}

// Layer 3: Resource Clients (domain-specific)
class UsersClient {
  constructor(private api: ApiClient) {}

  async getUser(id: string): Promise<User> {
    return this.api.get<User>(`/users/${id}`);
  }

  async createUser(data: CreateUserInput): Promise<User> {
    return this.api.post<User>('/users', data);
  }
}
```

### 3.2 Error Handling in Wrappers

```typescript
// TYPED ERROR HIERARCHY
class ApiError extends Error {
  constructor(
    public readonly statusCode: number,
    public readonly body: unknown,
    public readonly requestId?: string
  ) {
    super(`API Error ${statusCode}`);
  }
}

class RateLimitError extends ApiError {
  constructor(
    body: unknown,
    public readonly retryAfter: number
  ) {
    super(429, body);
  }
}

class AuthenticationError extends ApiError {
  constructor(body: unknown) {
    super(401, body);
  }
}

// AUTOMATIC RETRY WITH BACKOFF
async function withRetry<T>(
  operation: () => Promise<T>,
  options: RetryOptions = {}
): Promise<T> {
  const { maxRetries = 3, baseDelay = 1000 } = options;

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await operation();
    } catch (error) {
      if (!isRetryable(error) || attempt === maxRetries - 1) {
        throw error;
      }

      const delay = baseDelay * Math.pow(2, attempt);
      await sleep(delay + Math.random() * 1000);
    }
  }

  throw new Error('Unreachable');
}
```

### 3.3 Rate Limiting

```typescript
class RateLimiter {
  private tokens: number;
  private lastRefill: number;

  constructor(
    private readonly tokensPerSecond: number,
    private readonly maxTokens: number
  ) {
    this.tokens = maxTokens;
    this.lastRefill = Date.now();
  }

  async acquire(): Promise<void> {
    this.refill();

    if (this.tokens < 1) {
      const waitTime = (1 - this.tokens) / this.tokensPerSecond * 1000;
      await sleep(waitTime);
      this.refill();
    }

    this.tokens -= 1;
  }

  private refill(): void {
    const now = Date.now();
    const elapsed = (now - this.lastRefill) / 1000;
    this.tokens = Math.min(
      this.maxTokens,
      this.tokens + elapsed * this.tokensPerSecond
    );
    this.lastRefill = now;
  }
}

// USAGE IN API CLIENT
class ApiClient {
  private limiter = new RateLimiter(10, 100); // 10 req/s, burst of 100

  async request<T>(method: string, path: string): Promise<T> {
    await this.limiter.acquire();
    return this.http.request(method, path);
  }
}
```

---

## 4. WORKFLOW ORCHESTRATION

### 4.1 Pipeline Architecture

```python
# PIPELINE PATTERN
class Pipeline:
    def __init__(self, steps: list[Step]):
        self.steps = steps
        self.state = {}

    def run(self, initial_input: dict) -> PipelineResult:
        self.state = initial_input.copy()

        for step in self.steps:
            try:
                result = step.execute(self.state)
                self.state.update(result)
            except StepError as e:
                return PipelineResult(
                    success=False,
                    completed_steps=self.get_completed(),
                    error=e,
                    state=self.state
                )

        return PipelineResult(
            success=True,
            completed_steps=self.steps,
            state=self.state
        )

# STEP DEFINITION
class Step:
    def execute(self, state: dict) -> dict:
        """Execute step and return state updates."""
        raise NotImplementedError

    def rollback(self, state: dict) -> None:
        """Rollback this step's changes."""
        pass

# EXAMPLE PIPELINE
deploy_pipeline = Pipeline([
    RunTests(),
    BuildArtifact(),
    PushToRegistry(),
    UpdateKubernetes(),
    VerifyHealth(),
    NotifySlack(),
])
```

### 4.2 Idempotency

Operations should be safe to retry:

```python
class IdempotentOperation:
    """Operation that can be safely retried."""

    def __init__(self, operation_id: str):
        self.operation_id = operation_id

    def execute(self) -> Result:
        # Check if already completed
        if self.is_completed():
            return self.get_cached_result()

        # Execute with lock to prevent concurrent execution
        with self.acquire_lock():
            # Double-check after acquiring lock
            if self.is_completed():
                return self.get_cached_result()

            result = self._do_execute()
            self.mark_completed(result)
            return result

    def is_completed(self) -> bool:
        return cache.exists(f"op:{self.operation_id}")

    def mark_completed(self, result: Result) -> None:
        cache.set(f"op:{self.operation_id}", result, ttl=86400)

# IDEMPOTENCY PATTERNS
IDEMPOTENT:
├── Reading data
├── Setting to specific value
├── Deleting (if not exists = success)
├── Upsert operations

NON-IDEMPOTENT (make idempotent):
├── Incrementing counters → Use unique request IDs
├── Appending to list → Check for duplicates
├── Sending email → Track sent messages
```

### 4.3 State Management

```python
class WorkflowState:
    """Persistent state for long-running workflows."""

    def __init__(self, workflow_id: str, store: StateStore):
        self.workflow_id = workflow_id
        self.store = store

    def checkpoint(self, step_name: str, data: dict) -> None:
        """Save checkpoint after completing step."""
        self.store.save({
            'workflow_id': self.workflow_id,
            'step': step_name,
            'data': data,
            'timestamp': datetime.now(),
        })

    def resume_from_checkpoint(self) -> tuple[str, dict]:
        """Get last checkpoint for resuming."""
        checkpoint = self.store.get_latest(self.workflow_id)
        if checkpoint:
            return checkpoint['step'], checkpoint['data']
        return None, {}

    def clear(self) -> None:
        """Clear all checkpoints for this workflow."""
        self.store.delete_all(self.workflow_id)
```

---

## 5. SCRIPTING PATTERNS

### 5.1 Script Structure

```bash
#!/bin/bash
set -euo pipefail  # Exit on error, undefined vars, pipe failures

#######################################
# Script Description
# Globals:
#   REQUIRED_VAR - description
# Arguments:
#   $1 - first argument description
# Outputs:
#   Writes result to stdout
# Returns:
#   0 on success, non-zero on error
#######################################

# Constants
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LOG_FILE="${SCRIPT_DIR}/script.log"

# Logging functions
log_info() { echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG_FILE"; }
log_error() { echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG_FILE" >&2; }
log_debug() { [[ "${DEBUG:-}" == "true" ]] && echo "[DEBUG] $*"; }

# Cleanup on exit
cleanup() {
  local exit_code=$?
  # Cleanup commands here
  exit $exit_code
}
trap cleanup EXIT

# Validation
validate_args() {
  if [[ $# -lt 1 ]]; then
    log_error "Usage: $0 <argument>"
    exit 1
  fi
}

# Main function
main() {
  validate_args "$@"
  log_info "Starting script with args: $*"
  # Main logic here
  log_info "Script completed successfully"
}

main "$@"
```

### 5.2 Cross-Platform Scripting

```python
#!/usr/bin/env python3
"""Cross-platform automation script."""

import os
import sys
import platform
import subprocess
from pathlib import Path

def get_platform_config():
    """Get platform-specific configuration."""
    system = platform.system().lower()

    configs = {
        'darwin': {
            'home': Path.home(),
            'shell': '/bin/zsh',
            'package_manager': 'brew',
        },
        'linux': {
            'home': Path.home(),
            'shell': '/bin/bash',
            'package_manager': 'apt',
        },
        'windows': {
            'home': Path(os.environ.get('USERPROFILE', 'C:\\Users\\Default')),
            'shell': 'powershell.exe',
            'package_manager': 'choco',
        },
    }

    return configs.get(system, configs['linux'])

def run_command(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run command with proper error handling."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            **kwargs
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(cmd)}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise
```

### 5.3 Configuration Management

```python
from dataclasses import dataclass
from typing import Optional
import os
import json
from pathlib import Path

@dataclass
class Config:
    """Application configuration with multiple sources."""
    api_key: str
    api_url: str = "https://api.example.com"
    timeout: int = 30
    debug: bool = False
    retries: int = 3

    @classmethod
    def load(cls) -> 'Config':
        """Load config from multiple sources with priority."""
        # Priority: ENV > config file > defaults

        config_file = Path.home() / '.config' / 'myapp' / 'config.json'
        file_config = {}

        if config_file.exists():
            file_config = json.loads(config_file.read_text())

        return cls(
            api_key=os.environ.get('API_KEY', file_config.get('api_key', '')),
            api_url=os.environ.get('API_URL', file_config.get('api_url', cls.api_url)),
            timeout=int(os.environ.get('TIMEOUT', file_config.get('timeout', cls.timeout))),
            debug=os.environ.get('DEBUG', str(file_config.get('debug', cls.debug))).lower() == 'true',
            retries=int(os.environ.get('RETRIES', file_config.get('retries', cls.retries))),
        )

    def validate(self) -> None:
        """Validate configuration."""
        if not self.api_key:
            raise ValueError("API_KEY is required")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
```

---

## 6. MONITORING AND OBSERVABILITY

### 6.1 Logging Best Practices

```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    """Logger with structured JSON output."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.context = {}

    def with_context(self, **kwargs) -> 'StructuredLogger':
        """Add context to all subsequent logs."""
        new_logger = StructuredLogger(self.logger.name)
        new_logger.context = {**self.context, **kwargs}
        return new_logger

    def info(self, message: str, **kwargs) -> None:
        self._log('INFO', message, kwargs)

    def error(self, message: str, error: Exception = None, **kwargs) -> None:
        if error:
            kwargs['error'] = str(error)
            kwargs['error_type'] = type(error).__name__
        self._log('ERROR', message, kwargs)

    def _log(self, level: str, message: str, extra: dict) -> None:
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level,
            'message': message,
            **self.context,
            **extra,
        }
        print(json.dumps(log_entry))

# USAGE
logger = StructuredLogger('automation')
job_logger = logger.with_context(job_id='abc123', workflow='deploy')
job_logger.info('Starting deployment', environment='production')
```

### 6.2 Metrics Collection

```python
from dataclasses import dataclass, field
from typing import Dict
import time

@dataclass
class Metrics:
    """Collect and report automation metrics."""
    counters: Dict[str, int] = field(default_factory=dict)
    timers: Dict[str, list] = field(default_factory=dict)

    def increment(self, name: str, value: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + value

    def timer(self, name: str):
        """Context manager for timing operations."""
        return Timer(self, name)

    def report(self) -> dict:
        """Generate metrics report."""
        return {
            'counters': self.counters,
            'timers': {
                name: {
                    'count': len(values),
                    'avg_ms': sum(values) / len(values) if values else 0,
                    'max_ms': max(values) if values else 0,
                }
                for name, values in self.timers.items()
            }
        }

class Timer:
    def __init__(self, metrics: Metrics, name: str):
        self.metrics = metrics
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        elapsed_ms = (time.time() - self.start) * 1000
        if self.name not in self.metrics.timers:
            self.metrics.timers[self.name] = []
        self.metrics.timers[self.name].append(elapsed_ms)
```

### 6.3 Alerting

```python
class AlertManager:
    """Manage alerts for automation failures."""

    def __init__(self, channels: list[AlertChannel]):
        self.channels = channels
        self.alert_history = []

    def alert(self, severity: str, message: str, details: dict = None) -> None:
        """Send alert to all channels."""
        alert = {
            'severity': severity,
            'message': message,
            'details': details or {},
            'timestamp': datetime.utcnow().isoformat(),
        }

        # Deduplicate similar alerts
        if self._is_duplicate(alert):
            return

        self.alert_history.append(alert)

        for channel in self.channels:
            try:
                channel.send(alert)
            except Exception as e:
                # Don't let alerting failure stop execution
                print(f"Failed to send alert via {channel}: {e}")

    def _is_duplicate(self, alert: dict) -> bool:
        """Check if similar alert was sent recently."""
        cutoff = datetime.utcnow() - timedelta(minutes=5)

        for recent in self.alert_history:
            if (recent['message'] == alert['message'] and
                datetime.fromisoformat(recent['timestamp']) > cutoff):
                return True
        return False
```

---

## 7. TESTING AUTOMATION

### 7.1 Test Isolation

```python
import pytest
from unittest.mock import Mock, patch

class TestApiClient:
    """Test API client with mocked HTTP."""

    @pytest.fixture
    def mock_http(self):
        """Create mock HTTP client."""
        mock = Mock()
        mock.request.return_value = Mock(
            ok=True,
            json=lambda: {'id': '123', 'name': 'Test'}
        )
        return mock

    @pytest.fixture
    def client(self, mock_http):
        """Create API client with mocked dependencies."""
        return ApiClient(http=mock_http, config=ApiConfig(api_key='test'))

    def test_get_user_returns_user(self, client, mock_http):
        """Test successful user retrieval."""
        user = client.get_user('123')

        assert user['id'] == '123'
        mock_http.request.assert_called_once_with('GET', '/users/123')

    def test_get_user_handles_404(self, client, mock_http):
        """Test handling of not found error."""
        mock_http.request.return_value = Mock(ok=False, status_code=404)

        with pytest.raises(NotFoundError):
            client.get_user('nonexistent')
```

### 7.2 Integration Testing

```python
@pytest.mark.integration
class TestPipelineIntegration:
    """Integration tests for deployment pipeline."""

    @pytest.fixture
    def test_environment(self):
        """Set up isolated test environment."""
        env = TestEnvironment.create()
        yield env
        env.cleanup()

    def test_full_pipeline_execution(self, test_environment):
        """Test full pipeline runs successfully."""
        pipeline = DeployPipeline(
            environment=test_environment,
            config=PipelineConfig(timeout=60)
        )

        result = pipeline.run({
            'artifact': 'test-app:latest',
            'target': 'staging'
        })

        assert result.success
        assert test_environment.get_deployment('test-app').running
```

---

## 8. IMPLEMENTATION CHECKLIST

### For Every Automation:
- [ ] Clear purpose and scope
- [ ] Idempotent where possible
- [ ] Proper error handling
- [ ] Logging at key points
- [ ] Retry logic for transient failures
- [ ] Timeout handling
- [ ] Cleanup on failure

### For API Wrappers:
- [ ] Rate limiting
- [ ] Retry with backoff
- [ ] Typed errors
- [ ] Request/response logging
- [ ] Timeout configuration
- [ ] Authentication handling

### For Workflows:
- [ ] State management
- [ ] Checkpoint/resume capability
- [ ] Rollback on failure
- [ ] Progress reporting
- [ ] Alerting on failure

---

## 9. AUTOMATION ANTI-PATTERNS

```
DON'T: Automate without understanding the manual process
DO: Master manual first, then automate

DON'T: Ignore errors in automation
DO: Handle, log, and alert on all errors

DON'T: Run automation without monitoring
DO: Observe every automated system

DON'T: Hardcode secrets in scripts
DO: Use secret management (env vars, vaults)

DON'T: Skip testing automation code
DO: Test automation like production code

DON'T: Over-automate (everything must be automatic)
DO: Automate where it saves time and reduces errors

DON'T: Forget about maintenance
DO: Plan for updates, debugging, evolution
```

---

**Remember**: Automation is an investment. Invest wisely in what will pay dividends. Build for reliability. Monitor everything. And remember that the best automation is invisible—it just works.

Automate once. Run forever.
