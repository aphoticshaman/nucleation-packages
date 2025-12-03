"""
APT-Resilient Security Layer for LatticeForge Inference.

Implements:
- HMAC request signing (prevents replay, tampering)
- Nonce tracking (prevents replay attacks)
- Rate limiting with anomaly detection
- Request/response audit logging
- Canary tokens for breach detection

Threat model: Nation-state APT with persistent access attempts.
"""

import hmac
import hashlib
import time
import json
import os
import secrets
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import logging

# Configure secure logging (no sensitive data)
logger = logging.getLogger("latticeforge.security")


@dataclass
class SecurityConfig:
    """Security configuration."""

    # HMAC signing
    signing_key: str = field(default_factory=lambda: os.getenv("SIGNING_KEY", ""))
    signature_ttl_seconds: int = 300  # 5 minute window

    # Nonce tracking (prevent replay)
    nonce_window_seconds: int = 600  # Track nonces for 10 mins

    # Rate limiting
    requests_per_minute: int = 60
    requests_per_hour: int = 500
    burst_threshold: int = 20  # Requests in 10 seconds = anomaly

    # Anomaly detection
    anomaly_score_threshold: float = 0.7

    # Audit
    audit_log_path: Optional[str] = None

    # Canary
    canary_token: str = field(default_factory=lambda: os.getenv("CANARY_TOKEN", ""))


class NonceTracker:
    """
    Track used nonces to prevent replay attacks.

    Memory-efficient sliding window implementation.
    """

    def __init__(self, window_seconds: int = 600):
        self.window = window_seconds
        self.nonces: Dict[str, float] = {}  # nonce -> timestamp
        self._last_cleanup = time.time()

    def check_and_add(self, nonce: str) -> bool:
        """
        Check if nonce is fresh and add it.

        Returns True if nonce is valid (not seen before).
        Returns False if replay detected.
        """
        now = time.time()

        # Periodic cleanup
        if now - self._last_cleanup > 60:
            self._cleanup(now)

        # Check for replay
        if nonce in self.nonces:
            logger.warning(f"Replay attack detected: nonce reuse")
            return False

        # Add nonce
        self.nonces[nonce] = now
        return True

    def _cleanup(self, now: float) -> None:
        """Remove expired nonces."""
        cutoff = now - self.window
        self.nonces = {k: v for k, v in self.nonces.items() if v > cutoff}
        self._last_cleanup = now


class RateLimiter:
    """
    Rate limiter with anomaly detection.

    Tracks request patterns and flags suspicious behavior.
    """

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.minute_counts: Dict[str, List[float]] = defaultdict(list)
        self.hour_counts: Dict[str, List[float]] = defaultdict(list)
        self.burst_tracker: Dict[str, List[float]] = defaultdict(list)

    def check(self, client_id: str) -> Tuple[bool, float]:
        """
        Check if request is allowed.

        Returns (allowed, anomaly_score).
        """
        now = time.time()

        # Clean old entries
        minute_cutoff = now - 60
        hour_cutoff = now - 3600
        burst_cutoff = now - 10

        self.minute_counts[client_id] = [
            t for t in self.minute_counts[client_id] if t > minute_cutoff
        ]
        self.hour_counts[client_id] = [
            t for t in self.hour_counts[client_id] if t > hour_cutoff
        ]
        self.burst_tracker[client_id] = [
            t for t in self.burst_tracker[client_id] if t > burst_cutoff
        ]

        # Check limits
        minute_count = len(self.minute_counts[client_id])
        hour_count = len(self.hour_counts[client_id])
        burst_count = len(self.burst_tracker[client_id])

        # Compute anomaly score
        anomaly_score = 0.0

        if minute_count > self.config.requests_per_minute * 0.8:
            anomaly_score += 0.3
        if hour_count > self.config.requests_per_hour * 0.8:
            anomaly_score += 0.2
        if burst_count > self.config.burst_threshold:
            anomaly_score += 0.5
            logger.warning(f"Burst detected from {client_id}: {burst_count} requests in 10s")

        # Hard limits
        if minute_count >= self.config.requests_per_minute:
            logger.warning(f"Rate limit exceeded (minute): {client_id}")
            return False, 1.0

        if hour_count >= self.config.requests_per_hour:
            logger.warning(f"Rate limit exceeded (hour): {client_id}")
            return False, 1.0

        # Record request
        self.minute_counts[client_id].append(now)
        self.hour_counts[client_id].append(now)
        self.burst_tracker[client_id].append(now)

        return True, anomaly_score


class RequestSigner:
    """
    HMAC-SHA256 request signing.

    Prevents tampering and verifies origin.
    """

    def __init__(self, signing_key: str, ttl_seconds: int = 300):
        self.key = signing_key.encode() if signing_key else b""
        self.ttl = ttl_seconds

    def sign(self, payload: Dict[str, Any], timestamp: Optional[int] = None) -> str:
        """
        Sign a request payload.

        Returns signature string.
        """
        if not self.key:
            raise ValueError("Signing key not configured")

        ts = timestamp or int(time.time())
        nonce = secrets.token_hex(16)

        # Canonical string
        canonical = f"{ts}:{nonce}:{json.dumps(payload, sort_keys=True)}"

        # HMAC-SHA256
        signature = hmac.new(
            self.key,
            canonical.encode(),
            hashlib.sha256
        ).hexdigest()

        return f"{ts}:{nonce}:{signature}"

    def verify(
        self,
        payload: Dict[str, Any],
        signature: str,
        nonce_tracker: NonceTracker
    ) -> Tuple[bool, str]:
        """
        Verify a signed request.

        Returns (valid, error_message).
        """
        if not self.key:
            return False, "Signing key not configured"

        try:
            parts = signature.split(":")
            if len(parts) != 3:
                return False, "Invalid signature format"

            ts_str, nonce, provided_sig = parts
            ts = int(ts_str)

            # Check timestamp (prevent replay of old requests)
            now = int(time.time())
            if abs(now - ts) > self.ttl:
                return False, "Request expired"

            # Check nonce (prevent replay)
            if not nonce_tracker.check_and_add(nonce):
                return False, "Nonce already used (replay attack)"

            # Verify signature
            canonical = f"{ts}:{nonce}:{json.dumps(payload, sort_keys=True)}"
            expected_sig = hmac.new(
                self.key,
                canonical.encode(),
                hashlib.sha256
            ).hexdigest()

            if not hmac.compare_digest(provided_sig, expected_sig):
                return False, "Invalid signature"

            return True, ""

        except Exception as e:
            return False, f"Verification error: {str(e)}"


class AuditLogger:
    """
    Secure audit logging for compliance and forensics.

    Logs are append-only and include integrity hashes.
    """

    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path
        self._prev_hash = "0" * 64  # Genesis hash

    def log(
        self,
        event_type: str,
        client_id: str,
        request_hash: str,
        response_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log an audit event with integrity chain.

        Returns the event hash.
        """
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type,
            "client_id": client_id,
            "request_hash": request_hash,
            "response_hash": response_hash,
            "metadata": metadata or {},
            "prev_hash": self._prev_hash,
        }

        # Compute event hash (chain integrity)
        event_str = json.dumps(event, sort_keys=True)
        event_hash = hashlib.sha256(event_str.encode()).hexdigest()
        event["event_hash"] = event_hash

        # Update chain
        self._prev_hash = event_hash

        # Write to log
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(event) + "\n")

        logger.info(f"Audit: {event_type} from {client_id}")

        return event_hash

    @staticmethod
    def hash_content(content: str) -> str:
        """Hash content for audit trail."""
        return hashlib.sha256(content.encode()).hexdigest()


class CanaryChecker:
    """
    Canary token detection.

    Detects if adversary is probing with known canary values.
    """

    def __init__(self, canary_token: str):
        self.canary = canary_token
        self.triggered = False

    def check(self, content: str) -> bool:
        """
        Check if content contains canary.

        Returns True if canary detected (breach indicator).
        """
        if self.canary and self.canary in content:
            self.triggered = True
            logger.critical("CANARY TRIGGERED - POSSIBLE BREACH")
            return True
        return False


class SecurityMiddleware:
    """
    Combined security middleware for inference server.

    Usage:
        security = SecurityMiddleware(config)

        # In request handler:
        allowed, error = security.validate_request(
            client_id="user_123",
            payload=request_data,
            signature=request.headers.get("X-Signature")
        )
        if not allowed:
            return error_response(error)

        # After processing:
        security.log_request(client_id, request_data, response_data)
    """

    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()

        self.nonce_tracker = NonceTracker(self.config.nonce_window_seconds)
        self.rate_limiter = RateLimiter(self.config)
        self.signer = RequestSigner(
            self.config.signing_key,
            self.config.signature_ttl_seconds
        )
        self.audit = AuditLogger(self.config.audit_log_path)
        self.canary = CanaryChecker(self.config.canary_token)

    def validate_request(
        self,
        client_id: str,
        payload: Dict[str, Any],
        signature: Optional[str] = None,
        require_signature: bool = True
    ) -> Tuple[bool, str]:
        """
        Validate an incoming request.

        Returns (allowed, error_message).
        """
        # Rate limiting
        allowed, anomaly_score = self.rate_limiter.check(client_id)
        if not allowed:
            return False, "Rate limit exceeded"

        if anomaly_score > self.config.anomaly_score_threshold:
            logger.warning(f"High anomaly score for {client_id}: {anomaly_score}")
            # Could trigger additional verification here

        # Signature verification (if required)
        if require_signature:
            if not signature:
                return False, "Missing signature"

            valid, error = self.signer.verify(payload, signature, self.nonce_tracker)
            if not valid:
                return False, f"Signature verification failed: {error}"

        # Canary check
        payload_str = json.dumps(payload)
        if self.canary.check(payload_str):
            return False, "Request rejected"

        return True, ""

    def log_request(
        self,
        client_id: str,
        request: Dict[str, Any],
        response: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log request/response for audit trail."""
        request_hash = self.audit.hash_content(json.dumps(request, sort_keys=True))
        response_hash = self.audit.hash_content(json.dumps(response, sort_keys=True))

        self.audit.log(
            event_type="inference",
            client_id=client_id,
            request_hash=request_hash,
            response_hash=response_hash,
            metadata=metadata
        )

    def sign_request(self, payload: Dict[str, Any]) -> str:
        """Sign an outgoing request (for client use)."""
        return self.signer.sign(payload)


# FastAPI middleware integration
def create_security_dependency(config: Optional[SecurityConfig] = None):
    """
    Create FastAPI dependency for security middleware.

    Usage:
        security_dep = create_security_dependency()

        @app.post("/generate")
        async def generate(
            request: GenerateRequest,
            security: SecurityMiddleware = Depends(security_dep)
        ):
            ...
    """
    middleware = SecurityMiddleware(config)

    def get_security():
        return middleware

    return get_security
