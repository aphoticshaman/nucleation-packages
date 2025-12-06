# LatticeForge Backend API Contract Specification

## Document Purpose

This specification defines the complete API contract between LatticeForge's frontend and backend systems. It covers RESTful endpoints, real-time protocols, data schemas, error handling, pagination, and versioning strategies. Backend engineers should use this as the authoritative reference for implementing and maintaining API services.

---

## 1. API Architecture Overview

### 1.1 Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| HTTP Server | Axum (Rust) | REST endpoints, routing, middleware |
| WebSocket | tokio-tungstenite | Real-time subscriptions, streaming |
| Serialization | serde, JSON | Request/response encoding |
| Database | PostgreSQL + PostGIS | Primary data store |
| Cache | Redis | Session cache, rate limiting |
| Search | Tantivy or Meilisearch | Full-text search |
| Queue | RabbitMQ or Redis Streams | Background job processing |
| Object Storage | S3-compatible | PDF storage, exports |

### 1.2 Architectural Principles

1. **Contract-First Design**: API contracts defined before implementation
2. **Stateless Operations**: No server-side session state in REST handlers
3. **Idempotency**: PUT/DELETE operations safe to retry
4. **Pagination by Default**: All list endpoints paginated
5. **Consistent Error Format**: Uniform error response structure
6. **Versioned Evolution**: Breaking changes require version bump

### 1.3 Base URL Structure

```
Production:  https://api.latticeforge.io/v1
Staging:     https://api.staging.latticeforge.io/v1
Development: http://localhost:3001/v1
```

---

## 2. Authentication and Authorization

### 2.1 Authentication Methods

**JWT Bearer Tokens (Primary)**
```http
Authorization: Bearer eyJhbGciOiJSUzI1NiIs...
```

JWT payload structure:
```json
{
  "sub": "user_abc123",
  "email": "user@example.com",
  "org_id": "org_xyz789",
  "role": "member",
  "iat": 1699900000,
  "exp": 1699986400
}
```

**API Keys (Service-to-Service)**
```http
X-API-Key: lf_live_sk_abc123...
```

API key format:
- Prefix: `lf_live_` (production) or `lf_test_` (development)
- Type: `sk_` (secret key) or `pk_` (public key)
- Identifier: 32-character random string

### 2.2 Authorization Model

**Role-Based Access Control (RBAC):**

| Role | Capabilities |
|------|-------------|
| owner | Full workspace control, billing, member management |
| admin | All content operations, invite members |
| member | Create/edit own content, view shared content |
| viewer | Read-only access to shared content |

**Resource-Level Permissions:**
```json
{
  "resource_type": "stream",
  "resource_id": "stream_abc123",
  "permissions": ["read", "write", "delete", "share"]
}
```

### 2.3 Authentication Endpoints

**POST /v1/auth/login**

Request:
```json
{
  "email": "user@example.com",
  "password": "securepassword123"
}
```

Response (200):
```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIs...",
  "refresh_token": "rt_abc123...",
  "token_type": "Bearer",
  "expires_in": 86400,
  "user": {
    "id": "user_abc123",
    "email": "user@example.com",
    "name": "Jane Researcher"
  }
}
```

**POST /v1/auth/refresh**

Request:
```json
{
  "refresh_token": "rt_abc123..."
}
```

Response (200):
```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIs...",
  "expires_in": 86400
}
```

**POST /v1/auth/logout**

Request: (empty body, uses Authorization header)

Response (204): No content

---

## 3. Core Resource Endpoints

### 3.1 Research Streams

**Stream Object Schema:**
```json
{
  "id": "stream_abc123",
  "name": "CRISPR Delivery Mechanisms",
  "description": "Literature review on viral and non-viral CRISPR delivery",
  "visibility": "team",
  "owner_id": "user_abc123",
  "workspace_id": "ws_xyz789",
  "source_count": 24,
  "insight_count": 8,
  "created_at": "2024-03-15T10:30:00Z",
  "updated_at": "2024-03-18T14:22:00Z",
  "settings": {
    "auto_generate_insights": true,
    "notification_preferences": {
      "new_insights": true,
      "source_processed": false
    }
  }
}
```

**GET /v1/streams**

List user's streams with pagination and filtering.

Query Parameters:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| page | integer | 1 | Page number |
| per_page | integer | 20 | Items per page (max 100) |
| sort | string | "updated_at" | Sort field |
| order | string | "desc" | Sort order (asc/desc) |
| visibility | string | null | Filter by visibility |
| q | string | null | Search query |

Response (200):
```json
{
  "data": [
    { /* stream object */ },
    { /* stream object */ }
  ],
  "meta": {
    "page": 1,
    "per_page": 20,
    "total": 47,
    "total_pages": 3
  }
}
```

**POST /v1/streams**

Create a new stream.

Request:
```json
{
  "name": "Protein Folding Dynamics",
  "description": "Exploring AlphaFold and molecular dynamics",
  "visibility": "private"
}
```

Response (201):
```json
{
  "data": { /* stream object */ }
}
```

**GET /v1/streams/:stream_id**

Response (200):
```json
{
  "data": { /* stream object with expanded details */ }
}
```

**PATCH /v1/streams/:stream_id**

Partial update of stream properties.

Request:
```json
{
  "name": "Updated Stream Name",
  "settings": {
    "auto_generate_insights": false
  }
}
```

Response (200):
```json
{
  "data": { /* updated stream object */ }
}
```

**DELETE /v1/streams/:stream_id**

Response (204): No content

### 3.2 Sources

**Source Object Schema:**
```json
{
  "id": "source_abc123",
  "stream_id": "stream_xyz789",
  "type": "paper",
  "status": "processed",
  "title": "CRISPR-Cas9 genome editing: A review",
  "authors": [
    {"name": "Chen, L.", "affiliation": "Stanford University"}
  ],
  "publication": {
    "venue": "Nature Reviews Genetics",
    "year": 2023,
    "volume": "24",
    "pages": "123-145",
    "doi": "10.1038/s41576-023-00586-w"
  },
  "url": "https://doi.org/10.1038/s41576-023-00586-w",
  "file_path": "sources/source_abc123/original.pdf",
  "content_hash": "sha256:a1b2c3d4...",
  "extracted_text_length": 45230,
  "entity_count": 127,
  "created_at": "2024-03-15T10:30:00Z",
  "processed_at": "2024-03-15T10:32:15Z"
}
```

**GET /v1/streams/:stream_id/sources**

Query Parameters:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| page | integer | 1 | Page number |
| per_page | integer | 50 | Items per page |
| status | string | null | Filter: pending, processing, processed, failed |
| type | string | null | Filter: paper, article, book, preprint |
| q | string | null | Search in title/authors |

Response (200):
```json
{
  "data": [ /* source objects */ ],
  "meta": { /* pagination */ }
}
```

**POST /v1/streams/:stream_id/sources**

Add source by URL or file upload.

URL Import:
```json
{
  "url": "https://arxiv.org/abs/2301.00000",
  "type": "preprint"
}
```

File Upload: `multipart/form-data` with file field

Response (202 Accepted):
```json
{
  "data": {
    "id": "source_abc123",
    "status": "pending",
    "processing_job_id": "job_xyz789"
  }
}
```

**GET /v1/sources/:source_id**

Response includes extracted content:
```json
{
  "data": {
    /* source object */,
    "extracted_sections": [
      {"title": "Abstract", "content": "..."},
      {"title": "Introduction", "content": "..."}
    ],
    "entities": [
      {"id": "ent_123", "name": "CRISPR-Cas9", "type": "technology", "mention_count": 47}
    ]
  }
}
```

**DELETE /v1/sources/:source_id**

Response (204): No content

### 3.3 Syntheses

**Synthesis Object Schema:**
```json
{
  "id": "synth_abc123",
  "stream_id": "stream_xyz789",
  "title": "Overview of CRISPR Delivery Methods",
  "content": "# Overview\n\nRecent advances in CRISPR delivery...",
  "format": "markdown",
  "generation_params": {
    "type": "overview",
    "source_ids": ["source_1", "source_2"],
    "focus_prompt": null,
    "length": "standard"
  },
  "citations": [
    {"marker": "[1]", "source_id": "source_1", "text": "Chen et al., 2023"}
  ],
  "version": 3,
  "created_at": "2024-03-16T09:00:00Z",
  "updated_at": "2024-03-16T09:15:00Z"
}
```

**POST /v1/streams/:stream_id/syntheses**

Generate new synthesis.

Request:
```json
{
  "source_ids": ["source_1", "source_2", "source_3"],
  "type": "comparison",
  "focus_prompt": "Focus on delivery efficiency metrics",
  "length": "comprehensive"
}
```

Response (202 Accepted):
```json
{
  "data": {
    "id": "synth_abc123",
    "status": "generating",
    "stream_url": "/v1/syntheses/synth_abc123/stream"
  }
}
```

**GET /v1/syntheses/:synthesis_id/stream**

Server-Sent Events stream for generation progress:
```
event: token
data: {"text": "Recent "}

event: token
data: {"text": "advances "}

event: citation
data: {"marker": "[1]", "source_id": "source_1"}

event: complete
data: {"synthesis_id": "synth_abc123"}
```

**GET /v1/syntheses/:synthesis_id/versions**

List version history:
```json
{
  "data": [
    {"version": 3, "updated_at": "2024-03-16T09:15:00Z", "updated_by": "user_abc"},
    {"version": 2, "updated_at": "2024-03-16T09:10:00Z", "updated_by": "system"},
    {"version": 1, "updated_at": "2024-03-16T09:00:00Z", "updated_by": "system"}
  ]
}
```

### 3.4 Insights

**Insight Object Schema:**
```json
{
  "id": "insight_abc123",
  "stream_id": "stream_xyz789",
  "statement": "The protein folding mechanism in Chen 2023 may explain anomalous binding in Williams 2022",
  "confidence": 0.85,
  "confidence_category": "high",
  "evidence": [
    {
      "source_id": "source_1",
      "excerpt": "The folding pathway involves...",
      "relevance": 0.92
    }
  ],
  "entities": [
    {"id": "ent_1", "name": "protein folding", "type": "concept"}
  ],
  "generation_method": "cross_source_analysis",
  "status": "active",
  "user_feedback": null,
  "created_at": "2024-03-17T11:00:00Z"
}
```

**GET /v1/streams/:stream_id/insights**

Query Parameters:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| page | integer | 1 | Page number |
| per_page | integer | 20 | Items per page |
| status | string | "active" | Filter: active, saved, dismissed |
| confidence_min | float | 0 | Minimum confidence (0-1) |
| entity_id | string | null | Filter by related entity |
| sort | string | "created_at" | Sort field |

Response (200):
```json
{
  "data": [ /* insight objects */ ],
  "meta": { /* pagination */ }
}
```

**POST /v1/streams/:stream_id/insights/generate**

Trigger insight generation.

Request:
```json
{
  "mode": "cross_source",
  "source_ids": ["source_1", "source_2"],
  "focus": "contradictions"
}
```

Response (202 Accepted):
```json
{
  "data": {
    "job_id": "job_xyz789",
    "estimated_insights": 3
  }
}
```

**PATCH /v1/insights/:insight_id**

Update insight status (save/dismiss).

Request:
```json
{
  "status": "saved",
  "user_note": "Important finding for chapter 3"
}
```

**POST /v1/insights/:insight_id/feedback**

Provide feedback on insight quality.

Request:
```json
{
  "rating": "helpful",
  "comment": "This connection wasn't obvious from reading"
}
```

### 3.5 Entities

**Entity Object Schema:**
```json
{
  "id": "entity_abc123",
  "name": "CRISPR-Cas9",
  "canonical_name": "CRISPR-Cas9",
  "type": "technology",
  "aliases": ["CRISPR/Cas9", "Cas9"],
  "description": "A gene-editing technology...",
  "external_ids": {
    "wikidata": "Q24298912",
    "mesh": "D000071420"
  },
  "occurrence_count": 147,
  "stream_count": 12,
  "first_seen_at": "2024-01-10T08:00:00Z",
  "last_seen_at": "2024-03-18T14:00:00Z"
}
```

**GET /v1/entities**

Global entity search across workspace.

Query Parameters:
| Parameter | Type | Description |
|-----------|------|-------------|
| q | string | Search query (required) |
| type | string | Filter by entity type |
| limit | integer | Max results (default 20) |

Response (200):
```json
{
  "data": [ /* entity objects */ ]
}
```

**GET /v1/entities/:entity_id**

Detailed entity with relationships:
```json
{
  "data": {
    /* entity object */,
    "relationships": [
      {
        "related_entity_id": "entity_xyz",
        "relationship_type": "related_to",
        "strength": 0.78,
        "source_count": 5
      }
    ],
    "top_sources": [
      {"source_id": "source_1", "mention_count": 23}
    ]
  }
}
```

### 3.6 Graph Operations

**GET /v1/streams/:stream_id/graph**

Get graph data for visualization.

Query Parameters:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| center_id | string | null | Node to center on |
| depth | integer | 2 | Traversal depth (1-3) |
| node_types | string | "all" | Comma-separated types |
| min_edge_weight | float | 0.3 | Minimum relationship strength |

Response (200):
```json
{
  "data": {
    "nodes": [
      {
        "id": "source_1",
        "type": "source",
        "label": "Chen et al., 2023",
        "metadata": { /* source summary */ }
      },
      {
        "id": "entity_1",
        "type": "entity",
        "label": "CRISPR-Cas9",
        "metadata": { /* entity summary */ }
      }
    ],
    "edges": [
      {
        "source": "source_1",
        "target": "entity_1",
        "type": "mentions",
        "weight": 0.85
      }
    ],
    "clusters": [
      {
        "id": "cluster_1",
        "label": "Delivery Methods",
        "node_ids": ["entity_1", "entity_2"]
      }
    ]
  }
}
```

**POST /v1/streams/:stream_id/graph/paths**

Find paths between nodes.

Request:
```json
{
  "from_id": "entity_abc",
  "to_id": "entity_xyz",
  "max_hops": 3
}
```

Response (200):
```json
{
  "data": {
    "paths": [
      {
        "nodes": ["entity_abc", "source_1", "entity_xyz"],
        "edges": [
          {"source": "entity_abc", "target": "source_1", "type": "mentioned_in"},
          {"source": "source_1", "target": "entity_xyz", "type": "mentions"}
        ],
        "total_weight": 1.65
      }
    ]
  }
}
```

---

## 4. Real-Time APIs

### 4.1 WebSocket Connection

**Connection URL:**
```
wss://api.latticeforge.io/v1/ws?token=<jwt_token>
```

**Connection Lifecycle:**
```
Client                          Server
   |                              |
   |------ Connect + Token ------>|
   |<----- Connection ACK --------|
   |                              |
   |------ Subscribe Stream ----->|
   |<----- Subscription ACK ------|
   |                              |
   |<----- Event: source.added ---|
   |<----- Event: insight.new ----|
   |                              |
   |------ Unsubscribe ---------->|
   |<----- Unsubscribe ACK -------|
   |                              |
   |------ Disconnect ----------->|
```

### 4.2 Message Protocol

**Client → Server Messages:**

Subscribe to stream updates:
```json
{
  "type": "subscribe",
  "channel": "stream:stream_abc123",
  "request_id": "req_1"
}
```

Unsubscribe:
```json
{
  "type": "unsubscribe",
  "channel": "stream:stream_abc123",
  "request_id": "req_2"
}
```

Ping (keepalive):
```json
{
  "type": "ping",
  "timestamp": 1699900000
}
```

**Server → Client Messages:**

Subscription confirmation:
```json
{
  "type": "subscribed",
  "channel": "stream:stream_abc123",
  "request_id": "req_1"
}
```

Event notification:
```json
{
  "type": "event",
  "channel": "stream:stream_abc123",
  "event": "source.processed",
  "data": {
    "source_id": "source_xyz",
    "entity_count": 42
  },
  "timestamp": 1699900100
}
```

Error:
```json
{
  "type": "error",
  "code": "subscription_failed",
  "message": "Access denied to stream",
  "request_id": "req_1"
}
```

### 4.3 Event Types

**Stream Events:**
| Event | Description | Payload |
|-------|-------------|---------|
| source.added | New source added | source_id, title |
| source.processing | Processing started | source_id, stage |
| source.processed | Processing complete | source_id, entity_count |
| source.failed | Processing failed | source_id, error |
| insight.new | New insight generated | insight preview |
| synthesis.started | Generation started | synthesis_id |
| synthesis.token | Streaming token | text |
| synthesis.complete | Generation complete | synthesis_id |

**Presence Events (Team Features):**
| Event | Description | Payload |
|-------|-------------|---------|
| user.joined | User viewing stream | user_id, user_name |
| user.left | User left stream | user_id |
| user.typing | User editing | user_id, element_id |

### 4.4 Server-Sent Events Alternative

For simpler streaming needs (synthesis generation):

**GET /v1/syntheses/:synthesis_id/stream**

Headers:
```
Accept: text/event-stream
Cache-Control: no-cache
```

Response Stream:
```
event: token
data: {"text":"The ","position":0}

event: token
data: {"text":"primary ","position":4}

event: citation
data: {"marker":"[1]","source_id":"source_abc","position":156}

event: progress
data: {"tokens_generated":250,"estimated_total":400}

event: complete
data: {"synthesis_id":"synth_abc","total_tokens":385}
```

---

## 5. Error Handling

### 5.1 Error Response Format

All errors follow a consistent structure:

```json
{
  "error": {
    "code": "validation_error",
    "message": "Request validation failed",
    "details": [
      {
        "field": "email",
        "code": "invalid_format",
        "message": "Invalid email format"
      }
    ],
    "request_id": "req_abc123"
  }
}
```

### 5.2 Error Codes

**HTTP Status Mapping:**

| Status | Error Code | Description |
|--------|------------|-------------|
| 400 | validation_error | Invalid request format or parameters |
| 400 | bad_request | General client error |
| 401 | unauthorized | Missing or invalid authentication |
| 401 | token_expired | JWT has expired |
| 403 | forbidden | Authenticated but not authorized |
| 404 | not_found | Resource doesn't exist |
| 409 | conflict | Resource state conflict |
| 422 | unprocessable_entity | Semantic validation failed |
| 429 | rate_limited | Too many requests |
| 500 | internal_error | Server error |
| 502 | upstream_error | Dependency failure |
| 503 | service_unavailable | Temporary unavailability |

### 5.3 Validation Errors

Field-level validation errors include path and constraint:

```json
{
  "error": {
    "code": "validation_error",
    "message": "Request validation failed",
    "details": [
      {
        "field": "name",
        "code": "required",
        "message": "Name is required"
      },
      {
        "field": "source_ids",
        "code": "min_length",
        "message": "At least one source is required",
        "constraint": {"min": 1}
      },
      {
        "field": "visibility",
        "code": "invalid_enum",
        "message": "Must be one of: private, team, public",
        "constraint": {"allowed": ["private", "team", "public"]}
      }
    ]
  }
}
```

### 5.4 Rate Limiting

Rate limit headers on all responses:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1699900000
```

When rate limited (429):
```json
{
  "error": {
    "code": "rate_limited",
    "message": "Rate limit exceeded",
    "details": {
      "limit": 1000,
      "window_seconds": 3600,
      "retry_after": 1800
    }
  }
}
```

**Rate Limits by Tier:**

| Tier | Requests/Hour | Concurrent Connections |
|------|---------------|------------------------|
| Free | 100 | 2 |
| Pro | 1000 | 10 |
| Team | 5000 | 50 |
| Enterprise | Custom | Custom |

---

## 6. Pagination and Filtering

### 6.1 Pagination Strategy

**Offset-Based Pagination (Default):**

Suitable for most list endpoints.

Request:
```
GET /v1/streams?page=2&per_page=20
```

Response meta:
```json
{
  "meta": {
    "page": 2,
    "per_page": 20,
    "total": 47,
    "total_pages": 3
  }
}
```

**Cursor-Based Pagination (For Large Sets):**

For feeds, activity logs, and real-time synced lists.

Request:
```
GET /v1/streams/:stream_id/activity?cursor=eyJpZCI6MTIzNH0=&limit=50
```

Response:
```json
{
  "data": [ /* items */ ],
  "meta": {
    "has_more": true,
    "next_cursor": "eyJpZCI6MTI4NH0=",
    "prev_cursor": "eyJpZCI6MTE4NH0="
  }
}
```

### 6.2 Sorting

Standard sort parameters:

```
GET /v1/sources?sort=created_at&order=desc
GET /v1/sources?sort=-created_at  # Alternative: prefix with minus for desc
```

Sortable fields vary by resource. Common patterns:
- `created_at` (default descending)
- `updated_at`
- `name` (alphabetical)
- `relevance` (for search results)

### 6.3 Filtering

**Simple Filters:**
```
GET /v1/sources?status=processed&type=paper
```

**Range Filters:**
```
GET /v1/sources?created_after=2024-01-01&created_before=2024-03-01
GET /v1/insights?confidence_min=0.7&confidence_max=1.0
```

**Array Filters:**
```
GET /v1/sources?type[]=paper&type[]=preprint
```

**Full-Text Search:**
```
GET /v1/sources?q=CRISPR+delivery
```

### 6.4 Field Selection

Request only needed fields:

```
GET /v1/sources?fields=id,title,authors,created_at
```

Response includes only requested fields:
```json
{
  "data": [
    {
      "id": "source_abc",
      "title": "CRISPR Review",
      "authors": [...],
      "created_at": "2024-03-15T10:00:00Z"
    }
  ]
}
```

### 6.5 Expansion/Embedding

Include related resources:

```
GET /v1/streams/stream_abc?expand=sources,insights
```

Response includes expanded relations:
```json
{
  "data": {
    "id": "stream_abc",
    "name": "CRISPR Research",
    "sources": [
      { /* source object */ }
    ],
    "insights": [
      { /* insight object */ }
    ]
  }
}
```

---

## 7. Background Jobs and Processing

### 7.1 Job Status Endpoint

**GET /v1/jobs/:job_id**

Response:
```json
{
  "data": {
    "id": "job_abc123",
    "type": "source_processing",
    "status": "processing",
    "progress": {
      "current_step": "entity_extraction",
      "steps_completed": 2,
      "total_steps": 4,
      "percent": 50
    },
    "result": null,
    "error": null,
    "created_at": "2024-03-15T10:30:00Z",
    "started_at": "2024-03-15T10:30:05Z",
    "completed_at": null
  }
}
```

**Job Statuses:**
| Status | Description |
|--------|-------------|
| pending | Queued, not yet started |
| processing | Currently running |
| completed | Finished successfully |
| failed | Finished with error |
| cancelled | Cancelled by user |

### 7.2 Job Types and Steps

**Source Processing:**
```
Steps:
1. fetch (for URLs) / upload (for files)
2. text_extraction
3. metadata_extraction
4. entity_extraction
5. relationship_analysis
```

**Synthesis Generation:**
```
Steps:
1. source_retrieval
2. context_construction
3. generation
4. citation_linking
```

**Insight Generation:**
```
Steps:
1. source_analysis
2. pattern_detection
3. insight_formulation
4. confidence_scoring
```

### 7.3 Webhook Notifications

Configure webhooks for job completion:

**POST /v1/webhooks**

Request:
```json
{
  "url": "https://yourapp.com/hooks/latticeforge",
  "events": ["source.processed", "insight.generated"],
  "secret": "whsec_your_secret"
}
```

Webhook payload:
```json
{
  "event": "source.processed",
  "data": {
    "job_id": "job_abc123",
    "source_id": "source_xyz",
    "stream_id": "stream_123",
    "entity_count": 42
  },
  "timestamp": "2024-03-15T10:32:00Z",
  "signature": "sha256=..."
}
```

---

## 8. File Upload and Export

### 8.1 File Upload

**Direct Upload (Small Files < 10MB):**

```http
POST /v1/streams/:stream_id/sources
Content-Type: multipart/form-data

file: <binary>
metadata: {"type": "paper"}
```

**Presigned Upload (Large Files):**

Step 1 - Get presigned URL:
```http
POST /v1/uploads/presign
Content-Type: application/json

{
  "filename": "large_document.pdf",
  "content_type": "application/pdf",
  "size": 52428800
}
```

Response:
```json
{
  "data": {
    "upload_id": "upload_abc123",
    "presigned_url": "https://storage.latticeforge.io/...",
    "expires_at": "2024-03-15T11:00:00Z",
    "fields": {
      "key": "uploads/upload_abc123/original.pdf",
      "policy": "...",
      "signature": "..."
    }
  }
}
```

Step 2 - Upload to presigned URL (direct to storage)

Step 3 - Confirm upload:
```http
POST /v1/uploads/upload_abc123/confirm
Content-Type: application/json

{
  "stream_id": "stream_xyz",
  "type": "paper"
}
```

### 8.2 Export

**Export Synthesis:**

```http
POST /v1/syntheses/:synthesis_id/export
Content-Type: application/json

{
  "format": "docx",
  "options": {
    "include_citations": true,
    "citation_style": "apa"
  }
}
```

Response:
```json
{
  "data": {
    "export_id": "export_abc",
    "status": "processing",
    "download_url": null
  }
}
```

Poll or subscribe for completion:
```json
{
  "data": {
    "export_id": "export_abc",
    "status": "completed",
    "download_url": "https://storage.latticeforge.io/exports/...",
    "expires_at": "2024-03-16T10:00:00Z"
  }
}
```

**Supported Export Formats:**

| Resource | Formats |
|----------|---------|
| Synthesis | markdown, docx, pdf, latex |
| Sources (list) | bibtex, ris, json |
| Graph | json, graphml, png, svg |
| Insights | markdown, json |

---

## 9. Versioning and Deprecation

### 9.1 Version Strategy

**URL Path Versioning:**
```
/v1/streams
/v2/streams
```

**Version Lifecycle:**
1. **Current**: Active, fully supported
2. **Deprecated**: Functional but discouraged, sunset date announced
3. **Sunset**: No longer available

### 9.2 Deprecation Process

1. Announce deprecation 6 months before sunset
2. Add `Deprecation` header to responses:
   ```http
   Deprecation: true
   Sunset: Sat, 01 Jun 2025 00:00:00 GMT
   Link: </v2/streams>; rel="successor-version"
   ```
3. Log usage to identify affected clients
4. Send email notifications to API key owners
5. Sunset old version

### 9.3 Breaking vs Non-Breaking Changes

**Non-Breaking (No Version Bump):**
- Adding new endpoints
- Adding optional request parameters
- Adding response fields
- Adding new enum values (if clients ignore unknown)
- Bug fixes

**Breaking (Requires Version Bump):**
- Removing endpoints
- Removing request/response fields
- Changing field types
- Changing endpoint semantics
- Changing authentication requirements

---

## 10. Performance and Caching

### 10.1 Response Caching

**Cache-Control Headers:**

Static resources:
```http
Cache-Control: public, max-age=86400
ETag: "abc123"
```

User-specific resources:
```http
Cache-Control: private, max-age=0, must-revalidate
ETag: "user-abc-v42"
```

**Conditional Requests:**

```http
GET /v1/sources/source_abc
If-None-Match: "etag-value"
```

Response (304 Not Modified): Empty body if unchanged

### 10.2 Compression

All responses support:
```http
Accept-Encoding: gzip, br
```

Large responses (>1KB) are compressed by default.

### 10.3 Performance Guidelines

**Batch Operations:**

Instead of N individual requests:
```http
POST /v1/sources/batch
Content-Type: application/json

{
  "operations": [
    {"method": "DELETE", "id": "source_1"},
    {"method": "DELETE", "id": "source_2"},
    {"method": "PATCH", "id": "source_3", "data": {"type": "preprint"}}
  ]
}
```

**Field Selection:**
Always specify `fields` parameter for lists to reduce payload size.

**Pagination Limits:**
- Default: 20 items
- Maximum: 100 items
- Use cursor pagination for large result sets

---

## 11. OpenAPI Specification

### 11.1 Specification Location

```
Production: https://api.latticeforge.io/v1/openapi.json
Docs:       https://docs.latticeforge.io/api
```

### 11.2 Example Specification Excerpt

```yaml
openapi: 3.1.0
info:
  title: LatticeForge API
  version: 1.0.0
  description: Research intelligence platform API

servers:
  - url: https://api.latticeforge.io/v1
    description: Production

paths:
  /streams:
    get:
      summary: List research streams
      operationId: listStreams
      tags: [Streams]
      security:
        - bearerAuth: []
      parameters:
        - name: page
          in: query
          schema:
            type: integer
            default: 1
        - name: per_page
          in: query
          schema:
            type: integer
            default: 20
            maximum: 100
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/StreamListResponse'
        '401':
          $ref: '#/components/responses/Unauthorized'

components:
  schemas:
    Stream:
      type: object
      required: [id, name, visibility, owner_id, created_at]
      properties:
        id:
          type: string
          example: stream_abc123
        name:
          type: string
          maxLength: 255
        description:
          type: string
          nullable: true
        visibility:
          type: string
          enum: [private, team, public]
        owner_id:
          type: string
        source_count:
          type: integer
        created_at:
          type: string
          format: date-time

  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT

  responses:
    Unauthorized:
      description: Authentication required
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
```

---

## 12. Testing and Development

### 12.1 Sandbox Environment

Development sandbox:
```
Base URL: https://api.sandbox.latticeforge.io/v1
```

Sandbox features:
- Test API keys work only here
- Data resets nightly
- Rate limits relaxed
- Mock AI responses available for deterministic testing

### 12.2 Test Fixtures

Request fixtures for common scenarios:

```http
POST /v1/sandbox/fixtures
Content-Type: application/json

{
  "fixture": "populated_stream",
  "options": {
    "source_count": 10,
    "insight_count": 5
  }
}
```

Response:
```json
{
  "data": {
    "stream_id": "stream_test_abc",
    "source_ids": ["source_1", ...],
    "insight_ids": ["insight_1", ...]
  }
}
```

### 12.3 SDK Support

Official SDKs:
- **TypeScript/JavaScript**: `@latticeforge/sdk`
- **Python**: `latticeforge`
- **Rust**: `latticeforge-sdk`

Example (TypeScript):
```typescript
import { LatticeForge } from '@latticeforge/sdk';

const client = new LatticeForge({
  apiKey: process.env.LATTICEFORGE_API_KEY
});

const streams = await client.streams.list({
  page: 1,
  perPage: 20
});

const source = await client.sources.create('stream_abc', {
  url: 'https://arxiv.org/abs/2301.00000'
});

// Real-time subscription
client.subscribe('stream:stream_abc', (event) => {
  console.log('Event:', event.type, event.data);
});
```

---

## 13. Implementation Checklist

### 13.1 Endpoint Implementation Priority

**Phase 1 - Core CRUD:**
- [ ] Authentication endpoints (login, refresh, logout)
- [ ] Stream CRUD operations
- [ ] Source CRUD operations
- [ ] Basic list pagination

**Phase 2 - Processing:**
- [ ] Source upload and processing
- [ ] Job status tracking
- [ ] WebSocket connection handling
- [ ] Basic real-time events

**Phase 3 - AI Features:**
- [ ] Synthesis generation with streaming
- [ ] Insight generation
- [ ] Entity extraction and search
- [ ] Graph queries

**Phase 4 - Advanced:**
- [ ] Export functionality
- [ ] Webhook delivery
- [ ] Batch operations
- [ ] Rate limiting

### 13.2 Quality Requirements

- [ ] All endpoints return consistent error format
- [ ] All list endpoints support pagination
- [ ] All mutations are idempotent where appropriate
- [ ] OpenAPI spec matches implementation
- [ ] Integration tests cover happy path and errors
- [ ] Response times meet SLA (<200ms for reads, <500ms for writes)
- [ ] Rate limiting applied and tested

---

*This API contract is the source of truth for frontend-backend integration. Any deviations should be documented and the spec updated. Breaking changes require coordination with frontend team and version bump.*
