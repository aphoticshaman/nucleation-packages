# LatticeForge ML/AI Engineer Integration Specification

## Document Purpose

This specification provides comprehensive guidance for ML/AI engineers working on LatticeForge. It covers model integration, inference architecture, streaming responses, output quality, prompt engineering, fine-tuning strategies, and evaluation frameworks. Use this as the authoritative reference for building reliable, high-quality AI features.

---

## 1. AI Architecture Overview

### 1.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LatticeForge AI Stack                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Application Layer                            │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │   Synthesis  │  │   Insights   │  │   Entities   │              │   │
│  │  │   Service    │  │   Service    │  │   Service    │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  ┌─────────────────────────────────┴───────────────────────────────────┐   │
│  │                        AI Orchestration Layer                       │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │   Prompt     │  │   Response   │  │   Quality    │              │   │
│  │  │   Manager    │  │   Handler    │  │   Monitor    │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │   Context    │  │   Token      │  │   Fallback   │              │   │
│  │  │   Builder    │  │   Streamer   │  │   Handler    │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  ┌─────────────────────────────────┴───────────────────────────────────┐   │
│  │                         Model Gateway                               │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │   Anthropic  │  │   OpenAI     │  │   Local      │              │   │
│  │  │   (Claude)   │  │   (GPT-4)    │  │   (vLLM)     │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Supporting Services                          │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │   Embedding  │  │   Reranking  │  │   NER        │              │   │
│  │  │   Service    │  │   Service    │  │   Service    │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Model Selection Strategy

| Task | Primary Model | Fallback | Rationale |
|------|---------------|----------|-----------|
| Synthesis | Claude 3 Opus | Claude 3 Sonnet | Quality critical, long context |
| Insights | Claude 3 Sonnet | GPT-4 Turbo | Balance of quality and cost |
| Entity Extraction | Fine-tuned NER | SpaCy | Speed, determinism |
| Embeddings | text-embedding-3-large | Cohere embed v3 | Quality for retrieval |
| Classification | Claude 3 Haiku | GPT-3.5 | Speed, low cost |

### 1.3 Cost and Latency Targets

| Operation | Latency Target | Cost Target | Volume |
|-----------|---------------|-------------|--------|
| Synthesis (streaming) | TTFT <1s, 50 tok/s | <$0.10/synthesis | 10K/day |
| Insight Generation | <5s total | <$0.02/insight | 50K/day |
| Entity Extraction | <200ms | <$0.001/doc | 100K/day |
| Embedding | <100ms | <$0.0001/doc | 200K/day |
| Re-ranking | <500ms | <$0.005/query | 20K/day |

---

## 2. Inference Infrastructure

### 2.1 Model Gateway Architecture

```rust
// src/ai/gateway.rs

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[async_trait]
pub trait ModelProvider: Send + Sync {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse>;
    async fn stream(&self, request: CompletionRequest) -> Result<TokenStream>;
    fn name(&self) -> &str;
    fn max_context_length(&self) -> usize;
}

pub struct ModelGateway {
    providers: HashMap<String, Box<dyn ModelProvider>>,
    router: ModelRouter,
    rate_limiter: RateLimiter,
    circuit_breaker: CircuitBreaker,
    metrics: MetricsCollector,
}

impl ModelGateway {
    pub async fn complete(&self, request: GatewayRequest) -> Result<GatewayResponse> {
        // 1. Route to appropriate provider
        let provider = self.router.route(&request)?;

        // 2. Check circuit breaker
        if self.circuit_breaker.is_open(provider.name()) {
            return self.handle_fallback(request).await;
        }

        // 3. Apply rate limiting
        self.rate_limiter.acquire(provider.name()).await?;

        // 4. Execute request with retry
        let result = self.execute_with_retry(provider, request.clone()).await;

        // 5. Update circuit breaker and metrics
        match &result {
            Ok(response) => {
                self.circuit_breaker.record_success(provider.name());
                self.metrics.record_completion(provider.name(), response);
            }
            Err(e) => {
                self.circuit_breaker.record_failure(provider.name());
                self.metrics.record_error(provider.name(), e);
            }
        }

        result
    }

    pub async fn stream(&self, request: GatewayRequest) -> Result<impl Stream<Item = Token>> {
        let provider = self.router.route(&request)?;

        // Create streaming response with backpressure
        let stream = provider.stream(request.into()).await?;

        // Wrap with metrics and error handling
        Ok(StreamWrapper::new(stream, provider.name(), &self.metrics))
    }
}
```

### 2.2 Token Streaming

```rust
// src/ai/streaming.rs

use futures::Stream;
use tokio::sync::mpsc;

pub struct TokenStreamer {
    buffer_size: usize,
    metrics: Arc<StreamMetrics>,
}

impl TokenStreamer {
    pub fn stream_synthesis(
        &self,
        provider: &dyn ModelProvider,
        request: SynthesisRequest,
    ) -> impl Stream<Item = SynthesisEvent> {
        let (tx, rx) = mpsc::channel(self.buffer_size);

        // Spawn streaming task
        tokio::spawn(async move {
            let start = Instant::now();
            let mut token_count = 0;
            let mut first_token_time = None;

            match provider.stream(request.into()).await {
                Ok(mut stream) => {
                    while let Some(token) = stream.next().await {
                        // Record first token latency
                        if first_token_time.is_none() {
                            first_token_time = Some(start.elapsed());
                            let _ = tx.send(SynthesisEvent::Started).await;
                        }

                        token_count += 1;

                        // Detect and emit citations
                        if let Some(citation) = detect_citation(&token) {
                            let _ = tx.send(SynthesisEvent::Citation(citation)).await;
                        }

                        let _ = tx.send(SynthesisEvent::Token(token)).await;
                    }

                    let _ = tx.send(SynthesisEvent::Complete {
                        token_count,
                        duration: start.elapsed(),
                        ttft: first_token_time.unwrap_or_default(),
                    }).await;
                }
                Err(e) => {
                    let _ = tx.send(SynthesisEvent::Error(e.into())).await;
                }
            }
        });

        ReceiverStream::new(rx)
    }
}

#[derive(Debug, Clone)]
pub enum SynthesisEvent {
    Started,
    Token(String),
    Citation(Citation),
    Progress { tokens: usize, estimated_total: usize },
    Complete { token_count: usize, duration: Duration, ttft: Duration },
    Error(AiError),
}
```

### 2.3 Request Batching

```rust
// src/ai/batching.rs

pub struct RequestBatcher {
    max_batch_size: usize,
    max_wait_time: Duration,
    pending: Mutex<Vec<PendingRequest>>,
}

impl RequestBatcher {
    pub async fn add(&self, request: EmbeddingRequest) -> oneshot::Receiver<EmbeddingResult> {
        let (tx, rx) = oneshot::channel();

        let mut pending = self.pending.lock().await;
        pending.push(PendingRequest { request, response: tx });

        // Trigger batch if full
        if pending.len() >= self.max_batch_size {
            self.flush_batch().await;
        }

        rx
    }

    async fn flush_batch(&self) {
        let mut pending = self.pending.lock().await;
        if pending.is_empty() {
            return;
        }

        let batch: Vec<_> = pending.drain(..).collect();
        drop(pending);

        // Execute batch request
        let requests: Vec<_> = batch.iter().map(|p| &p.request).collect();
        let results = self.execute_batch(requests).await;

        // Distribute results
        for (pending, result) in batch.into_iter().zip(results) {
            let _ = pending.response.send(result);
        }
    }

    pub fn start_batch_timer(&self) -> JoinHandle<()> {
        let batcher = Arc::clone(&self);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(batcher.max_wait_time).await;
                batcher.flush_batch().await;
            }
        })
    }
}
```

### 2.4 Caching Layer

```rust
// src/ai/cache.rs

pub struct InferenceCache {
    embedding_cache: Arc<DashMap<String, CachedEmbedding>>,
    completion_cache: Arc<DashMap<String, CachedCompletion>>,
    redis: RedisPool,
}

impl InferenceCache {
    pub async fn get_embedding(&self, text: &str) -> Option<Vec<f32>> {
        let key = self.embedding_key(text);

        // Check in-memory cache first
        if let Some(cached) = self.embedding_cache.get(&key) {
            if !cached.is_expired() {
                return Some(cached.value.clone());
            }
        }

        // Check Redis
        if let Ok(Some(cached)) = self.redis.get(&key).await {
            let embedding: CachedEmbedding = serde_json::from_str(&cached)?;
            self.embedding_cache.insert(key, embedding.clone());
            return Some(embedding.value);
        }

        None
    }

    pub async fn set_embedding(&self, text: &str, embedding: Vec<f32>) {
        let key = self.embedding_key(text);
        let cached = CachedEmbedding {
            value: embedding.clone(),
            created_at: Utc::now(),
            ttl: Duration::hours(24),
        };

        // Set in both caches
        self.embedding_cache.insert(key.clone(), cached.clone());
        let _ = self.redis.set_ex(&key, &serde_json::to_string(&cached).unwrap(), 86400).await;
    }

    fn embedding_key(&self, text: &str) -> String {
        format!("emb:{}:{}", EMBEDDING_MODEL_VERSION, hash(text))
    }
}
```

---

## 3. Prompt Engineering

### 3.1 Prompt Templates

```rust
// src/ai/prompts/synthesis.rs

pub struct SynthesisPrompt;

impl SynthesisPrompt {
    pub fn overview(sources: &[SourceContext], focus: Option<&str>) -> String {
        let mut prompt = String::new();

        prompt.push_str(r#"You are a research synthesis expert. Your task is to create a comprehensive overview that synthesizes the key findings, methodologies, and conclusions from the provided sources.

Guidelines:
- Integrate findings across sources, don't just summarize each one
- Use inline citations in the format [1], [2], etc.
- Highlight areas of agreement and disagreement between sources
- Identify gaps or areas needing further research
- Maintain academic tone and precision
- Be concise but thorough

"#);

        if let Some(focus) = focus {
            prompt.push_str(&format!("Focus area: {}\n\n", focus));
        }

        prompt.push_str("Sources:\n\n");

        for (i, source) in sources.iter().enumerate() {
            prompt.push_str(&format!(
                "[{}] {}\nAuthors: {}\nYear: {}\nAbstract: {}\n\nKey excerpts:\n{}\n\n---\n\n",
                i + 1,
                source.title,
                source.authors.join(", "),
                source.year,
                source.abstract_text,
                source.relevant_excerpts.join("\n\n")
            ));
        }

        prompt.push_str("Generate a synthesis of these sources:\n");

        prompt
    }

    pub fn comparison(sources: &[SourceContext], dimensions: &[&str]) -> String {
        let mut prompt = String::new();

        prompt.push_str(r#"You are a research synthesis expert. Your task is to create a comparative analysis of the provided sources.

Guidelines:
- Compare sources across the specified dimensions
- Use a structured format (tables where appropriate)
- Highlight key differences and similarities
- Use inline citations [1], [2], etc.
- Draw conclusions from the comparison

"#);

        prompt.push_str(&format!("Comparison dimensions: {}\n\n", dimensions.join(", ")));

        // Add sources...

        prompt
    }

    pub fn gap_analysis(sources: &[SourceContext], domain: &str) -> String {
        // Similar structure for gap analysis
        todo!()
    }
}
```

### 3.2 Prompt Management System

```rust
// src/ai/prompts/manager.rs

pub struct PromptManager {
    templates: HashMap<String, PromptTemplate>,
    ab_tests: HashMap<String, AbTest>,
    metrics: PromptMetrics,
}

#[derive(Clone)]
pub struct PromptTemplate {
    pub id: String,
    pub version: String,
    pub template: String,
    pub variables: Vec<String>,
    pub model_hints: ModelHints,
    pub evaluation_criteria: Vec<String>,
}

impl PromptManager {
    pub fn render(&self, template_id: &str, variables: &HashMap<String, String>) -> Result<RenderedPrompt> {
        let template = self.templates.get(template_id)
            .ok_or(PromptError::TemplateNotFound)?;

        // Check for A/B test
        let variant = self.get_ab_variant(template_id);

        // Render with handlebars
        let rendered = handlebars::Handlebars::new()
            .render_template(&variant.template, variables)?;

        Ok(RenderedPrompt {
            content: rendered,
            template_id: template_id.to_string(),
            version: variant.version.clone(),
            variant_id: variant.id.clone(),
        })
    }

    pub fn record_outcome(&self, prompt: &RenderedPrompt, outcome: PromptOutcome) {
        self.metrics.record(
            &prompt.template_id,
            &prompt.variant_id,
            outcome,
        );
    }
}
```

### 3.3 Context Construction

```rust
// src/ai/context.rs

pub struct ContextBuilder {
    max_tokens: usize,
    embedding_service: Arc<EmbeddingService>,
    reranker: Arc<RerankerService>,
}

impl ContextBuilder {
    pub async fn build_synthesis_context(
        &self,
        sources: &[Source],
        query: Option<&str>,
    ) -> Result<Vec<SourceContext>> {
        // 1. Extract relevant chunks from each source
        let mut all_chunks: Vec<ScoredChunk> = Vec::new();

        for source in sources {
            let chunks = self.chunk_source(source);

            // If query provided, rank chunks by relevance
            if let Some(query) = query {
                let embeddings = self.embedding_service.embed_batch(&chunks).await?;
                let query_embedding = self.embedding_service.embed(query).await?;

                for (chunk, embedding) in chunks.iter().zip(embeddings) {
                    let score = cosine_similarity(&query_embedding, &embedding);
                    all_chunks.push(ScoredChunk {
                        chunk: chunk.clone(),
                        source_id: source.id.clone(),
                        score,
                    });
                }
            } else {
                // Without query, use position-based scoring
                for (i, chunk) in chunks.iter().enumerate() {
                    let score = self.position_score(i, chunks.len());
                    all_chunks.push(ScoredChunk {
                        chunk: chunk.clone(),
                        source_id: source.id.clone(),
                        score,
                    });
                }
            }
        }

        // 2. Rerank if we have a query
        if query.is_some() {
            all_chunks = self.reranker.rerank(query.unwrap(), all_chunks).await?;
        }

        // 3. Select top chunks within token budget
        let mut selected = Vec::new();
        let mut token_count = 0;

        for chunk in all_chunks {
            let chunk_tokens = count_tokens(&chunk.chunk);
            if token_count + chunk_tokens > self.max_tokens {
                break;
            }
            token_count += chunk_tokens;
            selected.push(chunk);
        }

        // 4. Group by source and build context objects
        self.build_source_contexts(sources, selected)
    }

    fn chunk_source(&self, source: &Source) -> Vec<String> {
        // Semantic chunking that respects section boundaries
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut current_tokens = 0;

        for section in &source.sections {
            let section_tokens = count_tokens(&section.content);

            if current_tokens + section_tokens > CHUNK_SIZE {
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk);
                    current_chunk = String::new();
                    current_tokens = 0;
                }

                // If section is too large, split it
                if section_tokens > CHUNK_SIZE {
                    let sub_chunks = self.split_section(&section.content);
                    chunks.extend(sub_chunks);
                } else {
                    current_chunk = section.content.clone();
                    current_tokens = section_tokens;
                }
            } else {
                current_chunk.push_str(&section.content);
                current_chunk.push_str("\n\n");
                current_tokens += section_tokens;
            }
        }

        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }

        chunks
    }
}
```

---

## 4. Feature Implementations

### 4.1 Synthesis Generation

```rust
// src/ai/features/synthesis.rs

pub struct SynthesisGenerator {
    gateway: Arc<ModelGateway>,
    context_builder: Arc<ContextBuilder>,
    citation_tracker: CitationTracker,
    quality_monitor: QualityMonitor,
}

impl SynthesisGenerator {
    pub async fn generate(
        &self,
        request: SynthesisRequest,
    ) -> Result<impl Stream<Item = SynthesisEvent>> {
        // 1. Build context
        let context = self.context_builder.build_synthesis_context(
            &request.sources,
            request.focus.as_deref(),
        ).await?;

        // 2. Construct prompt
        let prompt = match request.synthesis_type {
            SynthesisType::Overview => SynthesisPrompt::overview(&context, request.focus.as_deref()),
            SynthesisType::Comparison => SynthesisPrompt::comparison(&context, &request.dimensions),
            SynthesisType::GapAnalysis => SynthesisPrompt::gap_analysis(&context, &request.domain),
        };

        // 3. Create gateway request
        let gateway_request = GatewayRequest {
            prompt,
            model: ModelSelection::PreferClaude3Opus,
            max_tokens: request.max_tokens.unwrap_or(4096),
            temperature: 0.3,
            stream: true,
        };

        // 4. Stream response with citation tracking
        let stream = self.gateway.stream(gateway_request).await?;

        Ok(self.process_stream(stream, context))
    }

    fn process_stream(
        &self,
        stream: impl Stream<Item = Token>,
        context: Vec<SourceContext>,
    ) -> impl Stream<Item = SynthesisEvent> {
        let citation_tracker = self.citation_tracker.clone();
        let source_map: HashMap<_, _> = context.iter()
            .enumerate()
            .map(|(i, s)| (format!("[{}]", i + 1), s.source_id.clone()))
            .collect();

        stream.map(move |token| {
            // Detect citations in token
            if let Some(citation) = detect_citation(&token.text) {
                if let Some(source_id) = source_map.get(&citation) {
                    return SynthesisEvent::Citation(Citation {
                        marker: citation,
                        source_id: source_id.clone(),
                    });
                }
            }

            SynthesisEvent::Token(token.text)
        })
    }
}
```

### 4.2 Insight Generation

```rust
// src/ai/features/insights.rs

pub struct InsightGenerator {
    gateway: Arc<ModelGateway>,
    entity_service: Arc<EntityService>,
    embedding_service: Arc<EmbeddingService>,
}

impl InsightGenerator {
    pub async fn generate_cross_source_insights(
        &self,
        sources: &[Source],
    ) -> Result<Vec<Insight>> {
        // 1. Extract entities and their contexts
        let entity_contexts = self.entity_service.get_entity_contexts(sources).await?;

        // 2. Find entity pairs that appear in multiple sources
        let cross_source_entities = self.find_cross_source_entities(&entity_contexts);

        // 3. For each promising pair, generate insight
        let mut insights = Vec::new();

        for (entity_a, entity_b) in cross_source_entities.iter().take(10) {
            let insight = self.generate_connection_insight(
                entity_a,
                entity_b,
                &entity_contexts,
            ).await?;

            if let Some(insight) = insight {
                insights.push(insight);
            }
        }

        // 4. Rank insights by novelty and confidence
        self.rank_insights(&mut insights);

        Ok(insights)
    }

    async fn generate_connection_insight(
        &self,
        entity_a: &Entity,
        entity_b: &Entity,
        contexts: &HashMap<String, Vec<EntityContext>>,
    ) -> Result<Option<Insight>> {
        // Gather contexts for both entities
        let contexts_a = contexts.get(&entity_a.id).cloned().unwrap_or_default();
        let contexts_b = contexts.get(&entity_b.id).cloned().unwrap_or_default();

        let prompt = format!(r#"Analyze the relationship between "{}" and "{}" based on the following contexts.

Contexts for "{}":
{}

Contexts for "{}":
{}

If there is a meaningful, non-obvious connection, describe it in 1-2 sentences.
Then rate your confidence (0.0-1.0) and explain your reasoning.

If there is no meaningful connection, respond with "NO_CONNECTION".

Format:
INSIGHT: [your insight]
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation]
"#,
            entity_a.name, entity_b.name,
            entity_a.name, self.format_contexts(&contexts_a),
            entity_b.name, self.format_contexts(&contexts_b)
        );

        let response = self.gateway.complete(GatewayRequest {
            prompt,
            model: ModelSelection::PreferClaude3Sonnet,
            max_tokens: 500,
            temperature: 0.2,
            stream: false,
        }).await?;

        self.parse_insight_response(&response.text, entity_a, entity_b, &contexts_a, &contexts_b)
    }

    fn rank_insights(&self, insights: &mut Vec<Insight>) {
        insights.sort_by(|a, b| {
            // Score = confidence * novelty * source_diversity
            let score_a = a.confidence * a.novelty_score * (a.source_count as f32).sqrt();
            let score_b = b.confidence * b.novelty_score * (b.source_count as f32).sqrt();
            score_b.partial_cmp(&score_a).unwrap()
        });
    }
}
```

### 4.3 Entity Extraction

```rust
// src/ai/features/entities.rs

pub struct EntityExtractor {
    ner_model: Arc<NerModel>,  // Fine-tuned or SpaCy
    llm_classifier: Arc<ModelGateway>,
    entity_linker: Arc<EntityLinker>,
}

impl EntityExtractor {
    pub async fn extract(&self, document: &Document) -> Result<Vec<ExtractedEntity>> {
        // 1. Run NER model for initial extraction
        let ner_entities = self.ner_model.predict(&document.text).await?;

        // 2. Classify entity types with LLM for ambiguous cases
        let classified = self.classify_ambiguous_entities(ner_entities).await?;

        // 3. Link to canonical entities
        let linked = self.entity_linker.link(classified).await?;

        // 4. Extract relationships between entities
        let with_relationships = self.extract_relationships(&linked, &document.text).await?;

        Ok(with_relationships)
    }

    async fn extract_relationships(
        &self,
        entities: &[ExtractedEntity],
        text: &str,
    ) -> Result<Vec<ExtractedEntity>> {
        // For each pair of entities in proximity, check for relationship
        let mut relationships = Vec::new();

        for (i, entity_a) in entities.iter().enumerate() {
            for entity_b in entities.iter().skip(i + 1) {
                // Check if entities are within relationship window
                if !self.entities_in_proximity(entity_a, entity_b, text) {
                    continue;
                }

                // Extract relationship type
                let context = self.get_relationship_context(entity_a, entity_b, text);
                let relationship = self.classify_relationship(&context, entity_a, entity_b).await?;

                if let Some(rel) = relationship {
                    relationships.push(rel);
                }
            }
        }

        // Add relationships to entities
        // ...

        Ok(entities.to_vec())
    }

    async fn classify_relationship(
        &self,
        context: &str,
        entity_a: &ExtractedEntity,
        entity_b: &ExtractedEntity,
    ) -> Result<Option<Relationship>> {
        let prompt = format!(r#"Given the context, classify the relationship between "{}" and "{}".

Context: {}

Possible relationships:
- CITES: A cites/references B
- CONTRADICTS: A contradicts B
- EXTENDS: A builds upon B
- COLLABORATES: A and B work together
- USES: A uses/applies B
- CAUSES: A causes/leads to B
- NONE: No clear relationship

Respond with just the relationship type, or NONE.
"#,
            entity_a.name, entity_b.name, context
        );

        let response = self.llm_classifier.complete(GatewayRequest {
            prompt,
            model: ModelSelection::PreferFast,
            max_tokens: 20,
            temperature: 0.0,
            stream: false,
        }).await?;

        self.parse_relationship(&response.text)
    }
}
```

---

## 5. Quality and Evaluation

### 5.1 Output Quality Metrics

```rust
// src/ai/quality/metrics.rs

pub struct QualityMetrics {
    pub factual_accuracy: f32,    // 0-1: Verified against sources
    pub citation_coverage: f32,   // 0-1: Claims have citations
    pub coherence: f32,           // 0-1: Logical flow
    pub completeness: f32,        // 0-1: Covers key topics
    pub relevance: f32,           // 0-1: Focused on query
}

pub struct QualityEvaluator {
    reference_checker: Arc<ReferenceChecker>,
    coherence_model: Arc<CoherenceModel>,
}

impl QualityEvaluator {
    pub async fn evaluate_synthesis(
        &self,
        synthesis: &Synthesis,
        sources: &[Source],
    ) -> Result<QualityMetrics> {
        // 1. Check citation coverage
        let citations = extract_citations(&synthesis.content);
        let claims = extract_claims(&synthesis.content);
        let citation_coverage = citations.len() as f32 / claims.len().max(1) as f32;

        // 2. Verify factual accuracy (sample claims)
        let accuracy_samples: Vec<_> = claims.iter().take(10).collect();
        let accuracy_results = self.reference_checker.verify_claims(&accuracy_samples, sources).await?;
        let factual_accuracy = accuracy_results.iter().filter(|r| r.verified).count() as f32
            / accuracy_results.len() as f32;

        // 3. Assess coherence
        let coherence = self.coherence_model.score(&synthesis.content).await?;

        // 4. Check completeness against source topics
        let source_topics = extract_topics(sources);
        let synthesis_topics = extract_topics_from_text(&synthesis.content);
        let completeness = topic_coverage(&synthesis_topics, &source_topics);

        // 5. Relevance to original query (if provided)
        let relevance = if let Some(query) = &synthesis.query {
            self.compute_relevance(query, &synthesis.content).await?
        } else {
            1.0
        };

        Ok(QualityMetrics {
            factual_accuracy,
            citation_coverage,
            coherence,
            completeness,
            relevance,
        })
    }
}
```

### 5.2 A/B Testing Framework

```rust
// src/ai/quality/ab_testing.rs

pub struct AbTestManager {
    tests: RwLock<HashMap<String, AbTest>>,
    metrics: Arc<MetricsStore>,
}

#[derive(Clone)]
pub struct AbTest {
    pub id: String,
    pub name: String,
    pub variants: Vec<Variant>,
    pub allocation: AllocationStrategy,
    pub metrics: Vec<String>,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub status: TestStatus,
}

impl AbTestManager {
    pub fn assign_variant(&self, test_id: &str, user_id: &str) -> Option<&Variant> {
        let tests = self.tests.read().unwrap();
        let test = tests.get(test_id)?;

        if test.status != TestStatus::Running {
            return None;
        }

        // Consistent hashing for user assignment
        let hash = consistent_hash(user_id, test_id);
        let bucket = hash % 100;

        let mut cumulative = 0;
        for variant in &test.variants {
            cumulative += variant.allocation_percent;
            if bucket < cumulative {
                return Some(variant);
            }
        }

        None
    }

    pub fn record_outcome(&self, test_id: &str, variant_id: &str, outcome: Outcome) {
        self.metrics.record(MetricEvent {
            test_id: test_id.to_string(),
            variant_id: variant_id.to_string(),
            outcome,
            timestamp: Utc::now(),
        });
    }

    pub async fn analyze_test(&self, test_id: &str) -> Result<TestAnalysis> {
        let test = self.tests.read().unwrap().get(test_id).cloned()
            .ok_or(AbTestError::NotFound)?;

        let mut variant_stats = Vec::new();

        for variant in &test.variants {
            let outcomes = self.metrics.get_outcomes(test_id, &variant.id).await?;

            let stats = VariantStats {
                variant_id: variant.id.clone(),
                sample_size: outcomes.len(),
                conversion_rate: calculate_conversion_rate(&outcomes),
                mean_quality: calculate_mean_quality(&outcomes),
                confidence_interval: calculate_ci(&outcomes),
            };

            variant_stats.push(stats);
        }

        // Calculate statistical significance
        let significance = calculate_significance(&variant_stats);

        Ok(TestAnalysis {
            test_id: test_id.to_string(),
            variant_stats,
            is_significant: significance < 0.05,
            p_value: significance,
            recommendation: self.get_recommendation(&variant_stats, significance),
        })
    }
}
```

### 5.3 Evaluation Datasets

```rust
// src/ai/quality/evaluation.rs

pub struct EvaluationSuite {
    datasets: HashMap<String, EvaluationDataset>,
}

#[derive(Clone)]
pub struct EvaluationDataset {
    pub name: String,
    pub task_type: TaskType,
    pub examples: Vec<EvaluationExample>,
}

#[derive(Clone)]
pub struct EvaluationExample {
    pub id: String,
    pub input: serde_json::Value,
    pub expected_output: serde_json::Value,
    pub evaluation_criteria: Vec<EvaluationCriterion>,
}

impl EvaluationSuite {
    pub async fn run_evaluation(
        &self,
        model_config: &ModelConfig,
        dataset_name: &str,
    ) -> Result<EvaluationReport> {
        let dataset = self.datasets.get(dataset_name)
            .ok_or(EvaluationError::DatasetNotFound)?;

        let mut results = Vec::new();

        for example in &dataset.examples {
            let output = self.run_inference(model_config, &example.input).await?;
            let scores = self.evaluate_output(&output, example).await?;

            results.push(ExampleResult {
                example_id: example.id.clone(),
                output,
                scores,
            });
        }

        Ok(EvaluationReport {
            model_config: model_config.clone(),
            dataset_name: dataset_name.to_string(),
            results,
            aggregate_scores: self.aggregate_scores(&results),
            timestamp: Utc::now(),
        })
    }

    async fn evaluate_output(
        &self,
        output: &serde_json::Value,
        example: &EvaluationExample,
    ) -> Result<HashMap<String, f32>> {
        let mut scores = HashMap::new();

        for criterion in &example.evaluation_criteria {
            let score = match criterion {
                EvaluationCriterion::ExactMatch { field } => {
                    exact_match(output.get(field), example.expected_output.get(field))
                }
                EvaluationCriterion::SemanticSimilarity { field, threshold } => {
                    self.semantic_similarity(
                        output.get(field),
                        example.expected_output.get(field),
                        *threshold,
                    ).await?
                }
                EvaluationCriterion::LlmJudge { prompt_template } => {
                    self.llm_judge(output, &example.expected_output, prompt_template).await?
                }
                EvaluationCriterion::Custom { evaluator } => {
                    evaluator(output, &example.expected_output)?
                }
            };

            scores.insert(criterion.name().to_string(), score);
        }

        Ok(scores)
    }
}
```

---

## 6. Fine-Tuning and Optimization

### 6.1 Fine-Tuning Pipeline

```rust
// src/ai/training/finetuning.rs

pub struct FineTuningPipeline {
    data_processor: DataProcessor,
    trainer: ModelTrainer,
    evaluator: EvaluationSuite,
}

impl FineTuningPipeline {
    pub async fn prepare_dataset(
        &self,
        raw_data: &[RawTrainingExample],
        config: &DatasetConfig,
    ) -> Result<PreparedDataset> {
        // 1. Filter and clean data
        let filtered: Vec<_> = raw_data.iter()
            .filter(|ex| self.passes_quality_checks(ex))
            .collect();

        // 2. Format for target model
        let formatted: Vec<_> = filtered.iter()
            .map(|ex| self.format_example(ex, &config.format))
            .collect::<Result<Vec<_>>>()?;

        // 3. Split into train/val/test
        let (train, val, test) = self.split_data(&formatted, config.split_ratios);

        // 4. Apply augmentation if configured
        let train_augmented = if config.augment {
            self.augment_data(train).await?
        } else {
            train
        };

        Ok(PreparedDataset {
            train: train_augmented,
            validation: val,
            test,
            config: config.clone(),
        })
    }

    pub async fn fine_tune(
        &self,
        base_model: &str,
        dataset: PreparedDataset,
        config: &TrainingConfig,
    ) -> Result<FineTunedModel> {
        // 1. Start training job
        let job = self.trainer.start_training(
            base_model,
            &dataset.train,
            &dataset.validation,
            config,
        ).await?;

        // 2. Monitor training
        while !job.is_complete() {
            let status = job.get_status().await?;
            log::info!("Training progress: {:?}", status);
            tokio::time::sleep(Duration::from_secs(60)).await;
        }

        // 3. Evaluate on test set
        let model = job.get_model().await?;
        let eval_results = self.evaluator.run_evaluation(
            &model.into(),
            "fine_tuning_test",
        ).await?;

        // 4. Compare to baseline
        let baseline_results = self.evaluator.run_evaluation(
            &ModelConfig::from_id(base_model),
            "fine_tuning_test",
        ).await?;

        let improvement = calculate_improvement(&baseline_results, &eval_results);

        Ok(FineTunedModel {
            model_id: model.id,
            base_model: base_model.to_string(),
            evaluation_results: eval_results,
            improvement_over_baseline: improvement,
            training_config: config.clone(),
        })
    }
}
```

### 6.2 Prompt Optimization

```rust
// src/ai/training/prompt_optimization.rs

pub struct PromptOptimizer {
    gateway: Arc<ModelGateway>,
    evaluator: Arc<EvaluationSuite>,
}

impl PromptOptimizer {
    pub async fn optimize(
        &self,
        base_prompt: &str,
        examples: &[OptimizationExample],
        config: &OptimizationConfig,
    ) -> Result<OptimizedPrompt> {
        let mut current_prompt = base_prompt.to_string();
        let mut best_score = 0.0;
        let mut best_prompt = current_prompt.clone();
        let mut history = Vec::new();

        for iteration in 0..config.max_iterations {
            // 1. Evaluate current prompt
            let scores = self.evaluate_prompt(&current_prompt, examples).await?;
            let avg_score = scores.values().sum::<f32>() / scores.len() as f32;

            history.push(OptimizationStep {
                iteration,
                prompt: current_prompt.clone(),
                score: avg_score,
            });

            if avg_score > best_score {
                best_score = avg_score;
                best_prompt = current_prompt.clone();
            }

            // 2. Check convergence
            if self.has_converged(&history, config.convergence_threshold) {
                break;
            }

            // 3. Generate prompt variations
            let variations = self.generate_variations(&current_prompt, &scores, examples).await?;

            // 4. Evaluate variations and select best
            let mut best_variation_score = 0.0;
            let mut best_variation = current_prompt.clone();

            for variation in variations {
                let var_scores = self.evaluate_prompt(&variation, examples).await?;
                let var_avg = var_scores.values().sum::<f32>() / var_scores.len() as f32;

                if var_avg > best_variation_score {
                    best_variation_score = var_avg;
                    best_variation = variation;
                }
            }

            current_prompt = best_variation;
        }

        Ok(OptimizedPrompt {
            original: base_prompt.to_string(),
            optimized: best_prompt,
            improvement: best_score - self.evaluate_prompt(base_prompt, examples).await?.values().sum::<f32>() / examples.len() as f32,
            history,
        })
    }

    async fn generate_variations(
        &self,
        prompt: &str,
        scores: &HashMap<String, f32>,
        examples: &[OptimizationExample],
    ) -> Result<Vec<String>> {
        // Find lowest-scoring examples
        let weak_examples: Vec<_> = examples.iter()
            .filter(|ex| scores.get(&ex.id).unwrap_or(&0.0) < &0.8)
            .take(3)
            .collect();

        let meta_prompt = format!(r#"You are a prompt engineering expert. Improve this prompt to better handle the failure cases.

Current prompt:
{}

Failure cases:
{}

Generate 5 improved variations of the prompt. Each should:
1. Address the specific failure cases
2. Maintain the original intent
3. Be clear and specific

Output each variation on a new line, prefixed with "VARIATION:"
"#,
            prompt,
            self.format_failure_cases(&weak_examples)
        );

        let response = self.gateway.complete(GatewayRequest {
            prompt: meta_prompt,
            model: ModelSelection::PreferClaude3Opus,
            max_tokens: 2000,
            temperature: 0.7,
            stream: false,
        }).await?;

        self.parse_variations(&response.text)
    }
}
```

---

## 7. Monitoring and Observability

### 7.1 Inference Metrics

```rust
// src/ai/monitoring/metrics.rs

pub struct AiMetrics {
    request_count: IntCounter,
    request_duration: Histogram,
    token_count: IntCounter,
    error_count: IntCounter,
    cost_total: Counter,
    quality_score: Histogram,

    // Per-model metrics
    model_request_count: IntCounterVec,
    model_latency: HistogramVec,
    model_ttft: HistogramVec,  // Time to first token

    // Business metrics
    synthesis_count: IntCounter,
    insight_count: IntCounter,
    entity_count: IntCounter,
}

impl AiMetrics {
    pub fn record_completion(&self, event: CompletionEvent) {
        self.request_count.inc();
        self.request_duration.observe(event.duration.as_secs_f64());
        self.token_count.inc_by(event.token_count as u64);
        self.cost_total.inc_by(event.cost);

        self.model_request_count
            .with_label_values(&[&event.model])
            .inc();
        self.model_latency
            .with_label_values(&[&event.model])
            .observe(event.duration.as_secs_f64());

        if let Some(ttft) = event.time_to_first_token {
            self.model_ttft
                .with_label_values(&[&event.model])
                .observe(ttft.as_secs_f64());
        }

        if let Some(quality) = event.quality_score {
            self.quality_score.observe(quality as f64);
        }
    }

    pub fn record_error(&self, model: &str, error: &AiError) {
        self.error_count.inc();
        // Log error details
    }
}
```

### 7.2 Quality Monitoring Dashboard

```sql
-- Grafana dashboard queries

-- Synthesis quality over time
SELECT
    time_bucket('1 hour', created_at) AS time,
    AVG(quality_score) AS avg_quality,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY quality_score) AS median_quality,
    COUNT(*) AS count
FROM synthesis_metrics
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY time_bucket('1 hour', created_at)
ORDER BY time;

-- Model performance comparison
SELECT
    model_id,
    AVG(latency_ms) AS avg_latency,
    AVG(ttft_ms) AS avg_ttft,
    SUM(token_count) AS total_tokens,
    SUM(cost_usd) AS total_cost,
    AVG(quality_score) AS avg_quality
FROM inference_metrics
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY model_id;

-- Error rate by type
SELECT
    error_type,
    COUNT(*) AS count,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () AS percentage
FROM ai_errors
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY error_type
ORDER BY count DESC;
```

### 7.3 Alerting Rules

```yaml
# prometheus/alerts/ai.yaml
groups:
  - name: ai_alerts
    rules:
      - alert: HighAiErrorRate
        expr: |
          sum(rate(ai_errors_total[5m])) /
          sum(rate(ai_requests_total[5m])) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High AI error rate"
          description: "AI error rate is {{ $value | humanizePercentage }}"

      - alert: HighAiLatency
        expr: |
          histogram_quantile(0.99, sum(rate(ai_request_duration_seconds_bucket[5m])) by (le, model))
          > 30
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High AI latency for {{ $labels.model }}"
          description: "P99 latency is {{ $value }}s"

      - alert: LowSynthesisQuality
        expr: |
          avg(ai_synthesis_quality_score) < 0.7
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Synthesis quality degraded"
          description: "Average quality score is {{ $value }}"

      - alert: HighAiCost
        expr: |
          sum(increase(ai_cost_usd_total[1h])) > 100
        for: 0m
        labels:
          severity: warning
        annotations:
          summary: "High AI spending"
          description: "Spent ${{ $value }} in the last hour"

      - alert: ModelProviderDown
        expr: |
          up{job="ai_gateway"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "AI model provider unreachable"
```

---

## 8. Safety and Guardrails

### 8.1 Output Validation

```rust
// src/ai/safety/validation.rs

pub struct OutputValidator {
    content_filter: Arc<ContentFilter>,
    fact_checker: Arc<FactChecker>,
    pii_detector: Arc<PiiDetector>,
}

impl OutputValidator {
    pub async fn validate(&self, output: &str, context: &ValidationContext) -> Result<ValidationResult> {
        let mut issues = Vec::new();

        // 1. Content safety check
        let safety_result = self.content_filter.check(output).await?;
        if !safety_result.is_safe {
            issues.push(ValidationIssue {
                severity: IssueSeverity::Critical,
                category: "content_safety",
                description: safety_result.reason,
            });
        }

        // 2. PII detection
        let pii_result = self.pii_detector.detect(output).await?;
        if !pii_result.detected_pii.is_empty() {
            issues.push(ValidationIssue {
                severity: IssueSeverity::Warning,
                category: "pii_leak",
                description: format!("Detected PII types: {:?}", pii_result.detected_pii),
            });
        }

        // 3. Hallucination check (if sources provided)
        if let Some(sources) = &context.sources {
            let claims = extract_claims(output);
            for claim in claims {
                let verification = self.fact_checker.verify(&claim, sources).await?;
                if !verification.supported {
                    issues.push(ValidationIssue {
                        severity: IssueSeverity::Warning,
                        category: "potential_hallucination",
                        description: format!("Unverified claim: {}", claim),
                    });
                }
            }
        }

        // 4. Format validation
        if let Some(expected_format) = &context.expected_format {
            if !matches_format(output, expected_format) {
                issues.push(ValidationIssue {
                    severity: IssueSeverity::Info,
                    category: "format_mismatch",
                    description: "Output doesn't match expected format",
                });
            }
        }

        let passed = !issues.iter().any(|i| i.severity == IssueSeverity::Critical);

        Ok(ValidationResult {
            passed,
            issues,
            sanitized_output: if passed { Some(output.to_string()) } else { None },
        })
    }
}
```

### 8.2 Rate Limiting and Cost Control

```rust
// src/ai/safety/cost_control.rs

pub struct CostController {
    budgets: HashMap<String, Budget>,
    usage: Arc<RwLock<HashMap<String, UsageTracker>>>,
}

#[derive(Clone)]
pub struct Budget {
    pub daily_limit_usd: f64,
    pub monthly_limit_usd: f64,
    pub per_request_limit_usd: f64,
    pub alert_threshold: f64,  // Percentage at which to alert
}

impl CostController {
    pub async fn check_budget(&self, user_id: &str, estimated_cost: f64) -> Result<BudgetCheck> {
        let usage = self.usage.read().await;
        let tracker = usage.get(user_id).ok_or(CostError::NoTracker)?;
        let budget = self.budgets.get(&tracker.tier).ok_or(CostError::NoBudget)?;

        // Check per-request limit
        if estimated_cost > budget.per_request_limit_usd {
            return Ok(BudgetCheck::Denied {
                reason: "Request exceeds per-request limit",
            });
        }

        // Check daily limit
        if tracker.daily_spend + estimated_cost > budget.daily_limit_usd {
            return Ok(BudgetCheck::Denied {
                reason: "Daily budget exceeded",
            });
        }

        // Check monthly limit
        if tracker.monthly_spend + estimated_cost > budget.monthly_limit_usd {
            return Ok(BudgetCheck::Denied {
                reason: "Monthly budget exceeded",
            });
        }

        // Check if near threshold
        let daily_percentage = (tracker.daily_spend + estimated_cost) / budget.daily_limit_usd;
        if daily_percentage > budget.alert_threshold {
            return Ok(BudgetCheck::AllowedWithWarning {
                warning: format!("{}% of daily budget used", (daily_percentage * 100.0) as u32),
            });
        }

        Ok(BudgetCheck::Allowed)
    }

    pub async fn record_usage(&self, user_id: &str, actual_cost: f64) {
        let mut usage = self.usage.write().await;
        if let Some(tracker) = usage.get_mut(user_id) {
            tracker.daily_spend += actual_cost;
            tracker.monthly_spend += actual_cost;
            tracker.total_spend += actual_cost;
        }
    }
}
```

---

## 9. Development Guidelines

### 9.1 Prompt Development Workflow

1. **Design**: Define task, inputs, expected outputs
2. **Prototype**: Test with handful of examples manually
3. **Evaluate**: Run against evaluation dataset
4. **Optimize**: Use prompt optimization or manual iteration
5. **A/B Test**: Compare against existing prompts in production
6. **Monitor**: Track quality metrics post-deployment

### 9.2 Model Integration Checklist

- [ ] Define fallback behavior for model failures
- [ ] Implement retry logic with exponential backoff
- [ ] Set appropriate timeouts
- [ ] Add comprehensive logging
- [ ] Instrument with metrics
- [ ] Validate outputs before returning to user
- [ ] Implement cost controls
- [ ] Document expected latency and costs
- [ ] Create evaluation dataset
- [ ] Set up quality monitoring alerts

### 9.3 Code Review Guidelines

**For AI feature PRs:**
- [ ] Prompts are in version-controlled templates
- [ ] Error handling covers all failure modes
- [ ] Output validation is implemented
- [ ] Metrics are instrumented
- [ ] Cost estimation is included
- [ ] Tests include quality evaluation
- [ ] Documentation explains AI behavior

---

*AI capabilities are central to LatticeForge's value proposition. This document should evolve as models improve, new techniques emerge, and we learn from production usage. Review and update quarterly.*
