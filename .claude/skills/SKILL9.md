# RAG_SYSTEMS.skill.md

## Retrieval-Augmented Generation: Architecture, Optimization, and Production Deployment

**Version**: 1.0
**Domain**: Information Retrieval, NLP, LLM Applications
**Prerequisites**: LLM basics, vector databases, embeddings
**Complexity**: Intermediate to Advanced

---

## 1. EXECUTIVE SUMMARY

RAG grounds LLM responses in retrieved evidence, reducing hallucination and enabling knowledge updates without retraining. This skill covers embedding models, chunking strategies, retrieval optimization, reranking, and production patterns.

**Core Insight**: RAG quality depends more on retrieval quality than generation quality. Garbage in, garbage out.

---

## 2. RAG FUNDAMENTALS

### 2.1 The RAG Pipeline

```
Query → Retrieve Relevant Documents → Augment Prompt → Generate Response
```

**Detailed Flow**:
1. **Query Processing**: Parse, expand, embed user query
2. **Retrieval**: Find relevant chunks from corpus
3. **Reranking**: Re-score and filter results
4. **Context Assembly**: Format retrieved content for LLM
5. **Generation**: LLM produces answer grounded in context
6. **Citation**: Attribute claims to sources

### 2.2 Why RAG?

**vs. Fine-tuning**:
- No retraining needed for knowledge updates
- Explicit source attribution
- Works with any LLM
- Handles long-tail knowledge better

**vs. Long Context**:
- More focused retrieval
- Lower cost (only relevant context)
- Better for large corpora

### 2.3 RAG Components

| Component | Options |
|-----------|---------|
| Embedding | OpenAI, Cohere, BGE, E5, Sentence-BERT |
| Vector DB | Pinecone, Weaviate, Qdrant, Chroma, pgvector |
| Chunking | Fixed, semantic, recursive, agentic |
| Reranking | Cross-encoder, Cohere Rerank, ColBERT |
| Generation | GPT-4, Claude, Llama, Mistral |

---

## 3. EMBEDDING MODELS

### 3.1 Types of Embeddings

**Dense Embeddings**: Single vector per text
- Captures semantic meaning
- 384-4096 dimensions typical
- Good for semantic similarity

**Sparse Embeddings**: Bag-of-words style
- BM25, TF-IDF, SPLADE
- Better for keyword matching
- Interpretable

**Hybrid**: Combine dense + sparse
- Best of both worlds
- Weighted combination

### 3.2 Popular Models

| Model | Dimensions | Context | Notes |
|-------|------------|---------|-------|
| text-embedding-3-small | 1536 | 8191 | OpenAI, good default |
| text-embedding-3-large | 3072 | 8191 | OpenAI, higher quality |
| Cohere embed-v3 | 1024 | 512 | Good multilingual |
| BGE-large-en-v1.5 | 1024 | 512 | Open source, high quality |
| E5-large-v2 | 1024 | 512 | Open source |
| GTE-large | 1024 | 512 | Alibaba, open source |

### 3.3 Embedding Best Practices

**Query vs. Document**:
Some models use different prompts for queries vs documents:
```python
# E5 format
query = "query: What is the capital of France?"
doc = "passage: Paris is the capital of France."
```

**Normalization**:
```python
embedding = embedding / np.linalg.norm(embedding)
```

**Batching**:
```python
# Batch for efficiency
embeddings = model.encode(texts, batch_size=32)
```

### 3.4 Matryoshka Embeddings

Variable-dimension embeddings - use full dimensions for quality, truncate for speed:

```python
# Full embedding
full = model.encode(text)  # 3072 dims

# Truncated for faster search
truncated = full[:512]  # Still useful
```

---

## 4. CHUNKING STRATEGIES

### 4.1 Why Chunking Matters

Chunks must be:
- **Small enough**: Fit in context, dense information
- **Large enough**: Contain complete thoughts
- **Coherent**: Make sense standalone
- **Aligned**: Match query granularity

### 4.2 Chunking Methods

**Fixed Size**:
```python
chunks = []
for i in range(0, len(text), chunk_size):
    chunks.append(text[i:i + chunk_size])
```
- Simple, predictable size
- Breaks mid-sentence/thought

**Recursive Character**:
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_text(text)
```
- Tries natural boundaries first
- Falls back to character split

**Semantic**:
```python
from langchain_experimental.text_splitter import SemanticChunker

splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile"
)
chunks = splitter.split_text(text)
```
- Groups semantically similar sentences
- Produces coherent chunks

**Document-Aware**:
- Markdown: Split by headers
- Code: Split by functions/classes
- HTML: Split by structure

### 4.3 Chunk Size Guidelines

| Content Type | Recommended Size | Overlap |
|--------------|------------------|---------|
| Dense technical | 500-800 tokens | 50-100 |
| Conversational | 800-1200 tokens | 100-200 |
| Legal/formal | 300-500 tokens | 50 |
| Code | By function/class | 0 |

### 4.4 Metadata

Always store metadata with chunks:

```python
chunk = {
    "text": "...",
    "source": "doc.pdf",
    "page": 5,
    "section": "Introduction",
    "timestamp": "2024-01-01",
    "chunk_index": 12,
}
```

---

## 5. RETRIEVAL STRATEGIES

### 5.1 Vector Similarity Search

```python
query_embedding = embed(query)
results = vector_db.query(
    vector=query_embedding,
    top_k=10,
    filter={"source": "documentation"}
)
```

**Distance Metrics**:
- Cosine: Direction similarity (most common)
- Euclidean: Absolute distance
- Dot product: For normalized vectors

### 5.2 Hybrid Search

Combine dense and sparse:

```python
# Dense results
dense_results = vector_search(query, k=20)

# Sparse results (BM25)
sparse_results = bm25_search(query, k=20)

# Reciprocal Rank Fusion
def rrf(results_list, k=60):
    scores = {}
    for results in results_list:
        for rank, doc in enumerate(results):
            if doc.id not in scores:
                scores[doc.id] = 0
            scores[doc.id] += 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

final_results = rrf([dense_results, sparse_results])
```

### 5.3 Query Expansion

**HyDE (Hypothetical Document Embeddings)**:
```python
# Generate hypothetical answer
hypothetical = llm.generate(f"Write a passage that answers: {query}")

# Embed the hypothetical answer
hyde_embedding = embed(hypothetical)

# Search with hyde embedding
results = vector_search(hyde_embedding)
```

**Multi-Query**:
```python
# Generate query variations
queries = llm.generate(f"Generate 3 different versions of: {query}")

# Search with each
all_results = []
for q in queries:
    results = vector_search(q)
    all_results.extend(results)

# Deduplicate and rank
final = deduplicate_and_rank(all_results)
```

### 5.4 Filtering

```python
results = vector_db.query(
    vector=query_embedding,
    filter={
        "date": {"$gte": "2024-01-01"},
        "category": {"$in": ["tech", "science"]},
        "is_verified": True
    }
)
```

---

## 6. RERANKING

### 6.1 Why Rerank?

Bi-encoders (embeddings) are fast but less accurate. Cross-encoders are slow but more accurate. Solution: retrieve many with bi-encoder, rerank top-k with cross-encoder.

### 6.2 Cross-Encoder Reranking

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Score each (query, document) pair
scores = reranker.predict([
    (query, doc.text) for doc in retrieved_docs
])

# Sort by score
reranked = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
```

### 6.3 Cohere Rerank

```python
import cohere

co = cohere.Client(api_key)

results = co.rerank(
    query=query,
    documents=[doc.text for doc in retrieved_docs],
    top_n=5,
    model="rerank-english-v2.0"
)
```

### 6.4 LLM-as-Reranker

```python
prompt = f"""Score each document's relevance to the query from 0-10.

Query: {query}

Document 1: {doc1}
Document 2: {doc2}
...

Return scores as JSON: {{"doc1": 8, "doc2": 3, ...}}"""

scores = llm.generate(prompt)
```

### 6.5 Reranking Pipeline

```
Initial retrieval (k=100) → First rerank (k=20) → Second rerank (k=5) → LLM context
```

---

## 7. CONTEXT ASSEMBLY

### 7.1 Prompt Templates

```python
template = """Answer the question based only on the following context:

Context:
{context}

Question: {question}

If the context doesn't contain the answer, say "I don't have enough information to answer this question."

Answer:"""
```

### 7.2 Source Attribution

```python
template = """Answer using the provided sources. Cite sources using [1], [2], etc.

Sources:
[1] {source1}
[2] {source2}
[3] {source3}

Question: {question}

Answer (with citations):"""
```

### 7.3 Context Ordering

**Lost in the Middle**: LLMs attend more to beginning and end of context.

**Solutions**:
- Put most relevant chunks first and last
- Randomize to avoid position bias
- Use hierarchical summarization

### 7.4 Context Compression

If too much context:

```python
# Summarize each chunk
summaries = [llm.summarize(chunk) for chunk in chunks]

# Or extract relevant sentences
def extract_relevant(chunk, query):
    sentences = split_sentences(chunk)
    scores = [similarity(sent, query) for sent in sentences]
    return [s for s, score in zip(sentences, scores) if score > threshold]
```

---

## 8. ADVANCED PATTERNS

### 8.1 RAPTOR (Recursive Abstractive Processing)

Build hierarchical index:
1. Embed and cluster leaf chunks
2. Summarize each cluster
3. Embed summaries
4. Repeat until single root
5. Search at multiple levels

### 8.2 Parent Document Retrieval

Store small chunks for retrieval, return parent document:

```python
# Index small chunks (for precision)
small_chunks = split(doc, size=200)
for chunk in small_chunks:
    index(chunk, parent_id=doc.id)

# Retrieve small chunks
results = search(query)

# Return parent documents
parents = [get_parent(r.parent_id) for r in results]
```

### 8.3 Multi-Vector Retrieval

Multiple embeddings per document:
- Summary embedding
- Hypothetical questions embedding
- Keyword embedding

```python
# Store multiple vectors per doc
doc_embedding = embed(doc)
summary_embedding = embed(summarize(doc))
questions_embedding = embed(generate_questions(doc))

# Search across all
results = search_multi_vector(query_embedding, [doc_idx, summary_idx, questions_idx])
```

### 8.4 Agentic RAG

Agent decides retrieval strategy:

```python
def agentic_rag(query):
    # Agent analyzes query
    analysis = llm.analyze(f"What type of retrieval does this need? {query}")

    if analysis.needs_multiple_sources:
        results = multi_source_search(query)
    elif analysis.needs_temporal:
        results = temporal_search(query, analysis.time_range)
    elif analysis.needs_aggregation:
        results = aggregate_search(query)
    else:
        results = simple_search(query)

    # Agent may iterate
    if not sufficient(results):
        refined_query = llm.refine(query, results)
        results += search(refined_query)

    return generate(query, results)
```

### 8.5 Self-RAG

Model decides when to retrieve and verifies relevance:

```
Query → [Retrieve?] → [Relevance Check] → [Generate] → [Support Check] → Output
```

---

## 9. EVALUATION

### 9.1 Retrieval Metrics

**Recall@k**: Fraction of relevant docs in top-k
**Precision@k**: Fraction of top-k that are relevant
**MRR**: Mean reciprocal rank of first relevant
**NDCG**: Normalized discounted cumulative gain

```python
def recall_at_k(retrieved, relevant, k):
    retrieved_k = set(retrieved[:k])
    return len(retrieved_k & relevant) / len(relevant)
```

### 9.2 Generation Metrics

**Faithfulness**: Is response grounded in context?
**Relevance**: Does response answer the question?
**Coherence**: Is response well-formed?

### 9.3 Evaluation Frameworks

**RAGAS**:
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy]
)
```

**TruLens**:
```python
from trulens_eval import TruChain, Feedback

feedbacks = [
    Feedback(provider.groundedness).on_context(),
    Feedback(provider.relevance).on_input_output(),
]

tru_chain = TruChain(chain, feedbacks=feedbacks)
```

### 9.4 Debugging Retrieval

```python
def debug_retrieval(query, results):
    print(f"Query: {query}")
    print(f"Query embedding norm: {np.linalg.norm(embed(query))}")

    for i, r in enumerate(results):
        print(f"\nResult {i+1} (score: {r.score:.3f}):")
        print(f"  Text: {r.text[:200]}...")
        print(f"  Metadata: {r.metadata}")

        # Check semantic overlap
        overlap = compute_overlap(query, r.text)
        print(f"  Word overlap: {overlap:.2f}")
```

---

## 10. PRODUCTION PATTERNS

### 10.1 Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_embed(text):
    return model.encode(text)

# Cache retrieval results
@lru_cache(maxsize=100)
def cached_search(query):
    return vector_db.search(embed(query))
```

### 10.2 Async Pipeline

```python
async def rag_pipeline(query):
    # Parallel retrieval from multiple sources
    results = await asyncio.gather(
        search_docs(query),
        search_web(query),
        search_database(query)
    )

    # Combine and rerank
    combined = combine_results(results)
    reranked = await rerank(combined)

    # Generate
    response = await generate(query, reranked)
    return response
```

### 10.3 Monitoring

Track:
- Retrieval latency (p50, p99)
- Empty result rate
- Context utilization
- Citation accuracy
- User feedback

### 10.4 Index Management

```python
# Regular reindexing
def reindex_stale(threshold_days=7):
    stale_docs = get_docs_older_than(threshold_days)
    for doc in stale_docs:
        new_embedding = embed(doc.text)
        vector_db.update(doc.id, new_embedding)

# Incremental updates
def add_new_docs(docs):
    for doc in docs:
        chunks = chunk(doc)
        embeddings = embed_batch([c.text for c in chunks])
        vector_db.upsert(chunks, embeddings)
```

---

## 11. COMMON ISSUES

### 11.1 Poor Retrieval

**Symptoms**: Retrieved docs don't match query
**Causes**:
- Bad chunking (incomplete thoughts)
- Wrong embedding model
- Missing metadata filters
- Query/doc distribution mismatch

**Solutions**:
- Adjust chunk size/overlap
- Try hybrid search
- Add query expansion
- Fine-tune embeddings on domain

### 11.2 Context Overflow

**Symptoms**: Too much retrieved content for context window
**Solutions**:
- Reduce k
- Compress context
- Hierarchical summarization
- Use longer context model

### 11.3 Hallucination Despite RAG

**Symptoms**: LLM ignores or contradicts context
**Causes**:
- Context not relevant
- Conflicting information
- Model override
- Poor prompt design

**Solutions**:
- Better retrieval
- Explicit grounding instructions
- Citation requirements
- Lower temperature

---

## 12. REFERENCES

**Papers**:
- Lewis et al. (2020). RAG: Retrieval-Augmented Generation
- Gao et al. (2023). RAPTOR
- Asai et al. (2023). Self-RAG

**Resources**:
- LlamaIndex: https://llamaindex.ai
- LangChain: https://langchain.com
- Pinecone Learning: https://www.pinecone.io/learn/

---

## 13. VERSION HISTORY

- **v1.0** (2024-11): Initial comprehensive version

---

*RAG is 80% data engineering, 20% ML. Invest in chunking, metadata, and evaluation infrastructure.*
