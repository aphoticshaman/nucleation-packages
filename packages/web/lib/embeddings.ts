/**
 * EMBEDDING GENERATION UTILITY
 *
 * Optional semantic search via embeddings.
 * Zero-LLM architecture uses feature-based similarity by default.
 * Embeddings are only used if OPENAI_API_KEY is explicitly configured.
 */

// Embedding model configuration
const EMBEDDING_MODEL = 'text-embedding-3-small';
const EMBEDDING_DIMENSIONS = 1536;

export interface EmbeddingResult {
  embedding: number[];
  model: string;
  tokens_used: number;
}

/**
 * Generate embedding for a text query
 * Falls back to feature-based matching if no API key
 */
export async function generateEmbedding(text: string): Promise<EmbeddingResult | null> {
  const apiKey = process.env.OPENAI_API_KEY;

  if (!apiKey) {
    console.warn('OpenAI API key not configured, using fallback similarity');
    return null;
  }

  try {
    const response = await fetch('https://api.openai.com/v1/embeddings', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: EMBEDDING_MODEL,
        input: text,
        dimensions: EMBEDDING_DIMENSIONS,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      console.error('Embedding API error:', error);
      return null;
    }

    const data = await response.json();

    return {
      embedding: data.data[0].embedding,
      model: EMBEDDING_MODEL,
      tokens_used: data.usage?.total_tokens || 0,
    };
  } catch (error) {
    console.error('Failed to generate embedding:', error);
    return null;
  }
}

/**
 * Generate embedding for a context object
 * Converts context to a descriptive text for embedding
 */
export async function generateContextEmbedding(
  domain: string,
  context: Record<string, unknown>
): Promise<EmbeddingResult | null> {
  // Build descriptive text from context
  const contextParts: string[] = [`Domain: ${domain}`];

  for (const [key, value] of Object.entries(context)) {
    if (typeof value === 'number') {
      const level =
        value < 0.25 ? 'low' : value < 0.5 ? 'moderate' : value < 0.75 ? 'elevated' : 'high';
      contextParts.push(`${key.replace(/_/g, ' ')}: ${level} (${(value * 100).toFixed(0)}%)`);
    } else if (typeof value === 'boolean') {
      contextParts.push(`${key.replace(/_/g, ' ')}: ${value ? 'yes' : 'no'}`);
    } else if (typeof value === 'string') {
      contextParts.push(`${key.replace(/_/g, ' ')}: ${value}`);
    }
  }

  const text = contextParts.join('. ');
  return generateEmbedding(text);
}

/**
 * Compute cosine similarity between two vectors
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error('Vectors must have same length');
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  if (normA === 0 || normB === 0) {
    return 0;
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * Feature-based similarity (fallback when embeddings unavailable)
 * Uses normalized feature vectors for comparison
 */
export function featureSimilarity(
  queryFeatures: Record<string, number>,
  caseFeatures: Record<string, number>
): number {
  const allKeys = new Set([...Object.keys(queryFeatures), ...Object.keys(caseFeatures)]);

  if (allKeys.size === 0) {
    return 0;
  }

  let dotProduct = 0;
  let normQuery = 0;
  let normCase = 0;

  for (const key of allKeys) {
    const qVal = queryFeatures[key] || 0;
    const cVal = caseFeatures[key] || 0;

    dotProduct += qVal * cVal;
    normQuery += qVal * qVal;
    normCase += cVal * cVal;
  }

  if (normQuery === 0 || normCase === 0) {
    return 0;
  }

  return dotProduct / (Math.sqrt(normQuery) * Math.sqrt(normCase));
}

export { EMBEDDING_DIMENSIONS };
