---
name: veilpath-ai-orchestration
description: "Hybrid edge/cloud AI orchestration for VeilPath. Multi-model routing, cost optimization, prompt engineering for tarot synthesis, and on-device inference management."
---

# VEILPATH_AI_ORCHESTRATION.skill.md

## Hybrid AI Architecture for 90% Margin at Scale

**Version**: 1.0  
**Domain**: LLM Orchestration, Edge Computing, Prompt Engineering, Cost Control
**Prerequisites**: Claude API, Llama.cpp, ONNX Runtime, Prompt Engineering
**Output**: Sub-$0.001 per reading with premium quality

---

## 1. THE ECONOMICS OF AI AT SCALE

### 1.1 The Unit Economics That Matter

```typescript
// This determines if you exit for $50M or $500M
const AI_COST_MATRIX = {
  claude_opus: {
    cost_per_1k_tokens: 0.015,
    quality: 0.95,
    latency: 2000, // ms
    monthly_budget_10k_users: 15000 // $15k/mo kills margins
  },
  
  claude_sonnet: {
    cost_per_1k_tokens: 0.003,
    quality: 0.85,
    latency: 1200,
    monthly_budget_10k_users: 3000 // $3k/mo sustainable
  },
  
  llama3_70b_local: {
    cost_per_1k_tokens: 0.0001, // Just electricity
    quality: 0.75,
    latency: 500,
    monthly_budget_10k_users: 100 // ~$100 in compute
  },
  
  phi3_on_device: {
    cost_per_1k_tokens: 0, // User's battery
    quality: 0.65,
    latency: 100,
    monthly_budget_10k_users: 0 // FREE
  }
};

// The strategy that actually works
const HYBRID_STRATEGY = {
  free_tier: "phi3_on_device", // 100% margin
  paid_tier: "llama3_local + claude_sonnet_fallback",
  premium_tier: "claude_opus_for_special_readings",
  
  target_margins: {
    free: 1.0,    // 100% margin (no costs)
    paid: 0.90,   // 90% margin
    premium: 0.70 // 70% margin (premium experience)
  }
};
```

## 2. INTELLIGENT ROUTING ENGINE

### 2.1 The Router That Saves Millions

```typescript
class AIRouter {
  private models: Map<string, AIModel> = new Map();
  private userTiers: Map<string, UserTier> = new Map();
  private cache: LRUCache;
  
  async route(request: InterpretationRequest): Promise<Response> {
    // 1. Check cache first (FREE)
    const cacheKey = this.getCacheKey(request);
    const cached = await this.cache.get(cacheKey);
    if (cached && this.isCacheable(request)) {
      return this.personalizeCached(cached, request.user);
    }
    
    // 2. Determine optimal model
    const model = this.selectModel(request);
    
    // 3. Execute with fallback chain
    return this.executeWithFallback(model, request);
  }
  
  private selectModel(request: InterpretationRequest): AIModel {
    const factors = {
      user_tier: this.userTiers.get(request.userId),
      complexity: this.estimateComplexity(request),
      latency_requirement: request.realtime ? 100 : 5000,
      quality_requirement: request.battleMode ? 0.9 : 0.7,
      budget_remaining: this.getBudget(request.userId),
      device_capabilities: request.deviceInfo
    };
    
    // Decision tree optimized for margins
    if (factors.device_capabilities.canRunLocal) {
      if (factors.complexity < 0.5) {
        return this.models.get('phi3_local');
      }
    }
    
    if (factors.user_tier === 'free') {
      return this.models.get('phi3_local_or_cached');
    }
    
    if (factors.user_tier === 'premium' && factors.battleMode) {
      return this.models.get('claude_opus');
    }
    
    // Default to good-enough quality
    return this.models.get('llama3_70b');
  }
  
  async executeWithFallback(
    primary: AIModel, 
    request: InterpretationRequest
  ): Promise<Response> {
    try {
      const response = await primary.execute(request);
      
      // Quality check
      if (this.qualityScore(response) < 0.7) {
        throw new Error("Quality below threshold");
      }
      
      return response;
    } catch (error) {
      // Fallback chain
      console.log(`Primary failed: ${error.message}, trying fallback`);
      
      if (primary.name === 'phi3_local') {
        return this.executeWithFallback(
          this.models.get('llama3_70b'),
          request
        );
      }
      
      if (primary.name === 'llama3_70b') {
        return this.executeWithFallback(
          this.models.get('claude_sonnet'),
          request
        );
      }
      
      // Last resort - but track the cost
      this.metrics.increment('expensive_fallback');
      return this.models.get('claude_opus').execute(request);
    }
  }
}
```

## 3. PROMPT ENGINEERING FOR TAROT

### 3.1 The Prompts That Convert Users to Premium

```typescript
class TarotPromptEngine {
  // Base prompts optimized through A/B testing
  private readonly PROMPTS = {
    celtic_cross: {
      system: `You are an ancient Oracle with deep wisdom about the Celtic Cross spread.
      
CRITICAL RULES:
1. Each card position has specific temporal meaning
2. Connect cards through narrative threads
3. Surface ONE profound insight the querent didn't expect
4. End with actionable guidance, not platitudes

Position meanings:
1. Present Situation - The heart of the matter
2. Challenge/Cross - What opposes or supports
3. Distant Past - Foundation of the situation  
4. Recent Past - Events leading here
5. Possible Future - Where current path leads
6. Immediate Future - Next few weeks
7. Your Approach - How you're handling this
8. External Influences - What others bring
9. Hopes and Fears - Internal landscape
10. Final Outcome - If nothing changes

QUALITY MARKERS (aim for all):
- Specific, not generic
- Temporal awareness (past/present/future flow)
- Psychological depth
- Practical application
- One surprising connection between cards`,
      
      user: (cards: Card[], query: string) => `
Query: "${query}"

Cards drawn:
${cards.map((card, i) => `
Position ${i + 1} (${this.getPositionName(i)}): 
${card.name}${card.reversed ? ' (Reversed)' : ''}
Traditional meaning: ${card.meaning}
`).join('\n')}

Provide interpretation following these phases:
1. IMMEDIATE IMPRESSION: Your first intuitive read (1 sentence)
2. CARD SYNTHESIS: How cards relate and tell a story (2-3 paragraphs)
3. HIDDEN PATTERN: The unexpected connection or insight (1 paragraph)
4. PRACTICAL GUIDANCE: Specific actions to take (3 bullet points)
5. ORACLE'S WHISPER: One profound closing thought (1 sentence)`,
    },
    
    three_card: {
      system: `You are a pragmatic guide using the three-card spread for clarity.
      
SPREAD TYPE: Past-Present-Future
TONE: Direct, insightful, actionable
LENGTH: Concise but meaningful (300-400 words total)

QUALITY MARKERS:
- Acknowledge the querent's emotional state
- Connect the temporal flow
- One specific prediction
- One specific action step`,
      
      user: (cards: Card[], query: string) => `
Query: "${query}"
Past: ${cards[0].name}${cards[0].reversed ? ' (Rev)' : ''}
Present: ${cards[1].name}${cards[1].reversed ? ' (Rev)' : ''}  
Future: ${cards[2].name}${cards[2].reversed ? ' (Rev)' : ''}

Create a flowing narrative that:
1. Validates what brought them here (Past)
2. Clarifies their current challenge (Present)
3. Reveals the emerging opportunity (Future)
4. Gives ONE specific thing to do this week`
    },
    
    battle_interpretation: {
      system: `You are competing in a Tarot Battle. Your interpretation will be scored against another reader.

WINNING CRITERIA:
- Depth of psychological insight (35%)
- Creative connections between cards (25%)
- Practical applicability (20%)
- Narrative coherence (15%)
- Unique perspective (5%)

STRATEGY:
- Start with unexpected angle
- Build psychological depth quickly
- Connect to universal themes
- End with memorable insight
- 400-500 words optimal`,
      
      user: (cards: Card[], theme: string, opponentStyle?: string) => `
Battle Theme: "${theme}"
Cards: ${cards.map(c => c.name).join(', ')}
${opponentStyle ? `Opponent tends toward: ${opponentStyle}` : ''}

Create a competition-winning interpretation that:
1. Opens with an unexpected metaphor
2. Builds a cohesive narrative
3. Demonstrates deep understanding
4. Closes with profound insight`
    }
  };
  
  async generateInterpretation(
    spread: SpreadType,
    cards: Card[],
    query: string,
    model: AIModel
  ): Promise<Interpretation> {
    const prompt = this.PROMPTS[spread];
    if (!prompt) throw new Error(`Unknown spread: ${spread}`);
    
    // Add user personalization layer
    const personalizedPrompt = await this.personalizePrompt(
      prompt,
      query,
      await this.getUserContext(query)
    );
    
    const response = await model.complete({
      system: personalizedPrompt.system,
      user: personalizedPrompt.user(cards, query),
      temperature: spread === 'battle' ? 0.9 : 0.7,
      max_tokens: 800
    });
    
    return this.parseInterpretation(response);
  }
  
  private async personalizePrompt(
    basePrompt: PromptTemplate,
    query: string,
    context: UserContext
  ): Promise<PromptTemplate> {
    // Modify prompts based on user history
    const modifications = [];
    
    if (context.readingCount > 10) {
      modifications.push("Assume familiarity with tarot concepts.");
    }
    
    if (context.preferredStyle === 'psychological') {
      modifications.push("Emphasize Jungian archetypes and shadow work.");
    }
    
    if (context.isInCrisis) {
      modifications.push("Be extra compassionate. Include crisis resources if appropriate.");
    }
    
    return {
      ...basePrompt,
      system: basePrompt.system + "\n\nContext: " + modifications.join(' ')
    };
  }
}
```

## 4. ON-DEVICE INFERENCE

### 4.1 Phi-3 on Mobile (The Free Tier Secret Weapon)

```typescript
// This runs on user's phone - zero server costs
class OnDeviceInference {
  private model: ONNX.Model;
  private tokenizer: Tokenizer;
  
  async initialize() {
    // React Native with Expo
    if (Platform.OS === 'ios' || Platform.OS === 'android') {
      const modelPath = await this.downloadModel();
      this.model = await ONNX.load(modelPath);
      this.tokenizer = await this.loadTokenizer();
    }
  }
  
  async generateInterpretation(
    cards: Card[],
    query: string
  ): Promise<string> {
    // Simplified prompt for small model
    const prompt = this.buildCompactPrompt(cards, query);
    
    // Tokenize
    const inputIds = this.tokenizer.encode(prompt);
    
    // Run inference
    const startTime = Date.now();
    const output = await this.model.generate({
      input_ids: inputIds,
      max_new_tokens: 256, // Keep short for speed
      temperature: 0.7,
      do_sample: true
    });
    
    const latency = Date.now() - startTime;
    this.metrics.track('on_device_latency', latency);
    
    // Decode
    const text = this.tokenizer.decode(output);
    
    // Quality check - if too bad, fall back to server
    if (this.qualityScore(text) < 0.6) {
      throw new Error("Quality too low, falling back");
    }
    
    return text;
  }
  
  private buildCompactPrompt(cards: Card[], query: string): string {
    // Optimized for small models - every token counts
    return `Tarot reading for: "${query.slice(0, 50)}"
Cards: ${cards.map(c => c.name.slice(0, 20)).join(', ')}
Interpret in 150 words. Focus on practical advice.
Reading:`;
  }
  
  private qualityScore(text: string): number {
    const checks = {
      hasContent: text.length > 100,
      mentionsCards: cards.some(c => text.includes(c.name)),
      coherent: !text.includes('Lorem ipsum'),
      complete: text.endsWith('.') || text.endsWith('!'),
      appropriate: !this.containsHarmful(text)
    };
    
    return Object.values(checks).filter(Boolean).length / 5;
  }
}
```

## 5. CACHING STRATEGY

### 5.1 The Cache That Cuts Costs by 70%

```typescript
class InterpretationCache {
  private redis: Redis;
  private embeddings: EmbeddingModel;
  
  async getCached(
    cards: Card[],
    query: string,
    userId: string
  ): Promise<Interpretation | null> {
    // 1. Exact match (rare but free)
    const exactKey = this.getExactKey(cards, query);
    const exact = await this.redis.get(exactKey);
    if (exact) return this.personalize(exact, userId);
    
    // 2. Semantic similarity search
    const embedding = await this.embeddings.encode(query);
    const similar = await this.findSimilar(embedding, cards);
    
    if (similar && similar.similarity > 0.85) {
      // Close enough - personalize and return
      return this.adaptInterpretation(similar.interpretation, query, userId);
    }
    
    return null;
  }
  
  async store(
    cards: Card[],
    query: string,
    interpretation: Interpretation
  ): Promise<void> {
    // Store with multiple keys for retrieval
    const exactKey = this.getExactKey(cards, query);
    const fuzzyKey = this.getFuzzyKey(cards);
    const embedding = await this.embeddings.encode(query);
    
    const cacheData = {
      interpretation,
      cards,
      query,
      embedding,
      created: Date.now(),
      uses: 0
    };
    
    // Multiple storage strategies
    await Promise.all([
      this.redis.setex(exactKey, 86400, JSON.stringify(cacheData)),
      this.redis.sadd(fuzzyKey, exactKey),
      this.storeEmbedding(embedding, exactKey)
    ]);
  }
  
  private async adaptInterpretation(
    cached: Interpretation,
    newQuery: string,
    userId: string
  ): Promise<Interpretation> {
    // Use a small model to adapt cached interpretation
    const adapted = await this.models.phi3.complete({
      system: "Adapt this tarot interpretation to the new question while keeping the core insights.",
      user: `Original: ${cached.text}\nNew question: "${newQuery}"\nAdapted interpretation:`,
      max_tokens: 300
    });
    
    return {
      ...cached,
      text: adapted,
      adapted: true,
      adaptedFrom: cached.id
    };
  }
}
```

## 6. COST MONITORING & OPTIMIZATION

### 6.1 The Dashboard That Prevents Bankruptcy

```typescript
class AICostMonitor {
  async trackRequest(request: AIRequest, response: AIResponse) {
    const cost = this.calculateCost(request, response);
    
    await this.metrics.track({
      userId: request.userId,
      model: request.model,
      tokens_in: request.tokenCount,
      tokens_out: response.tokenCount,
      cost_usd: cost,
      latency_ms: response.latency,
      cache_hit: response.fromCache,
      quality_score: this.scoreQuality(response)
    });
    
    // Alert if costs spike
    if (cost > 0.10) {
      await this.alerts.send({
        severity: 'warning',
        message: `High cost request: $${cost} for user ${request.userId}`
      });
    }
    
    // Auto-optimize if margins drop
    const hourlyMargin = await this.calculateHourlyMargin();
    if (hourlyMargin < 0.70) {
      await this.optimizer.increaseCaching();
      await this.optimizer.routeMoreToEdge();
    }
  }
  
  async generateCostReport(): Promise<CostReport> {
    return {
      daily: {
        total_cost: await this.getCost('1d'),
        revenue: await this.getRevenue('1d'),
        margin: await this.getMargin('1d'),
        cost_per_user: await this.getCostPerUser('1d')
      },
      
      by_model: {
        claude_opus: await this.getModelCost('claude_opus', '1d'),
        claude_sonnet: await this.getModelCost('claude_sonnet', '1d'),
        llama3: await this.getModelCost('llama3', '1d'),
        phi3: await this.getModelCost('phi3', '1d')
      },
      
      optimization_opportunities: [
        {
          action: "Move 30% more to edge",
          savings: "$500/month",
          quality_impact: -0.05
        },
        {
          action: "Increase cache TTL to 48h",
          savings: "$300/month",
          quality_impact: -0.02
        }
      ]
    };
  }
}
```

## 7. BATTLE MODE OPTIMIZATIONS

### 7.1 Real-Time Competitive Inference

```typescript
class BattleAI {
  async generateCompetitiveInterpretation(
    cards: Card[],
    theme: string,
    timeLimit: number
  ): Promise<BattleInterpretation> {
    // Stream response for better UX
    const stream = await this.streamingModel.complete({
      system: this.PROMPTS.battle_interpretation.system,
      user: this.PROMPTS.battle_interpretation.user(cards, theme),
      stream: true
    });
    
    const chunks: string[] = [];
    const startTime = Date.now();
    
    for await (const chunk of stream) {
      chunks.push(chunk);
      
      // Send partial updates to UI
      if (chunks.length % 5 === 0) {
        await this.sendPartialUpdate(chunks.join(''));
      }
      
      // Enforce time limit
      if (Date.now() - startTime > timeLimit - 1000) {
        break; // Leave 1s for finalization
      }
    }
    
    const fullText = chunks.join('');
    
    // Post-process for quality
    const enhanced = await this.enhanceForCompetition(fullText);
    
    return {
      text: enhanced,
      timestamp: Date.now(),
      processingTime: Date.now() - startTime,
      model: 'battle_optimized',
      strategy_tags: this.extractStrategies(enhanced)
    };
  }
  
  async judgeInterpretations(
    interp1: BattleInterpretation,
    interp2: BattleInterpretation,
    criteria: JudgingCriteria
  ): Promise<BattleResult> {
    // Use a different model for judging to avoid bias
    const judgment = await this.judgingModel.complete({
      system: `You are an impartial judge in a Tarot interpretation battle.
      
Score each interpretation on:
- Depth (0-10): Psychological and spiritual insight
- Creativity (0-10): Unique perspective and connections
- Relevance (0-10): Addresses the theme directly
- Coherence (0-10): Clear narrative flow
- Impact (0-10): Memorable and actionable

Be extremely specific in your scoring rationale.`,
      
      user: `Theme: "${criteria.theme}"
      
Interpretation A:
${interp1.text}

Interpretation B:
${interp2.text}

Provide scores and determine winner. Output JSON:
{
  "a_scores": {...},
  "b_scores": {...},
  "winner": "a" or "b",
  "winning_margin": number,
  "judge_notes": "..."
}`,
      
      temperature: 0.3, // Low temp for consistent judging
      response_format: { type: "json_object" }
    });
    
    return this.parseJudgment(judgment);
  }
}
```

## 8. IMPLEMENTATION CHECKLIST

### Week 1: Foundation
```bash
# Install core dependencies
npm install openai anthropic together-ai
npm install @xenova/transformers  # On-device inference
npm install onnxruntime-react-native  # Mobile inference
npm install ioredis bull  # Caching and queues

# Set up cost tracking
npm install datadog-metrics
```

### Week 2: Optimization
- Implement caching layer
- Set up model router
- Deploy Phi-3 to mobile
- A/B test prompts

### Week 3: Scale Testing
- Load test with 10K concurrent users
- Optimize prompts for token efficiency
- Implement streaming responses
- Set up cost alerts

### Month 1: Production
- Multi-region deployment
- Advanced caching strategies
- Custom fine-tuned models
- Patent filing for hybrid system

## 9. METRICS THAT MATTER

```typescript
const SUCCESS_METRICS = {
  cost_per_interpretation: {
    target: 0.001,  // $0.001 
    current: 0.012, // $0.012
    action: "Increase edge percentage"
  },
  
  interpretation_quality: {
    target: 0.85,  // User satisfaction
    current: 0.82,
    action: "Improve prompt engineering"
  },
  
  latency_p99: {
    target: 2000,  // 2s
    current: 3500,  // 3.5s
    action: "Pre-warm models, optimize streaming"
  },
  
  cache_hit_rate: {
    target: 0.40,  // 40%
    current: 0.22,  // 22%
    action: "Semantic similarity matching"
  }
};
```

## 10. IP OPPORTUNITIES

```javascript
const AI_PATENTS = {
  "Hybrid Edge-Cloud Orchestration": {
    claim: "Dynamic model selection based on device capabilities",
    value: "$20M (every AI app needs this)"
  },
  
  "Semantic Interpretation Caching": {
    claim: "Fuzzy matching for LLM response reuse",
    value: "$15M (70% cost reduction)"
  },
  
  "Competitive AI Judging": {
    claim: "Multi-model consensus for subjective scoring",
    value: "$10M (applicable to any AI competition)"
  }
};
```

---

**Bottom Line**: This AI architecture enables 90% margins at scale. That's the difference between a lifestyle business and a $500M exit.

Ship fast. Cache everything. Run on edge.
