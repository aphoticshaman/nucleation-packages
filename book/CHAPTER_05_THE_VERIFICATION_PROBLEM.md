# Chapter 5: The Verification Problem

LLMs lie.

Not maliciously. They're not trying to deceive you. They're doing exactly what they were built to do: generate plausible-sounding text that continues from your prompt.

Sometimes plausible-sounding text happens to be true. Sometimes it doesn't. The model doesn't know the difference. It has no concept of truth—only patterns.

This chapter teaches you to verify. When to trust, when to check, and how to build habits that protect you from confident nonsense.

---

## Why They're Wrong

Understanding why LLMs are wrong helps you predict when.

### The Training Problem

LLMs learn patterns from training data. If the training data contains errors, the model learns errors. If the data contains multiple conflicting claims, the model learns them all—and might reproduce any of them.

This means:
- Obscure topics (less training data) → more errors
- Contested topics (conflicting data) → inconsistent answers
- Recent events (after training cutoff) → hallucinations or outdated info
- Technical details (easy to get slightly wrong) → subtle mistakes

### The Generation Problem

LLMs generate text one token at a time, each token conditioned on previous tokens. Once they start down a path, they continue it.

If the model starts a response with "The first president of France was..." it will complete that sentence with something plausible—even if it's wrong. The early words commit it to a course.

This creates:
- Confident delivery of false information
- Elaborate details supporting false premises
- Plausible-sounding but fabricated citations

### The Confidence Problem

Humans calibrate confidence with knowledge. We say "I think" when uncertain, "I know" when certain.

LLMs have no uncertainty. Every response is generated the same way, with the same process. They can't distinguish between something they "know" (appears often in training) and something they're "guessing" (reconstructing from sparse patterns).

This makes them dangerous. Wrong answers come with the same fluency and confidence as right answers.

---

## What Goes Wrong

Specific failure modes to watch for:

### Hallucinated Facts

The model states something specific that isn't true.

**Example:**
> Prompt: "When was the Treaty of Westphalia signed?"
> Response: "The Treaty of Westphalia was signed on October 24, 1648, in the city of Münster."

This looks great. Specific date, specific city, confident tone. The date is correct. But there were actually two treaties signed in two different cities (Münster and Osnabrück). The model gave a clean answer to a messy question.

### Fabricated Citations

The model invents sources that don't exist.

**Example:**
> Prompt: "Give me citations for the health benefits of intermittent fasting."
> Response: "According to Johnson et al. (2019) in the Journal of Nutritional Medicine..."

There's a good chance this paper doesn't exist. The model generated a plausible-sounding citation based on what citations look like. Never trust a citation from an LLM without verifying it exists.

### Outdated Information

The model's training has a cutoff date. Anything after that is unknown.

**Example:**
> Prompt: "Who won the 2024 presidential election?"
> Response: [Could be wrong, outdated, or hallucinated depending on training cutoff]

### Subtle Technical Errors

The hardest to catch. The answer is 90% right but wrong in a detail that matters.

**Example:**
> Prompt: "Show me how to securely hash a password in Python."
> Response: [Might use a deprecated method, have a subtle vulnerability, or miss a crucial step]

These errors look right to non-experts. The code runs. It seems to work. But there's a flaw that only an expert would catch.

### Plausible Nonsense

Complete fabrication that sounds reasonable.

**Example:**
> Prompt: "Tell me about the Stanford Prison Experiment's replication study in 2018."
> Response: [Might invent a detailed account of a study that never happened]

If the topic is obscure enough, the model will generate something that sounds like it could be true. Your inability to immediately disprove it doesn't make it true.

---

## The Verification Hierarchy

Not everything needs the same level of verification. Here's a hierarchy from low to high stakes:

### Level 1: Low Stakes (Verify Casually)

- Brainstorming ideas
- Writing drafts
- General explanations of well-known topics
- Creative content

For these, basic sanity checks suffice. Does it sound reasonable? Does anything seem obviously off?

### Level 2: Medium Stakes (Verify Intentionally)

- Factual claims you'll repeat to others
- Advice you'll act on
- Explanations of technical topics
- Historical or scientific information

For these, cross-reference key claims. Google the specific facts. Check if the logic holds up.

### Level 3: High Stakes (Verify Rigorously)

- Legal or medical information
- Code that handles security or money
- Decisions with significant consequences
- Anything you'll publish or present professionally

For these, assume it's wrong until proven right. Expert review. Multiple sources. Professional verification if needed.

---

## Verification Techniques

Practical methods to catch errors:

### Ask "Are You Sure?"

Surprisingly effective. After a response:

> "Are you sure about [specific claim]? Double-check your reasoning."

This prompts the model to reconsider. Sometimes it will correct itself: "Actually, I may have confused X with Y. Let me reconsider..."

Not foolproof, but catches some errors.

### Request Sources

> "What are your sources for this? Give me specific citations I can verify."

Then actually verify them. Google the citation. Check if the paper/article exists. Read the abstract to see if it supports the claim.

If the sources don't check out, the information is suspect.

### Ask for Uncertainty

> "Rate your confidence in this answer from 1-10, and explain what you might be wrong about."

This forces the model to articulate where it's on shakier ground. The low-confidence parts need more verification.

### Adversarial Follow-ups

> "Play devil's advocate. What's the strongest argument against what you just said?"

> "What would an expert in this field criticize about your answer?"

> "What am I missing? What did you oversimplify?"

These prompts surface weaknesses the model knows about but didn't mention.

### Cross-Reference

Ask the same question different ways:

> "Explain X to me."
> [get response]
> "Now explain X from first principles without referencing what you just said."

If the answers conflict, something's wrong. Investigate the discrepancy.

### External Verification

When it matters:
- Google the specific claims
- Check Wikipedia for factual grounding
- Use specialized databases for technical topics
- Ask an expert (human) for high-stakes decisions

---

## The Two-Source Rule

Here's a simple policy that prevents most harm:

**Never act on important information from an LLM without verification from at least one independent source.**

Independent means: not the same LLM rephrasing itself. A different LLM, a search engine, a human expert, an official document.

This catches:
- Hallucinations (won't be corroborated)
- Outdated info (current sources will contradict)
- Model-specific biases (different models will vary)

It won't catch:
- Errors shared across sources (rare but possible)
- Fast-moving situations where sources haven't updated

For truly high-stakes decisions, two sources may not be enough. Use judgment.

---

## Domain-Specific Risks

Different domains have different risks:

### Medical Information

LLMs should never be your primary medical source. They can:
- Confuse similar conditions
- Miss dangerous interactions
- Give outdated treatment recommendations
- Fail to recognize when something is serious

Use them for general health literacy, not diagnosis or treatment decisions. Always verify with actual medical professionals.

### Legal Information

Law is jurisdictional, situational, and constantly changing. LLMs:
- Might cite laws from the wrong jurisdiction
- May reference outdated statutes
- Can miss crucial exceptions or conditions
- Don't know your specific situation

Use them to understand general concepts, not as legal advice. Real legal decisions need real lawyers.

### Financial Information

Tax law, investment regulations, and financial rules are complex. LLMs:
- May not know current tax brackets or limits
- Can miss important qualifications and conditions
- Might not account for your specific situation
- Could suggest strategies with unintended consequences

Use them to learn concepts, not to make financial decisions without verification.

### Code and Security

Code that looks correct can have subtle bugs or vulnerabilities. LLMs:
- May use deprecated methods
- Can introduce security vulnerabilities
- Might miss edge cases
- Sometimes produce code that works but is inefficient

For anything involving security, authentication, or financial transactions: expert code review.

### Historical and Scientific Facts

These seem safe but aren't always. LLMs:
- Can conflate similar events or people
- May present disputed claims as fact
- Sometimes invent plausible-sounding details
- Can perpetuate common misconceptions

Cross-reference specific facts, especially if you're using them professionally.

---

## Building Verification Habits

Verification should be automatic, not occasional.

### The Pause

Before acting on LLM output, pause and ask: "Is this something I should verify?"

Quick mental checklist:
- Will I act on this?
- Will I repeat this to others?
- Could being wrong cause harm?
- Is this a domain with high error risk?

If any answer is yes, verify.

### The Annotate

When taking notes from LLM conversations, mark what's verified vs. unverified.

```
[VERIFIED] The population of Tokyo is approximately 14 million
[UNVERIFIED] The city's name means "Eastern Capital" in Japanese
[NEED TO CHECK] The population increased by 3% in the last decade
```

### The Default

Make skepticism your default posture. Trust nothing automatically. Verify anything that matters.

This isn't paranoia—it's professionalism. You wouldn't publish a statistic without checking it. You wouldn't cite a study without confirming it exists. Apply the same standard to LLM output.

---

## The Meta-Verification Problem

Here's the uncomfortable truth: you can't verify everything.

You don't have time to fact-check every claim in a conversation. You're not an expert in every domain. Some verification requires expertise you don't have.

So what do you do?

### Prioritize by Stakes

Verify the high-stakes stuff. Let the low-stakes stuff go. Your brainstorming session doesn't need footnotes. Your published article does.

### Know Your Blindspots

What domains are you NOT equipped to verify? Medical, legal, financial, technical? Be extra cautious in those areas.

### Use Trust Networks

You can't personally verify everything, but you can rely on institutions and people who do. Peer-reviewed journals. Established experts. Official sources. These aren't perfect, but they have verification built in.

### Accept Uncertainty

Sometimes you won't be sure. That's okay. The goal isn't perfect knowledge—it's appropriate confidence. "I think this is correct but haven't fully verified it" is an honest position.

---

## Summary

- **LLMs lie confidently** because they generate plausible text, not true text.

- **Failure modes:** hallucinated facts, fabricated citations, outdated info, subtle technical errors, plausible nonsense.

- **Verification hierarchy:** casual for low stakes, intentional for medium stakes, rigorous for high stakes.

- **Techniques:** ask "are you sure?", request sources, ask for uncertainty, adversarial follow-ups, cross-reference, external verification.

- **Two-source rule:** Never act on important LLM information without independent verification.

- **Domain risks:** Medical, legal, financial, code, and factual domains each have specific risks.

- **Build habits:** Pause before acting, annotate verification status, default to skepticism.

The next chapter takes everything you've learned and turns it inward: using AI not just to get things done, but to improve how you think.
