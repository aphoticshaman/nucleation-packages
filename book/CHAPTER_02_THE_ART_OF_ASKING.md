# Chapter 2: The Art of Asking

The difference between a useless LLM response and a brilliant one is usually the prompt.

Not the model. Not the temperature setting. Not which company made it. The prompt.

This chapter teaches you how to ask in ways that get useful answers. No jargon, no theory—just the patterns that work and the mistakes that don't.

---

## The Anatomy of a Good Prompt

Every effective prompt has three parts:

1. **Context:** What does the AI need to know about your situation?
2. **Task:** What specifically do you want it to do?
3. **Constraints:** What are the boundaries, format, or requirements?

Bad prompts are missing one or more of these. Good prompts nail all three.

### Example: Planning a Presentation

**Bad prompt:**
> "Help me with my presentation."

This gives the AI nothing to work with. What kind of presentation? For whom? About what? How long? What's the goal?

**Good prompt:**
> "I'm giving a 10-minute presentation to my company's executive team (CEO, CFO, CTO) about why we should adopt a new project management tool. They're skeptical of change and care most about ROI and implementation time. Help me structure the presentation and anticipate their objections."

See the difference?

- **Context:** Executive team presentation, 10 minutes, skeptical audience, ROI-focused
- **Task:** Structure the presentation, anticipate objections
- **Constraints:** Time limit, specific audience concerns

The AI now has everything it needs to give you a tailored, useful response.

---

## The Six Patterns That Work

After analyzing thousands of prompts, I've identified six patterns that consistently produce better results. Use them.

### Pattern 1: Role Assignment

Tell the AI who to be.

> "You are an experienced immigration lawyer..."
> "Act as a senior software engineer reviewing code..."
> "Pretend you're a hostile journalist looking for holes in my argument..."

Why this works: The AI has been trained on text from many perspectives. When you assign a role, you're telling it which subset of its knowledge to emphasize. A "senior software engineer" gives different code feedback than a "junior developer" or "security auditor."

**Example:**

Without role:
> "Review this email."

With role:
> "You are a communications consultant who has coached Fortune 500 executives. Review this email I'm about to send to my CEO requesting a promotion. Is my tone right? Am I being too aggressive or too passive?"

The second prompt will catch things the first one misses—because you've told the AI what expertise to bring.

### Pattern 2: Step-by-Step Instruction

Break complex tasks into explicit steps.

> "First, analyze X. Then, based on that analysis, suggest Y. Finally, explain the tradeoffs of each suggestion."

Why this works: LLMs process text sequentially. When you give one giant request, they might skip steps or combine them sloppily. When you break it into explicit steps, they follow the sequence.

**Example:**

Without steps:
> "Help me improve my resume."

With steps:
> "I'm going to paste my resume below. I want you to:
> 1. First, identify the three strongest bullet points and explain why they work
> 2. Then, identify the three weakest bullet points and explain what's wrong with them
> 3. Finally, rewrite the weak ones using the same patterns that made the strong ones work
> 
> Here's my resume: [paste]"

The step-by-step version produces structured, actionable feedback instead of vague suggestions.

### Pattern 3: Examples (Few-Shot)

Show the AI what you want.

> "Here's an example of what I'm looking for: [example]. Now do the same for [my thing]."

Why this works: Examples are worth a thousand words of instruction. Instead of describing the format, tone, or style you want, you can demonstrate it.

**Example:**

Without example:
> "Write a product description for my candle."

With example:
> "Here's a product description style I like:
> 
> 'The Morning Ritual mug isn't just ceramic—it's the first five minutes of a better day. Handcrafted in Portland, holds 12oz of whatever gets you going, and keeps it warm for the entire morning meeting you're secretly dreading.'
> 
> Now write a similar description for my lavender candle. It's hand-poured, made with soy wax, burns for 40 hours, and I want the same slightly irreverent but warm tone."

The AI now knows exactly what style you're after.

### Pattern 4: Negative Constraints

Tell the AI what NOT to do.

> "Don't use jargon."
> "Avoid bullet points—use prose only."
> "Don't give me generic advice—be specific to my situation."
> "Skip the introduction and get straight to the point."

Why this works: LLMs default to common patterns from their training data. If you don't want those patterns, you have to explicitly exclude them.

**Example:**

Without negative constraints:
> "Explain blockchain to me."

With negative constraints:
> "Explain blockchain to me. Don't use any technical jargon, don't use the words 'decentralized' or 'distributed,' and don't give me the standard cryptocurrency pitch. Explain it like you're telling a curious 10-year-old who asked 'what's that?' when they heard it on TV."

The second version forces creativity instead of getting the stock explanation.

### Pattern 5: Thinking Out Loud (Chain of Thought)

Ask the AI to show its reasoning.

> "Think through this step by step."
> "Explain your reasoning as you go."
> "Before giving your final answer, consider the alternatives."

Why this works: When LLMs explain their reasoning, they actually reason better. The act of articulating the logic prevents shortcuts and catches errors.

**Example:**

Without chain of thought:
> "Should I take this job offer?"

With chain of thought:
> "I'm deciding whether to take a job offer. Before you give me advice, think through it step by step:
> - What are the key factors I should weigh?
> - What are the potential upsides of taking it?
> - What are the potential downsides?
> - What questions should I be asking about the offer that I might not have thought of?
> - What would the 'regret minimization' framework say about this decision?
> 
> Then, based on all of that, give me your recommendation."

This produces analysis, not just an answer.

### Pattern 6: Persona + Audience

Specify who's writing and who's reading.

> "Write this as [persona] for [audience]."

Why this works: The same information can be communicated a hundred different ways depending on speaker and listener. A doctor explaining a diagnosis to a patient writes differently than a doctor writing to another doctor.

**Example:**

Without persona/audience:
> "Write an explanation of compound interest."

With persona/audience:
> "Write an explanation of compound interest as a friendly financial advisor would explain it to a 22-year-old who just got their first real job and has $500/month they could potentially save. Make it conversational, not textbook-y, and end with a concrete action step."

---

## The Mistakes That Kill Prompts

Just as important as what works is what doesn't. Here are the prompt-killers:

### Mistake 1: Assuming the AI Knows Your Context

**Bad:** "Continue where we left off."
**Problem:** Every conversation starts fresh. The AI doesn't remember previous sessions.

**Fix:** Provide context every time, or copy relevant information from previous conversations.

### Mistake 2: Being Vague About Format

**Bad:** "Give me some ideas."
**Problem:** You'll get a random format—maybe bullets, maybe paragraphs, maybe 5 ideas, maybe 50.

**Fix:** Specify format: "Give me exactly 5 ideas, each in one sentence, ranked by feasibility."

### Mistake 3: Asking Compound Questions

**Bad:** "What's the best programming language and also how do I learn it and what jobs use it?"
**Problem:** The AI will superficially address all three instead of deeply addressing any.

**Fix:** Break into separate prompts, or explicitly structure: "I have three questions. Answer each one separately..."

### Mistake 4: Not Iterating

**Bad:** *Gets mediocre response, gives up*
**Problem:** You're leaving value on the table. First responses are rarely best responses.

**Fix:** Follow up. "That's close but too formal—make it more casual." "Expand on the second point." "Now critique what you just said."

### Mistake 5: Accepting Confident Wrong Answers

**Bad:** *Trusts everything the AI says*
**Problem:** LLMs sound confident even when they're wrong. They don't know what they don't know.

**Fix:** Ask for sources. Ask "Are you sure?" (surprisingly effective). Cross-reference important claims. More on this in Chapter 5.

---

## The Meta-Prompt: Prompts That Write Prompts

Here's a trick that changes everything:

Instead of writing prompts yourself, ask the AI to help you write better prompts.

> "I want to [goal]. What questions should you ask me before you help with this? What context do you need?"

This flips the script. Instead of guessing what information the AI needs, you let the AI tell you.

**Example:**

You type:
> "I want you to help me negotiate a raise. Before we start, what do you need to know about my situation to give me good advice?"

AI responds with questions:
> - What's your current salary and role?
> - How long have you been at this company?
> - What's the market rate for your position?
> - Do you have other offers or leverage?
> - What's your relationship like with your manager?
> - Has the company done raises recently?
> - What's your BATNA (best alternative)?

You answer those questions, and now the AI has perfect context for helping you.

### The Advanced Version

Take this a step further:

> "I'm going to describe what I want help with. Don't help me yet—instead, write a better prompt that I should use to get the best possible help. Then I'll use that prompt."

You describe your situation. The AI writes an optimized prompt for your situation. You use that prompt. Better results with less trial and error.

---

## Real-World Examples

Let's see these patterns in action across different use cases.

### Example 1: Email Writing

**Amateur prompt:**
> "Write an email asking for a meeting."

**Professional prompt:**
> "Write a professional but warm email to a potential client named Sarah, who I met briefly at a conference last week. We talked about their company's challenge with customer retention. I want to propose a 20-minute call to discuss how I might help. Keep it under 100 words, and don't be salesy—be helpful and low-pressure. Use a casual but professional tone."

### Example 2: Research

**Amateur prompt:**
> "Tell me about climate change."

**Professional prompt:**
> "I'm preparing for a debate where I need to argue for stronger climate regulations. I already know the basic science. What I need are:
> 1. The three strongest economic arguments FOR regulation (not moral arguments—I need to convince business-minded skeptics)
> 2. The most common counter-arguments I'll face
> 3. Data or studies I can cite for each point
> 
> Prioritize recent data (last 5 years) and sources that aren't easy to dismiss as partisan."

### Example 3: Problem-Solving

**Amateur prompt:**
> "Why isn't my business making money?"

**Professional prompt:**
> "I run a small online tutoring service. Revenue is $8K/month but expenses are $9K/month. I have 45 active students paying $50/month for 4 hours of tutoring. I have 3 tutors I pay $25/hour. My other costs are $500/month for the platform and $500/month for marketing.
> 
> Before suggesting solutions, first diagnose the problem. Walk me through the unit economics and identify where the leak is. Then suggest 3-5 realistic ways to fix it, ranked by how fast I could implement each one."

### Example 4: Learning

**Amateur prompt:**
> "Explain machine learning."

**Professional prompt:**
> "Explain machine learning to me in three phases:
> 1. First, the 'explain like I'm 5' version—maximum simplicity, use analogies
> 2. Then, the 'explain like I'm a smart adult who knows basic math'—more precise but still accessible
> 3. Finally, give me a concrete example I can try myself, even if I don't know how to code
> 
> Don't use any jargon without defining it first."

---

## The 10-Second Prompt Checklist

Before you hit enter, ask yourself:

1. **Context?** — Does the AI know my situation?
2. **Task?** — Is what I want crystal clear?
3. **Constraints?** — Format, length, tone, audience?
4. **Examples?** — Would showing what I want be clearer than telling?
5. **Role?** — Who should the AI "be" for this task?

If you're missing any of these, add them. It takes 30 seconds and doubles the quality of your response.

---

## Summary

- **Good prompts have three parts:** context, task, and constraints. Don't skip any.

- **Use the six patterns:** role assignment, step-by-step, examples, negative constraints, chain of thought, and persona+audience.

- **Avoid the five mistakes:** assuming context, vague format, compound questions, not iterating, and blind trust.

- **Use meta-prompts:** Ask the AI what it needs to know. Have it write better prompts for you.

- **Always iterate:** First responses aren't final responses. Follow up, refine, challenge.

The next chapter takes this further: instead of asking for single responses, you'll learn to build systems—workflows, templates, and automations that work for you repeatedly without re-prompting from scratch.
