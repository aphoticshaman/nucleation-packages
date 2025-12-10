# Chapter 3: Building Your Second Brain

By now you know how to ask good questions. Single prompts that get good responses.

But here's the problem: you're starting from scratch every time.

Every new conversation, you re-explain your context. Every similar task, you re-write similar prompts. Every workflow, you manually repeat the same steps.

This is exhausting. And it doesn't scale.

This chapter teaches you to build systems—reusable frameworks that work for you repeatedly. Templates you fill in. Workflows that remember your patterns. A second brain that compounds your effort over time.

---

## The Problem With Starting Fresh

Most people use LLMs like this:

**Monday:** "Help me write a cold email to a potential client."
*AI produces decent email*

**Tuesday:** "I need to write another cold email to a different client."
*Repeats the entire prompt, gets inconsistent quality*

**Wednesday:** "Another cold email..."
*Gets tired, writes lazy prompt, gets lazy response*

By Friday, they've written five versions of basically the same prompt and gotten five inconsistent responses.

There's a better way.

---

## Template Thinking

Instead of writing prompts from scratch, create templates—reusable structures with blanks you fill in.

### The Basic Template Formula

```
[CONTEXT BLOCK]
I am [your role/situation].
I need to [goal].
The audience is [who will see this].
The constraints are [limits, requirements].

[TASK BLOCK]
Please [specific action].
Format: [how you want it structured].
Length: [word count, paragraphs, etc.].
Tone: [voice, style].

[EXAMPLES BLOCK - optional]
Good example: [show what you want]
Bad example: [show what to avoid]
```

### Example: The Cold Email Template

Instead of re-explaining every time, create once:

```
---COLD EMAIL TEMPLATE---

Context: I sell [product/service]. I'm reaching out to [type of person] 
who might benefit because [reason they'd care].

Target: [Name], [Title] at [Company].
Their likely pain: [what problem they probably have].
Hook: [why they should read this—recent event, mutual connection, insight].

Task: Write a cold email that's under 100 words, sounds human (not salesy), 
and ends with a soft ask (not "let's get on a call tomorrow").

Tone: Professional but warm. Like a smart peer, not a desperate vendor.
---
```

Now every cold email is: copy template, fill blanks, paste.

Your "prompt" becomes:
> [paste filled template]

Same quality every time. 30 seconds instead of 5 minutes.

---

## The Template Library

Build templates for everything you do repeatedly:

### Professional Templates

**Meeting Summary:**
```
I just had a meeting about [topic] with [attendees].
Key decisions made: [list]
Action items mentioned: [list]
Create a clean summary with: decisions, action items with owners, and open questions.
Format for sharing via email.
```

**Performance Review:**
```
Write feedback for [employee name] who is a [role].
Strengths I've observed: [list]
Areas for growth: [list]
Specific examples: [list incidents]
Tone: Direct but supportive. Growth-minded.
Format: 3 paragraphs—what they're doing well, where to focus, overall assessment.
```

**Status Update:**
```
Create a weekly status update for my manager.
Project: [name]
Progress this week: [what got done]
Blockers: [what's stuck]
Next week focus: [priorities]
Keep it under 200 words, use bullets, lead with most important info.
```

### Personal Templates

**Decision Framework:**
```
I need to decide: [decision]
Options: [list them]
What I care about: [values, priorities]
Time horizon: [when do I need to decide]

Walk me through this decision using these frameworks:
1. Pro/con analysis
2. Regret minimization (what will I regret more in 10 years?)
3. Reversibility (how hard is it to undo each option?)
Then give me your recommendation.
```

**Learning Plan:**
```
I want to learn [skill/topic].
My current level: [beginner/intermediate/advanced]
Time available: [hours per week]
Learning style preference: [videos/reading/projects/etc.]
Goal: [what I want to be able to do]

Create a 4-week learning plan with:
- Week by week breakdown
- Specific resources (be specific—name courses, books, channels)
- Milestones to check progress
- One project to build at the end
```

**Difficult Conversation Prep:**
```
I need to have a tough conversation with [person].
Topic: [what it's about]
My goal: [what I want to achieve]
Their likely perspective: [what they probably think/feel]
Risk: [what could go wrong]

Help me:
1. Plan my opening line
2. Anticipate their responses and plan my replies
3. Know when to push and when to back off
4. End the conversation productively
```

---

## System Prompts: The Invisible Instructor

Templates handle structure. System prompts handle behavior.

A system prompt is instructions that persist across an entire conversation. You set them once at the start; the AI follows them for every response.

### How to Use System Prompts

At the start of a conversation:

> "For this entire conversation, follow these rules:
> 
> 1. You are a [role] with expertise in [domain]
> 2. Always ask clarifying questions before giving advice
> 3. Keep responses under 200 words unless I ask for more
> 4. When you're uncertain, say so—don't guess
> 5. End each response with a question to keep the conversation productive"

Now every response follows these rules without you re-stating them.

### Example System Prompts

**The Editor:**
> "You are a senior editor at a major publication. Your job is to make my writing clearer and punchier without changing my voice. For each piece I share:
> - First, identify the strongest line
> - Then, identify the weakest section
> - Finally, suggest 2-3 specific edits
> Never rewrite the whole thing—just point me in the right direction."

**The Devil's Advocate:**
> "For this conversation, your job is to argue against whatever position I take. I need to stress-test my thinking. Be rigorous but not hostile. Find the weakest points in my reasoning. If I have a good counter-argument, acknowledge it, but then find the next weakness."

**The Teacher:**
> "You are tutoring me in [subject]. Start at my level (I know [X] but not [Y]). Use the Socratic method—ask me questions instead of giving answers directly. When I get stuck, give hints before giving solutions. After each concept, check my understanding with a quick question."

**The Consultant:**
> "You are a management consultant with 20 years of experience. I'm going to describe business problems. For each one:
> - Ask 2-3 diagnostic questions first
> - Then give me a structured framework to think about it
> - Finally, suggest specific next steps
> Be direct. I'm paying you for candor, not comfort."

---

## Workflows: Chaining Prompts Together

Single prompts solve single problems. Workflows solve multi-step problems.

A workflow is a sequence of prompts, where each step's output feeds into the next step's input.

### Example Workflow: Research to Article

**Step 1: Research**
> "I need to write about [topic]. Give me:
> - The 5 most important things someone should know
> - The 3 most common misconceptions
> - 2-3 interesting angles that aren't obvious"

*Save the output*

**Step 2: Outline**
> "Based on this research [paste Step 1 output], create an outline for a 1500-word article. Structure: hook → context → 3 main points → actionable takeaway."

*Save the output*

**Step 3: Draft**
> "Using this outline [paste Step 2 output], write the full article. Tone: [specify]. Audience: [specify]."

*Save the output*

**Step 4: Edit**
> "Edit this article for clarity and punch [paste Step 3 output]. Cut at least 20%. Make every paragraph earn its place."

Each step is a simple prompt. Together, they produce a polished article.

### Example Workflow: Decision Making

**Step 1: Frame**
> "I need to decide [decision]. Help me frame this properly:
> - What type of decision is this? (reversible/irreversible, high stakes/low stakes)
> - What am I actually deciding between? (clarify the real options)
> - What's my actual goal? (what does success look like)"

**Step 2: Gather**
> "Now that we've framed the decision [paste Step 1], what information do I need to make a good choice? Ask me questions."

*Answer the AI's questions*

**Step 3: Analyze**
> "Given what you now know [paste conversation], analyze each option against my stated goal. Be honest about tradeoffs."

**Step 4: Pressure Test**
> "Before I decide, play devil's advocate on whatever option you're leaning toward. What could go wrong? What am I missing?"

**Step 5: Decide**
> "Based on everything, what's your recommendation? Give me 2-3 sentences—clear and direct."

### Building Your Own Workflows

Ask yourself:

1. What multi-step task do I do regularly?
2. What are the steps I always follow?
3. Which steps could AI handle or accelerate?
4. How do the steps connect (what's the input/output)?

Then build the chain. Document it. Reuse it.

---

## The Meta-Template: Templates That Make Templates

Here's where it gets powerful.

Instead of manually creating templates, have the AI create them for you.

> "I frequently need to [task description]. Create a reusable template I can use every time. The template should:
> - Have clear fill-in-the-blank sections
> - Include example responses
> - Specify the format and constraints
> - Be copy-paste ready
> 
> Make it good enough that I don't need to think—I just fill in the blanks."

The AI produces a template. You save it. You use it forever.

### Example

> "I frequently need to summarize long documents for my team. Create a template I can use for this."

AI produces:

```
---DOCUMENT SUMMARY TEMPLATE---

Document: [paste document or describe it]
Length: [how long is the original]
Audience: [who will read the summary]
Purpose: [why do they need it summarized—decision, reference, update?]

Requested summary:
- Length: [1 paragraph / 1 page / 3 bullets]
- Must include: [any specific info that must appear]
- Exclude: [anything to leave out]
- Format: [bullets, prose, structured sections]

Tone: [formal/casual, technical/accessible]

---
```

Now summarizing documents is fill-in-the-blanks, not starting from scratch.

---

## Organizing Your Second Brain

Creating templates and workflows is only useful if you can find them later.

### Simple Organization

**Option 1: Notes App**

Create a folder called "AI Templates" with sub-folders:
- Work
- Personal
- Research
- Writing

Store each template as a note. Title clearly. Update when you improve them.

**Option 2: Spreadsheet**

Columns:
- Template Name
- Category
- Description (when to use it)
- The Template (full text)
- Last Updated

Filter by category. Search by keyword. Export/share easily.

**Option 3: Document**

One long document with a table of contents. Sections by use case. Ctrl+F to find what you need.

### The Review Ritual

Templates go stale. Once a month:

1. Review which templates you actually used
2. Delete the ones you didn't
3. Update the ones that needed tweaking
4. Create templates for any tasks you found yourself repeating

Your second brain should evolve with your needs.

---

## The Compound Effect

Here's why this matters beyond convenience:

Every template you create is a distillation of your thinking. You figure out what works once, capture it, and never have to figure it out again.

**Week 1:** You spend an hour crafting the perfect cold email prompt. One great email.

**Week 10:** You've used that template 40 times. 40 great emails in total time spent: that first hour plus maybe 30 minutes total for filling blanks.

**Week 50:** 200 great emails. Same hour of initial investment. You're now 100x more efficient at this task than someone starting fresh every time.

This compounds across every repeated task. Templates, workflows, systems—they're all leverage. The work you do once pays dividends forever.

Most people think using AI well means writing clever prompts. That's Level 2.

Level 3 is building systems that write the prompts for you.

---

## Summary

- **Don't start from scratch.** Create templates with fill-in-the-blank structures for repeated tasks.

- **Use system prompts** to set persistent behavior across entire conversations.

- **Build workflows** that chain prompts together for multi-step tasks.

- **Use meta-templates** to have the AI create templates for you.

- **Organize and maintain** your template library so it grows more valuable over time.

The next chapter takes this further: instead of just getting answers and documents, you'll learn to make the AI build actual tools—code that runs, automations that work, things you can use without knowing how to program.
