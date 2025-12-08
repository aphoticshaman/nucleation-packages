# VeilPath Tarot - Ethical Boundaries & Professional Standards

## Document Authority
**Created by:** Multi-disciplinary mental health team perspective
**Roles Represented:** Clinical Psychologist (PhD), Psychiatrist (MD), LCSW, LPC, LMFT, Peer Support Specialist
**Purpose:** Ensure app provides benefit while maintaining ethical boundaries and user safety

---

## CORE PRINCIPLES

### 1. We Are NOT Therapy
**Boundary:** VeilPath is a **self-help tool** that complements professional treatment.
**NOT:** A replacement for therapy, medical treatment, or crisis intervention.

**Implementation:**
- Prominent disclaimers on first launch
- Never use language like "cures," "treats," "diagnoses"
- Use: "supports," "complements," "enhances self-awareness"
- Regular reminders to consult mental health professionals

### 2. Crisis Protocol - MANDATORY
**Boundary:** If user indicates suicidal ideation, self-harm, or crisis, IMMEDIATE resources.

**Triggers** (Auto-detect in journal/AI analysis):
- "I want to die"
- "I'm going to hurt myself"
- "I can't go on"
- "There's no point"
- "Suicide"

**Response Protocol:**
```
IMMEDIATE FULL-SCREEN OVERLAY:
━━━━━━━━━━━━━━━━━━━━━━━━━━
🚨 WE'RE CONCERNED ABOUT YOU 🚨

If you're in crisis or having thoughts of self-harm:

📞 National Suicide Prevention Lifeline
   988 (US) - 24/7 support

📱 Crisis Text Line
   Text HOME to 741741

🌐 International Resources
   findahelpline.com

This app cannot provide emergency support.
Please reach out to these professionals who can help.

[I'm Safe - Continue] [Call 988 Now]
━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Implementation:**
- AI analysis flags crisis language
- Human cannot override (safety > UX)
- Log crisis events (locally, privacy preserved)
- Recommend professional help in next session

### 3. Age Restrictions
**Boundary:** App is for ages 17+ (Apple rating: 17+)

**Reasoning:**
- Mental health content requires maturity
- Tarot imagery can be intense
- Self-directed therapy work isn't appropriate for children
- Parental supervision not feasible

**Implementation:**
- Age gate on first launch
- Terms of Service specify 17+
- Content assumes adult developmental stage

### 4. Data Privacy - Therapeutic Privilege
**Boundary:** User journal data is SACRED. Treat like HIPAA (even though we're not covered entity).

**Principles:**
- **Local-first:** Journals stored on-device by default
- **Opt-in Cloud:** Cloud sync requires explicit consent
- **Encryption:** End-to-end if cloud sync enabled
- **No selling:** NEVER sell or share user data
- **AI Analysis:** Ephemeral (processed, not stored)
- **Export Control:** User can export/delete ALL data anytime

**Implementation:**
```json
{
  "privacy_settings": {
    "local_only": true (default),
    "cloud_sync": false (opt-in),
    "ai_analysis": false (opt-in),
    "analytics": "minimal" (aggregated only),
    "third_party": "none"
  }
}
```

### 5. No Diagnosis - Ever
**Boundary:** App NEVER diagnoses mental health conditions.

**Prohibited:**
- ❌ "You have depression"
- ❌ "This indicates PTSD"
- ❌ "You may be bipolar"

**Allowed:**
- ✅ "You're experiencing symptoms of low mood"
- ✅ "This pattern is common in anxiety"
- ✅ "Consider discussing this with a therapist"

**Implementation:**
- Content review by licensed professional
- AI prompts explicitly avoid diagnostic language
- Disclaimers in any symptom-related content

### 6. Cultural Sensitivity & Inclusivity
**Boundary:** Tarot + mental health crosses many cultures. Respect all.

**Considerations:**
- **Western therapy bias:** CBT/DBT are Western frameworks
  - Acknowledge: "These are evidence-based Western approaches. We respect that healing looks different across cultures."

- **Spiritual beliefs:** Some see tarot as spiritual, others as psychological
  - Support both: "Use this tool in whatever way aligns with your beliefs."

- **Language:** Avoid ableist language ("crazy," "insane")

- **Representation:** Card imagery should include diverse bodies, genders, races

- **Accessibility:** VoiceOver support for blind users, dyslexia-friendly fonts

### 7. Scope of Practice - What We Can/Can't Address
**IN SCOPE (Self-help support):**
- ✅ Mild-moderate anxiety
- ✅ Mild-moderate depression
- ✅ Stress management
- ✅ Cognitive distortions
- ✅ Journaling for self-reflection
- ✅ Mindfulness practice
- ✅ Emotional regulation skills

**OUT OF SCOPE (Requires professional):**
- ❌ Severe depression (PHQ-9 > 15)
- ❌ Acute suicidal ideation
- ❌ Psychosis
- ❌ Bipolar disorder (manic episodes)
- ❌ Eating disorders
- ❌ Substance abuse
- ❌ PTSD flashbacks
- ❌ Personality disorders

**Screening Implementation:**
- Onboarding asks: "Are you currently in therapy?" (optional)
- If severe symptoms detected → recommend professional help
- Regular reminders: "This app supports, but doesn't replace therapy"

### 8. AI Ethics - Claude Integration
**Boundary:** AI analysis is a tool, not a therapist.

**Principles:**
- **Transparency:** User knows when AI is analyzing
- **Opt-in:** AI features off by default
- **Explainability:** Show AI reasoning ("I noticed you used 'always' and 'never' - this might be all-or-nothing thinking")
- **Human-in-loop:** User can disagree with AI suggestions
- **Bias awareness:** AI trained on Western therapy - may not fit all

**Prohibited AI Uses:**
- ❌ Predict suicide risk (too high stakes)
- ❌ Diagnose conditions
- ❌ Give medical advice
- ❌ Store journal entries on Anthropic servers

**Allowed AI Uses:**
- ✅ Identify cognitive distortions (with explanation)
- ✅ Suggest reframing questions
- ✅ Recommend DBT skills based on patterns
- ✅ Generate personalized journal prompts

### 9. Gamification Ethics
**Boundary:** Gamification should MOTIVATE, not MANIPULATE.

**Ethical Gamification:**
- ✅ Celebrate effort (streaks for showing up)
- ✅ Unlock features (progression feels rewarding)
- ✅ Positive reinforcement (achievements for growth)

**Unethical Gamification:**
- ❌ Shame for missing days
- ❌ Punishment mechanics
- ❌ Social comparison (leaderboards with others)
- ❌ Addictive dark patterns (infinite scroll, variable rewards)

**Implementation:**
- Streaks: Can be "frozen" if user needs a break (mental health days)
- Achievements: Never shame-based ("Finally logged in after 30 days")
- XP: Earned for depth, not just frequency (quality > quantity)

### 10. Disclaimers - Required Everywhere
**Placement:** First launch, Settings, Terms of Service, Help section

**Standard Disclaimer:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANT: PLEASE READ

VeilPath is a self-help tool designed to COMPLEMENT
professional mental health care, not replace it.

• We do NOT provide therapy, counseling, or medical advice
• We do NOT diagnose mental health conditions
• We are NOT a substitute for emergency services

If you're in crisis:
• Call 988 (Suicide & Crisis Lifeline)
• Text HOME to 741741 (Crisis Text Line)
• Call 911 for emergencies

If you have a mental health condition, please
consult with a licensed mental health professional
about whether this app is appropriate for you.

By continuing, you acknowledge this limitation.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## CONTENT REVIEW CHECKLIST

### Before ANY Content Goes Live:
- [ ] Does it claim to diagnose? (RED FLAG)
- [ ] Does it claim to treat/cure? (RED FLAG)
- [ ] Does it use medical language inappropriately? (YELLOW FLAG)
- [ ] Is it culturally sensitive? (REQUIRED)
- [ ] Is it evidence-based (cited sources)? (REQUIRED)
- [ ] Does it include appropriate disclaimers? (REQUIRED)
- [ ] Would a licensed therapist approve? (REQUIRED)

### Red Flags to NEVER Ship:
- "This card means you have [diagnosis]"
- "Tarot can cure depression"
- "You don't need therapy if you use this app"
- "Skip your meds and journal instead"
- Any magical thinking about mental health

---

## REFERRAL RESOURCES

### Built-in Resource Directory
**Categories:**
1. **Crisis Lines** (suicide, abuse, assault)
2. **Therapy Directories** (Psychology Today, SAMHSA, Open Path)
3. **Support Groups** (NAMI, DBSA, local groups)
4. **Self-Help Books** (evidence-based recommendations)
5. **Apps** (Headspace, Calm for meditation)
6. **Hotlines** (Veterans Crisis Line, Trevor Project for LGBTQ+)

**Update Frequency:** Quarterly review of links/phone numbers

---

## TERMS OF SERVICE - Key Points

### User Responsibilities:
- Understand this is NOT therapy
- Consult professionals for serious concerns
- Keep login credentials secure
- Don't share crisis-level content expecting emergency response

### Our Responsibilities:
- Maintain user privacy
- Provide evidence-based content
- Update resources regularly
- Respond to support inquiries within 48 hours
- Remove harmful content if reported

### Liability Limitations:
- "To the maximum extent permitted by law, VeilPath is not liable for any harm arising from app use or non-use."
- "User assumes all risk in using self-help tools."

**Note:** Terms reviewed by lawyer before launch (budget: $1500).

---

## PROFESSIONAL CONSULTATION

### Required Reviews:
1. **Clinical Psychologist (PhD):** Review all CBT content
   Cost: $1000 | Timeline: 2 weeks

2. **DBT-Certified Therapist:** Review DBT skills content
   Cost: $800 | Timeline: 1 week

3. **Lawyer (Tech/Health Law):** Review Terms, Privacy Policy
   Cost: $1500 | Timeline: 1 week

4. **Accessibility Consultant:** Review VoiceOver, WCAG compliance
   Cost: $500 | Timeline: 1 week

**Total Investment in Professional Review:** $3800
**ROI:** Protects users, ensures Apple approval, builds trust

---

## MONITORING & ACCOUNTABILITY

### Ongoing Safety Measures:
- **Crash Reporting:** Monitor for any crisis-related crashes
- **User Feedback:** In-app "Report Concern" button
- **Content Audits:** Quarterly review of all therapeutic content
- **AI Monitoring:** Monthly review of Claude API logs (detect inappropriate suggestions)
- **Therapist Advisory Board:** 3 licensed professionals (volunteer or small stipend)

### Incident Response Plan:
If user reports:
1. **Harmful content:** Remove within 24 hours, review similar content
2. **Privacy breach:** Immediate investigation, user notification, fix
3. **AI gave bad advice:** Log, analyze, update prompts, notify user
4. **Accessibility issue:** Prioritize fix, compensate affected users if applicable

---

## ETHICAL EVOLUTION

Mental health understanding evolves. So must we.

**Commitments:**
- Review literature: Stay current with CBT/DBT research
- User feedback: Listen to what helps/harms
- Diverse perspectives: Include BIPOC, LGBTQ+, disabled voices in design
- Humility: Acknowledge what we don't know

**Annual Review:**
- Are we still within ethical boundaries?
- Has research invalidated any content?
- Are there emerging populations we're not serving?
- Should we sunset features that aren't helping?

---

## CONCLUSION

**The North Star:** First, do no harm.

When in doubt:
1. Would a licensed therapist approve this?
2. Would I want my loved one using this if they were struggling?
3. Does this empower the user or create dependence?
4. Are we being transparent about limitations?

If the answer to any is "no" → Don't ship it.

**Mental health is sacred ground. We walk it with humility, care, and responsibility.**

---

**Document Version:** 1.0
**Next Review:** Before App Store submission
**Owner:** Clinical Ethics Board (to be established)
