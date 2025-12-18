# Mega-Prompt Plan for Enterprise-Grade LatticeForge.ai Overhaul

## Goals
- Deliver a modern, enterprise-ready UI/UX across 40+ screens with deterministic intelligence assessments.
- Ensure clarity, accessibility, and consistency for regulated buyers (security, compliance, procurement, exec stakeholders).
- Provide prompts and inputs that let the assistant operate autonomously while preserving verifiability and traceability.

## Inputs to Collect Before Prompting
1. **Business & Branding**: value prop, target personas, tone/voice, visual identity, brand constraints, legal/disclaimer language.
2. **Product & Content Inventory**: sitemap, screen list (40+), data flows, current copy, data schemas, APIs, integrations (SSO, SIEM, ticketing, GRC), and domain-specific glossaries.
3. **User Journeys & Outcomes**: onboarding paths (self-serve vs. sales-assisted), daily workflows, escalation flows, and measurable success metrics.
4. **UX Guardrails**: accessibility requirements (WCAG level), responsive breakpoints, motion guidelines, localization needs, privacy and consent flows, error states, empty/loading/skeleton treatments.
5. **Security & Compliance**: attestations, policies, evidence sources, audit requirements, data residency, encryption/SOAR/SIEM integrations, redaction rules.
6. **Engineering Constraints**: tech stack (framework versions, design system tokens, component library), performance budgets, testing expectations, analytics/telemetry hooks, feature flag strategy, CI/CD + lint/format/husky steps.
7. **AI/Intelligence Quality**: model capabilities, deterministic scoring criteria, source-of-truth data, explainability format, confidence thresholds, citation requirements, red-team scenarios, non-goals.
8. **Benchmarks & References**: competitor examples, enterprise design systems to emulate, SLA/SLO definitions, and examples of “best-in-class” interactions.
9. **Environment & Access**: env vars, API keys, mock/stub plans, seeded datasets, and sandbox URLs for visual QA.

## Research & Analysis Tasks (Pre-Prompt)
- Crawl and map the current IA: pages, components, routes, and reusable primitives.
- Audit UX quality: accessibility, empty/loading/error states, information density, navigation clarity, form ergonomics, and enterprise trust signals.
- Audit performance: bundle size hotspots, code splitting, image optimization, LCP/CLS targets, caching strategy.
- Audit reliability: test coverage gaps, typesafety hotspots, network/stream handling, error boundaries.
- Audit content: outdated copy, inconsistent terminology, missing disclaimers, and evidence-backed claims.
- Audit integrations: SSO/OAuth flows, webhook robustness, observability hooks, SAML/JWT handling, and data lineage for intelligence outputs.

## Mega-Prompt Skeleton
```text
You are an enterprise-grade product engineer, designer, and analyst for LatticeForge.ai. You must ship a modern, regulated-market-ready overhaul across 40+ screens.

Context:
- Business: {value proposition, personas, brand tone, visual direction}
- Product inventory: {sitemap, screen list, flows, component library, design tokens}
- Data & AI: {models, scoring rules, evidence sources, confidence thresholds, citation format}
- Constraints: {accessibility level, performance budgets, security/compliance requirements, localization}
- Engineering: {framework versions, lint/format/test commands, feature flags, analytics/telemetry hooks}
- Benchmarks: {competitors, design systems, SLAs/SLOs}
- Assets/Links: {design mocks, API docs, sample payloads, seeded data, sandbox URLs}

Directives:
1) Propose a phased plan (audit → IA → UX → build → QA) covering all 40+ screens with owners, timelines, and exit criteria.
2) For each screen, define: purpose, target persona, key actions, data dependencies, empty/loading/error states, success/failure copy, and security/compliance checks.
3) Generate component-level tasks with acceptance criteria, test cases (unit/e2e/a11y), and observability hooks.
4) Produce deterministic intelligence patterns: scoring algorithms, evidence citations, red-team cases, and fallback behaviors.
5) Output implementation-ready specs (wireframe descriptions, API contracts, design tokens) plus Jira-ready task breakdowns.
6) Include review checklists for accessibility, performance, security, and UX consistency.
7) Return results in Markdown with tables and IDs to track execution.
```

## Execution Guidance
- **Phase sequencing**: 1) Collect inputs, 2) Audit and map IA, 3) Define target UX patterns, 4) Write implementation specs, 5) Deliver code/tasks.
- **Determinism**: Require evidence-linked outputs with confidence thresholds and rejection criteria; avoid unverifiable claims.
- **Traceability**: Tag every task/spec with IDs and link to screens/components for cross-checking.
- **Review rigor**: Enforce checklists (a11y/perf/security/UX) per component and per flow; mandate tests and telemetry hooks.
- **Collaboration**: Request missing assets/decisions explicitly; prefer structured tables for gaps and risks.

## Quick Intake Checklist (fill before prompting)
- [ ] Personas & brand voice defined
- [ ] Full screen/flow inventory documented
- [ ] API contracts & data schemas available
- [ ] Compliance/security requirements captured
- [ ] Design tokens & component library confirmed
- [ ] Performance, accessibility, and reliability targets agreed
- [ ] Analytics/telemetry events and destinations listed
- [ ] Environments and credentials ready (or mocks defined)

## Deliverable Expectations
- Blueprint covering all screens with IA map, wireframe notes, and component reuse plan.
- End-to-end task matrix with owners, estimates, and acceptance criteria.
- Test strategy (unit, integration, e2e, a11y) aligned to CI commands.
- Post-ship validation plan: observability, dashboards, and runbooks for regressions.

Use this plan as the prompt-engineering baseline; populate the placeholders with project-specific details, then execute the mega-prompt to generate workstreams and implementation-ready specs.

## How to Use This Today
1) **Collect inputs quickly:** Start with the “Quick Intake Checklist.” If any box is unchecked, gather that item (or document what’s missing) before running the mega-prompt.
2) **Fill the skeleton:** Replace the placeholders in the “Mega-Prompt Skeleton” with concrete details (personas, tokens, routes, APIs, a11y/perf/security targets). Keep links handy for designs and API docs.
3) **Run the mega-prompt:** Paste the filled skeleton into your assistant of choice and request a phased plan plus screen-by-screen specs with IDs and acceptance criteria.
4) **Triage output into tasks:** Convert the returned plan into tickets grouped by phases (audit → IA → UX → build → QA). Ensure every task has owners, estimates, and explicit a11y/perf/security checks.
5) **Instrument quality gates:** Before building, lock in lint/format/test/telemetry commands and add any missing Husky/CI hooks so reviewers can verify determinism and compliance.
6) **Ship in slices:** Deliver by screen/flow slices with observable checkpoints (metrics, logs, dashboards). Use the IDs from the prompt output to trace coverage across all 40+ screens.
