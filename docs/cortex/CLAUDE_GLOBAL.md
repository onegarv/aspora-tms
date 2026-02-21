# CLAUDE.md — Aspora Cortex (Compressed)

<philosophy>
Velocity = Speed x Direction. Prove by building, not debating. Production-grade or it doesn't merge.
Rhythm: QUESTION → PRIMITIVE (< 2hrs, real data) → DEMO → SHARPEN → SCALE → SKILL-IFY
Steps 2-4 repeat 3-5x before step 5. Bias toward shipping.
</philosophy>

<before_any_task>
1. Check for DECISIONS.md and GUARDRAILS.md in the project root — read them if they exist. These contain prior decisions (don't re-decide) and anti-patterns (don't repeat mistakes).
2. Analyze the codebase — infer patterns, don't ask what you can read.
3. Brief plan for non-trivial work (3-5 sentences).
4. Match existing patterns exactly.
</before_any_task>

<investigate_before_answering>
Never speculate about code you have not opened. If the user references a specific file, read the file before answering. Make sure to investigate and read relevant files BEFORE answering questions about the codebase. If data is insufficient to draw conclusions, say so rather than speculating. "I don't know" is a valid and respected answer.
</investigate_before_answering>

<principles>
- SOLID across all languages. DRY (rule of three). YAGNI.
- Clean Architecture: core/domain/ has zero framework deps. Dependencies point inward.
- Composition over inheritance. Explicit over implicit. Immutability by default.
- Config-driven behavior (YAML/config, not hardcoded if-else).
- Functions: < 20 lines, < 3 params, pure when possible.
- No placeholder code. Comments explain WHY only.
</principles>

<language_patterns>
- Java: DDD, rich domain models, record value objects, constructor injection, Micrometer, JUnit 5
- Go: cmd/internal/pkg, accept interfaces return structs, error wrapping with %w, table-driven tests, bounded concurrency
</language_patterns>

<observability>
- Four Golden Signals: Latency, Traffic, Errors, Saturation on every service.
- Zero Logic Mutation: instrumentation is purely additive. Changing behavior, control flow, or exception types when adding metrics breaks production systems — see GUARDRAILS.md for RCA examples.
- Anomaly-based alerting over static thresholds. Cardinality control on all labels.
</observability>

<testing>
- Behavior not implementation. AAA pattern. Table-driven in Go. DRY test utilities.
- Test every change. Boundaries: empty, max, concurrent, error paths.
</testing>

<guardrails>
- Idempotency — don't rewrite my SQL/commands unless asked.
- Scope honesty — state uncertainty explicitly rather than guessing.
- No fabricated mocks unless asked. Use real data from actual sources.
</guardrails>

<documentation>
Minto Pyramid: answer first, details after. Use CORRECT / INCORRECT examples.
</documentation>

<git>
Conventional commits (feat: fix: refactor: docs: test: chore:). Atomic. Every commit compiles + passes tests.
</git>

<skill_design>
- Detect first: plugin.yaml, skills/ folder → skills-first approach.
- In agentic projects, the deliverable is a SKILL, not Python/Java/Go code.
- Load AGENTIC_SKILLS.md when starting skill work.
- Load SKILL_DESIGN_PRINCIPLES.md only during deep design work.
- Never load CORTEX.md or SKILL_DESIGN_PRINCIPLES.md as a system prompt — they are reference files, not instructions.
</skill_design>

<context_management>
- Context window is a shared resource — be concise, load references lazily.
- Don't explain what Claude already knows. Only add context Claude doesn't have.
- Your context window will be automatically compacted as it approaches its limit. Do not stop tasks early due to token budget concerns. Save progress to files before context refreshes. Be persistent and complete tasks fully.
- When approaching context limits, summarize intermediate work rather than keeping raw details.
</context_management>

<claude_4x_awareness>
- Claude Opus 4.6 is more responsive to instructions than previous models. Avoid over-prompting — tools and patterns that previously needed encouragement now trigger naturally.
- Avoid excessive upfront exploration. Choose an approach and commit to it. Course-correct only if you encounter new information that contradicts your reasoning.
- Keep solutions minimal. Don't add features, abstractions, or flexibility beyond what was requested. A bug fix doesn't need surrounding code cleaned up.
- Use parallel tool calls when actions are independent. Sequential when dependent.
</claude_4x_awareness>

<layered_config>
| File | When to Load | Purpose |
|------|--------------|---------|
| This CLAUDE.md | Always | Compressed core principles |
| DECISIONS.md | Before any task (if exists) | Prior decisions — don't re-decide |
| GUARDRAILS.md | Before any task (if exists) | Anti-patterns — don't repeat mistakes |
| CORTEX.md | Deep work, architecture | Full engineering system (reference only) |
| AGENTIC_SKILLS.md | Starting skill work | Quick ref: SAGE, structure, checklists |
| SKILL_DESIGN_PRINCIPLES.md | Deep skill design | DDD→Skill, SOLID→SKILL, Lazy Loading |
| DOMAIN_CONTEXT.md | Domain-specific work | Fintech, payments, ECM terminology |
</layered_config>

<decision_recording>
During your work, persist learnings to the project's Decision Record Room:
- Architecture choice made → append to DECISIONS.md (Chose/Over/Why/Constraint format, ~50 tokens)
- Mistake or anti-pattern caught → append to GUARDRAILS.md (Mistake/Impact/Rule/Detection format)
- Before a long session ends → review if any undocumented decisions need recording
- Use /save-decisions for a structured flush of session learnings
- Max 20 decisions per project. Constraints should be binary with reasoning.
</decision_recording>

<prompt_format>
CONTEXT: ... TASK: ... CONSTRAINTS: ... DONE WHEN: ...
Ask for missing pieces before starting. Full brain file: see CORTEX.md in project root.
</prompt_format>
