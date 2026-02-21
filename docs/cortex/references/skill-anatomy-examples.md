# Real-World Skill Anatomy Analysis

> **When to load:** When studying examples of well-designed skills or reviewing your own skill against known patterns.
> **Source:** Extracted from SKILL_DESIGN_PRINCIPLES.md Part 9.

## The docx Skill — Mechanical Precision

**Strength**: Outstanding mechanical rules — every XML rule, API gotcha, validation step.
**Weakness**: No reasoning framework — WHY is missing.
**Enhancement**: Add Domain Model + Reasoning sections for counterintuitive rules.

**Lesson:** Mechanical skills need WHY for their non-obvious rules. Without reasoning, the LLM can't generalize when it encounters an edge case the rules don't cover.

## The product-critique Skill — Judgment-Heavy

**Strength**: Excellent Layers 1+2. "You are not a template auditor. You are a senior product leader."
**Weakness**: Approaches 500-line limit.
**This is closest to ideal**: Domain model, philosophy, workflow, calibration guide.

**Lesson:** Judgment-heavy skills benefit most from the WHAT and WHY layers. The HOW layer can be lighter because the LLM reasons well when it understands the domain model and the optimization target.

## The frontend-design Skill — Creative Judgment

**Strength**: Declares intent masterfully. Constraints target mediocrity, not specific implementations.
**This is how you write creative skills**: Inspire, bound mediocrity, trust the system.

**Lesson:** For creative work, declare what "good" looks like and what to avoid, then trust the LLM. Over-specifying creative output produces generic results — the "AI slop" aesthetic.

## Mapping Skills to Instruction Style

| Skill Type | WHAT Layer | WHY Layer | HOW Layer | Instruction Style |
|------------|-----------|-----------|-----------|-------------------|
| Mechanical (docx, config) | Light | Medium | Heavy, precise | Low degrees of freedom |
| Judgment (product-critique, code-review) | Heavy | Heavy | Light, intent-based | High degrees of freedom |
| Creative (frontend-design, writing) | Medium | Heavy | Light, inspirational | Highest degrees of freedom |
| Operational (ECM, monitoring) | Heavy | Medium | Medium, phased | Mixed — phases with gates |
