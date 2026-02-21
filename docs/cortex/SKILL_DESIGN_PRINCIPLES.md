# From SOLID to SKILL: Domain-Driven Design for the Agentic Age

## A First-Principles Framework for Building LLM Skills

> **When to read this file:** Deep skill design work — creating new skills, debugging skill behavior, architecting multi-agent systems. For quick reference, see AGENTIC_SKILLS.md (SAGE principles).
>
> **Last updated:** 2026-02-16

---

## Table of Contents

1. [Why Traditional Principles Need Translation, Not Abandonment](#part-1)
2. [Domain-Driven Design → Skill-Driven Design](#part-2)
3. [SOLID Principles → SKILL Principles](#part-3)
4. [Clean Code Principles → Clean Skill Principles](#part-4)
5. [Lazy Loading → Lazy Context Principles](#part-5)
6. [First-Principles Thinking for LLM Skills](#part-6)
7. [References (Loaded on Demand)](#part-7) — Template, Anti-Patterns, Examples, Anthropic Standards

---

<a id="part-1"></a>
## Part 1: Why Traditional Principles Need Translation, Not Abandonment

Traditional software engineering principles (SOLID, DRY, DDD, Clean Architecture) were designed for a world where:
- **The consumer is a CPU** — it executes instructions literally
- **State is explicit** — you declare variables, manage memory
- **Composition is syntactic** — functions call functions, classes inherit
- **Testing is deterministic** — same input → same output

Agentic skills operate in a fundamentally different world:
- **The consumer is a reasoning system** — it *interprets* instructions with judgment
- **State is contextual** — the LLM's context window IS the working memory
- **Composition is semantic** — skills combine through *meaning*, not syntax
- **Testing is probabilistic** — same input → *similar* output (with variance)

This means we can't just copy-paste SOLID. We need to understand *why* each principle exists, extract the underlying wisdom, and re-express it for the new medium.

**The fundamental shift**: In traditional code, you optimize for the machine's execution. In skill design, you optimize for the LLM's *comprehension and reasoning*.

---

<a id="part-2"></a>
## Part 2: Domain-Driven Design → Skill-Driven Design

### DDD Core Concepts and Their Skill Equivalents

#### 1. Ubiquitous Language → Skill Vocabulary

**DDD says**: The entire team shares a single vocabulary. The code uses the same terms the business uses.

**In skill design**: The skill must speak the language of the *domain it serves*, not the language of LLM engineering.

```
BAD:  "Place configurable numeric values in designated input cells"
GOOD: "Place ALL assumptions (growth rates, margins, multiples) in
       separate assumption cells. Use cell references instead of
       hardcoded values in formulas."
```

The good version uses the domain's actual vocabulary. The LLM has been trained on millions of financial documents and *knows* what "growth rates" and "margins" mean.

**Principle: SPEAK THE DOMAIN**
> Use the domain's actual terminology. The LLM already knows the domain — your job is to activate that knowledge, not replace it with generic abstractions.

---

#### 2. Bounded Context → Skill Boundary

**DDD says**: A bounded context is a boundary within which a particular domain model is defined and applicable.

**In skill design**: Each skill defines a bounded context — a clear domain boundary with explicit entry/exit conditions.

```
"Do NOT use for PDFs, spreadsheets, Google Docs, or general
coding tasks unrelated to document generation."
```

**Anti-pattern**: A skill called "document-processor" that handles DOCX, PDF, PPTX, and HTML. Each format has different internal models, different tools, different failure modes.

**Principle: DRAW THE BOUNDARY**
> Define what's IN your skill's domain and what's OUT — explicitly. Ambiguous boundaries cause the worst failures: the LLM uses the wrong skill confidently.

---

#### 3. Entities vs. Value Objects → Stateful Artifacts vs. Transient Outputs

**DDD says**: Entities have identity and lifecycle. Value objects are defined by their attributes.

**In skill design**: Some outputs are **stateful artifacts** (a spreadsheet with formulas) and some are **transient outputs** (a summary, a recommendation).

Stateful artifacts need:
- Preservation of internal structure (formulas, styles, relationships)
- Round-trip safety (can be edited and re-processed)
- Explicit lifecycle management

```
"Warning: If opened with data_only=True and saved, formulas are
replaced with values and permanently lost"
```

**Principle: KNOW YOUR ARTIFACT'S LIFECYCLE**
> If your skill creates something the user will modify later, treat it as an entity with lifecycle concerns.

---

#### 4. Aggregates → Skill Bundles

**DDD says**: An aggregate is a cluster of domain objects treated as a single unit. External objects reference only the aggregate root.

**In skill design**: SKILL.md is the aggregate root. Scripts, references, and assets are internal objects accessed *through* the SKILL.md.

```
skill-name/              ← The Aggregate
├── SKILL.md             ← Aggregate Root (single entry point)
├── scripts/             ← Internal entities
├── references/          ← Internal value objects
└── assets/              ← Internal resources
```

**Principle: ONE ENTRY POINT, GUIDED INTERNALS**
> The LLM enters through SKILL.md and is guided to internal resources by the skill's own logic.

---

#### 5. Domain Events → Trigger Conditions

**DDD says**: Domain events capture something that happened that domain experts care about.

**In skill design**: Trigger conditions are the "events" that activate a skill.

```
"Make sure to use this skill whenever the user mentions dashboards,
data visualization, internal metrics, or wants to display any kind
of company data, even if they don't explicitly ask for a dashboard."
```

LLMs tend to "undertrigger" — they miss opportunities to use skills.

**Principle: OVER-SUBSCRIBE TO TRIGGERS**
> List every conceivable phrase, context, and scenario that should activate your skill.

---

#### 6. Repositories → Skill Registries

Skill registries (like OpenClaw's ClawHub) are repositories that the system queries to find the right skill. The metadata (name + description) is the query interface.

#### 7. Domain Services → Orchestrator Skills

Orchestrator skills coordinate multiple building blocks without owning any single artifact. They delegate to executors, graders, comparators.

---

<a id="part-3"></a>
## Part 3: SOLID Principles → SKILL Principles

### S — Single Responsibility Principle

**SOLID**: A class should have one, and only one, reason to change.

**SKILL equivalent: Single Domain Ownership**

| Reason to Change | Skill Impact | Verdict |
|-----------------|-------------|---------|
| Word document API changes | docx skill updates | ✅ Single responsibility |
| User wants PDF output | docx skill stays unchanged | ✅ Different skill |
| Both DOCX creation AND critique logic | docx skill becomes two things | ❌ Split them |

---

### O — Open/Closed Principle

**SOLID**: Open for extension, closed for modification.

**SKILL equivalent: Extensible by Reference, Stable at Core**

```
cloud-deploy/
├── SKILL.md                    ← Stable core (rarely changes)
└── references/
    ├── aws.md                  ← Extension point
    ├── gcp.md                  ← Extension point
    └── azure.md                ← Added later, core unchanged
```

---

### L — Liskov Substitution Principle

**SOLID**: Subtypes must be substitutable for their base types.

**SKILL equivalent: Skill Contract Fidelity**

If your skill description says it handles X, it must actually handle ALL of X reliably.

---

### I — Interface Segregation Principle

**SOLID**: No client should be forced to depend on interfaces it does not use.

**SKILL equivalent: Progressive Disclosure (Three-Level Loading)**

| Level | What | When Loaded | Size Target |
|-------|------|-------------|-------------|
| **Metadata** | Name + description | Always | ~100 words |
| **Body** | SKILL.md instructions | When triggered | <500 lines |
| **Resources** | Scripts, references | On demand | Unlimited |

Every token loaded into context is a token *not* available for reasoning.

---

### D — Dependency Inversion Principle

**SOLID**: Depend on abstractions, not concretions.

**SKILL equivalent: Depend on Capabilities, Not Implementations**

```
BAD:  "Run `python3 -c 'from docx import Document; ...'`"
GOOD: "Generate a DOCX using the docx skill (read /skills/docx/SKILL.md)"
```

---

<a id="part-4"></a>
## Part 4: Clean Code Principles → Clean Skill Principles

### DRY → DON'T REPEAT CONTEXT

Don't repeat what the LLM already knows. But use **strategic repetition** for safety-critical rules:

```
"## Creating Tables
CRITICAL: Tables need dual widths — set both columnWidths AND cell width.
...
## Critical Rules
- Tables need dual widths — columnWidths array AND cell width must match"
```

**Principle: DRY for facts, repeat for safety-critical rules.**

---

### KISS → KEEP INSTRUCTIONS SIMPLE

```
BAD:  "Apply the inverse of the negation of the user's formatting
       preferences unless overridden by template conventions..."

GOOD: "Match the existing document's formatting. If no existing
       format, use Arial 12pt with 1-inch margins."
```

---

### YAGNI → PRUNE RUTHLESSLY

> "Keep the prompt lean; remove things that aren't pulling their weight."

**The weight test**: If I remove this, does output quality measurably drop? If not, remove it.

---

### Tell, Don't Ask → DECLARE INTENT, DON'T MICROMANAGE

```
BAD:  "Check if the title is longer than 50 characters. If it is,
       truncate to 47 and add '...'..."

GOOD: "Format the title section: titles should be concise, centered,
       and visually distinct. The title should be the most prominent
       element on the page."
```

**But**: For *deterministic, mechanical* operations, precision IS correct:
```
"Comment markers are direct children of w:p, never inside w:r."
```

**Principle: Declare intent for judgment work. Be precise for mechanical work.**

---

### Separation of Concerns → SEPARATE WHAT FROM HOW

```
1. PHILOSOPHY / MENTAL MODEL     ← What and Why (SKILL.md)
2. WORKFLOW / DECISION LOGIC     ← When to do what (SKILL.md)
3. TECHNICAL REFERENCE           ← How to do it (references/)
4. SCRIPTS / TOOLS               ← Deterministic execution (scripts/)
```

---

### Law of Demeter → MINIMIZE CHAIN DEPTH

```
BAD:  Skill → reads skill → reads reference → reads script
GOOD: Skill → reads its reference + delegates to another skill directly
```

Deep chains fill the context window and dilute earlier instructions.

---

### Composition Over Inheritance → COMPOSE SKILLS, DON'T EXTEND THEM

```
BAD:  "This skill extends the base-document-skill..."

GOOD: "For DOCX output, use the docx skill. For styling, use brand-guidelines."
```

Skills compose through *delegation*, not inheritance.

---

<a id="part-5"></a>
## Part 5: Lazy Loading → Lazy Context Principles

Traditional lazy loading defers resource allocation until needed. In agentic systems, the scarce resource is **context window space** and **reasoning capacity**. These principles translate lazy loading patterns to skill design.

### LC1: LAZY REFERENCE LOADING

**Traditional**: Don't load objects until accessed.

**Agentic**: Don't load reference files until the execution branch requires them.

```
BAD (eager loading):
"First, read all reference files:
 - references/aws.md
 - references/gcp.md
 - references/azure.md
Then determine which cloud provider the user wants..."

GOOD (lazy loading):
"Determine the target cloud provider from the user's request.
Then read ONLY the relevant reference:
 - AWS → read references/aws.md
 - GCP → read references/gcp.md
 - Azure → read references/azure.md"
```

**Why it matters**: Loading all three providers consumes ~3x the context for no benefit. The LLM can only use one at a time.

---

### LC2: DEFERRED TOOL EXECUTION

**Traditional**: Lazy evaluation — don't compute until the value is needed.

**Agentic**: Don't call tools until you need their output for the next decision.

```
BAD (eager execution):
"Step 1: Query the database for all orders
 Step 2: Query the API for all customers
 Step 3: Query the cache for all products
 Step 4: Now analyze which orders need attention..."

GOOD (deferred execution):
"Step 1: Determine what analysis the user needs
 Step 2: Query ONLY the data sources required for that analysis
 Step 3: If the analysis reveals need for more data, query then"
```

**Why it matters**: Each tool call adds latency and context. Unnecessary queries waste both and may hit rate limits.

---

### LC3: PROGRESSIVE CONTEXT ENRICHMENT

**Traditional**: Virtual proxy — placeholder that loads real object on first use.

**Agentic**: Start with minimal context, enrich only when the task demands it.

```
SKILL.md structure for progressive enrichment:

## Quick Path (most common cases)
[Minimal instructions that handle 80% of requests]
[~100 lines]

## Extended Path (complex cases)
"If the request involves [complex scenario], read references/complex-handling.md"
[Loaded only for the 20%]

## Edge Cases
"If you encounter [rare situation], read references/edge-cases.md"
[Loaded only when needed]
```

**Pattern**: The skill body handles common cases directly. Complex cases are pointers to deeper content.

---

### LC4: CONTEXT GARBAGE COLLECTION

**Traditional**: Release memory when objects are no longer referenced.

**Agentic**: Actively summarize and discard intermediate context that's no longer needed.

```
GOOD (context-aware skill):
"After completing the analysis phase:
 1. Summarize key findings in 3-5 bullet points
 2. These bullets become the input for the next phase
 3. The detailed analysis can be discarded from working memory

The user doesn't need to see the intermediate work — only the
final output matters."
```

**Why it matters**: Long conversations accumulate context. Skills that explicitly manage what to retain vs. discard maintain reasoning quality.

---

### LC5: BATCH VS. STREAMING TOOL CALLS

**Traditional**: Batch processing vs. stream processing tradeoffs.

**Agentic**: Know when to batch tool calls vs. process incrementally.

```
BATCH (when results are independent):
"Query all three data sources in parallel:
 - Tool call 1: fetch_orders()
 - Tool call 2: fetch_inventory()
 - Tool call 3: fetch_shipments()
Then correlate the results."

STREAM (when results inform next query):
"1. Fetch the order details
 2. Based on order status, determine which additional data is needed
 3. Fetch only that additional data
 4. Repeat until analysis is complete"
```

**Principle**: Batch when independent. Stream when dependent. Never batch just to seem efficient.

---

### LC6: SKILL DELEGATION AS LAZY COMPOSITION

**Traditional**: Lazy initialization of composed objects.

**Agentic**: Don't read dependent skills until delegation point.

```
BAD (eager skill loading):
"This skill uses docx, xlsx, and pptx skills.
First, read all three SKILL.md files to understand capabilities..."

GOOD (lazy skill delegation):
"Determine the output format the user needs.
When ready to generate output:
 - DOCX → delegate to docx skill (read its SKILL.md at that point)
 - XLSX → delegate to xlsx skill
 - PPTX → delegate to pptx skill"
```

**Why it matters**: Reading three skill files upfront wastes context. You'll only use one.

---

### LC7: CACHING PATTERNS FOR SKILLS

**Traditional**: Cache computed results for reuse.

**Agentic**: Establish facts early that can be referenced later without recomputation.

```
GOOD (fact establishment):
"## Step 1: Establish Context (do this once)
Determine and record:
 - Document type: [PRD / PRFAQ / RFC / Other]
 - Target audience: [Technical / Executive / Mixed]
 - Quality bar: [Draft / Review-ready / Publication-ready]

Reference these established facts throughout execution.
Do not re-analyze the document type in later steps."
```

**Why it matters**: The LLM might "forget" and re-analyze. Explicit caching instructions prevent redundant work.

---

### LC8: CONDITIONAL ASSET LOADING

**Traditional**: Load assets (images, fonts, templates) on demand.

**Agentic**: Only reference templates/assets when the execution path needs them.

```
GOOD (conditional loading):
"## Template Selection
Based on document type:
 - PRD → use assets/prd-template.docx as base
 - PRFAQ → use assets/prfaq-template.docx as base
 - Custom → create from scratch (no template needed)

Only read the selected template. Do not load templates
you won't use."
```

---

### LC9: FAIL-FAST VALIDATION (Anti-Lazy Pattern)

**Traditional**: Sometimes eager validation prevents wasted work.

**Agentic**: Validate preconditions BEFORE doing expensive operations.

```
GOOD (eager validation):
"## Before Starting
Verify these preconditions:
 1. User has provided the required input file
 2. File format is supported (.xlsx, .xls, .csv)
 3. File is accessible and not corrupted

If any precondition fails, stop immediately and ask the user
to fix the issue. Do NOT proceed with partial data."
```

**Why it matters**: An hour of work invalidated by a missing file is worse than checking upfront.

---

### LC10: TIERED SKILL COMPLEXITY

**Traditional**: Multiple implementations (simple, standard, complex) chosen at runtime.

**Agentic**: Skill offers multiple "modes" based on task complexity.

```
## Execution Modes

### Quick Mode (simple requests)
For straightforward tasks with clear requirements:
[Minimal 5-step process]
[No references needed]

### Standard Mode (typical requests)
For typical complexity:
[Full workflow]
[May load 1-2 references]

### Deep Mode (complex requests)
For edge cases, debugging, or unusual requirements:
[Comprehensive workflow]
[Loads full reference suite]
[Includes validation and verification loops]

Assess complexity from user's request and choose appropriate mode.
When uncertain, start with Standard and escalate if needed.
```

---

### Lazy Context Principles Summary

| Principle | Traditional Pattern | Agentic Translation |
|-----------|--------------------|--------------------|
| LC1: Lazy Reference | Lazy initialization | Load references only when branch requires |
| LC2: Deferred Tools | Lazy evaluation | Call tools only when output needed |
| LC3: Progressive Enrichment | Virtual proxy | Start minimal, enrich on demand |
| LC4: Context GC | Garbage collection | Summarize and discard intermediate work |
| LC5: Batch vs Stream | Processing patterns | Batch independent, stream dependent |
| LC6: Lazy Delegation | Lazy composition | Read dependent skills at delegation point |
| LC7: Fact Caching | Result caching | Establish facts once, reference throughout |
| LC8: Conditional Assets | Asset loading | Load only selected templates |
| LC9: Fail-Fast | Eager validation | Validate preconditions before expensive work |
| LC10: Tiered Complexity | Strategy pattern | Quick/Standard/Deep modes |

---

<a id="part-6"></a>
## Part 6: First-Principles Thinking for LLM Skills

### The Three Cognitive Layers

An LLM consuming a skill thinks in three layers:

#### Layer 1: WHAT (The Domain Model)

The LLM needs a clear mental model of the domain:
- What are the key entities?
- What are the relationships?
- What are the invariants?
- What does "done" look like?

```markdown
## Domain Model

A financial spreadsheet is a living analytical tool, not a static data dump.
Its value comes from formulas that let the user change assumptions and see
results update automatically.

Key invariants:
- Every calculation uses Excel formulas, never hardcoded Python values
- All assumptions live in clearly marked input cells (blue text)
- Zero formula errors in the delivered file
```

---

#### Layer 2: WHY (The Reasoning Framework)

The most powerful and most neglected layer.

```markdown
## Why This Matters

We use Excel formulas instead of Python calculations because:
- The user will change assumptions after we deliver
- Hardcoded values break silently — looks right but won't update
- A financial model's purpose is scenario analysis

We put assumptions in separate cells because:
- Blue-text assumptions are an industry standard
- Buried assumptions make the model a black box
- Auditors need to trace every number to its source
```

---

#### Layer 3: HOW (The Execution Protocol)

Technical steps, informed by Layers 1 and 2:

```markdown
## Execution

1. Create the workbook with openpyxl
2. Define assumptions in dedicated cells (blue font)
3. Build formulas that reference assumption cells
4. NEVER calculate values in Python and hardcode them
5. Run scripts/recalc.py to validate all formulas
6. Zero tolerance for #REF!, #DIV/0! errors
```

---

### The First-Principles Skill Stack

```
┌─────────────────────────────────────────────┐
│  LAYER 3: HOW (Execution Protocol)          │
│  Steps, tools, scripts, validation          │
├─────────────────────────────────────────────┤
│  LAYER 2: WHY (Reasoning Framework)         │
│  Consequences, tradeoffs, edge case logic   │
├─────────────────────────────────────────────┤
│  LAYER 1: WHAT (Domain Model)               │
│  Entities, relationships, invariants        │
└─────────────────────────────────────────────┘
```

Most skills only have Layer 3. Good skills have Layers 2+3. Great skills have all three.

---

### The Five First Principles

#### FP1: The LLM is a reasoning system, not an executor
Write skills that *inform judgment*, not just dictate steps.

#### FP2: Context is finite and precious
Progressive disclosure isn't optional — it's survival.

#### FP3: The LLM already knows the domain (probably)
Skills *constrain and direct* existing knowledge. Think lenses, not textbooks.

#### FP4: Mechanical precision and creative judgment require opposite approaches
XML syntax → be precise. Design choices → declare intent.

#### FP5: Skills are used millions of times — they must generalize
Write for the distribution, not the examples.

---

<a id="part-7"></a>
## Part 7: References (Loaded on Demand)

The following content has been moved to `references/` to keep this file focused on core principles. Load them only when you need them:

| Reference | When to Load | File |
|-----------|-------------|------|
| **SKILL.md Template** | Creating a new skill | [references/skill-template.md](./references/skill-template.md) |
| **Anti-Patterns Catalog** | Debugging skill behavior, quality review | [references/anti-patterns-catalog.md](./references/anti-patterns-catalog.md) |
| **Real-World Skill Analysis** | Studying examples of well-designed skills | [references/skill-anatomy-examples.md](./references/skill-anatomy-examples.md) |
| **Anthropic Official Standards** | Verifying against canonical spec, Claude 4.x guidance | [references/anthropic-standards-2026.md](./references/anthropic-standards-2026.md) |

---

## Conclusion: The Translation Table

| Traditional Principle | Skill Equivalent | Core Insight |
|----------------------|-----------------|--------------|
| Domain-Driven Design | Domain Model section | Activate existing knowledge |
| Single Responsibility | Single Domain Ownership | One skill, one domain |
| Open/Closed | Extensible by Reference | Add references, don't rewrite core |
| Interface Segregation | Progressive Disclosure | Load only what's needed |
| Dependency Inversion | Depend on Capabilities | Reference skills, don't absorb |
| DRY | Don't Repeat Context | But repeat safety-critical rules |
| KISS | Keep Instructions Simple | One-sentence test |
| YAGNI | Prune Ruthlessly | Remove what doesn't improve output |
| Tell Don't Ask | Declare Intent | Trust the reasoning system |
| Lazy Loading | Lazy Context | Load references/skills/tools on demand |
| Composition over Inheritance | Compose by delegation | No "base skills" |

---

**The ultimate test**: Does your skill make the LLM *smarter* about the domain, or just more *obedient* about the steps? The best skills do both.

> *"If you find yourself writing ALWAYS or NEVER in all caps, that's a yellow flag — try to explain the reasoning so the model understands why."*
> — Anthropic Skill Creator
