You are reviewing or writing Go code against production-grade standards. Follow these phases IN ORDER.
Do not skip phases. Each phase has a GATE — complete it before proceeding.

## Pre-Flight: Decision Context

Before starting, check for these files in the current project root (read them if they exist, skip if not):
1. `DECISIONS.md` — prior decisions for this project. Do NOT re-decide what's already settled.
2. `GUARDRAILS.md` — anti-patterns from past incidents. These are hard constraints.

Also check: $HOME/code/aspora/ai-velocity/golang-coding-standards/ for DECISIONS.md and GUARDRAILS.md

## Phase 0: Analyze Existing Codebase

Before making changes:
- Read the project layout — identify cmd/, internal/, pkg/ structure
- Read existing patterns: error handling style, interface usage, concurrency patterns
- Check go.mod for dependencies and Go version
- Identify existing test patterns (table-driven? testify? gomock?)

GATE: Describe the patterns you found. If writing new code, state which existing patterns you will follow.

## Phase 1: Package Design

Verify or create the standard Go project layout:
- [ ] `cmd/` — entry points (main packages), one per binary
- [ ] `internal/` — private application code (not importable by other modules)
- [ ] `pkg/` — public reusable packages (only if genuinely reusable)
- [ ] `api/` — API definitions (OpenAPI specs, protobuf)
- [ ] `configs/` — configuration file templates

Rules:
- One package = one responsibility
- Package names are short, lowercase, no underscores
- No `utils/`, `helpers/`, `common/` packages — find a better name that describes the domain

## Phase 2: Interface Design

Read the skill reference for interface patterns:
$HOME/code/aspora/ai-velocity/golang-coding-standards/SKILL.md

Apply these rules:
- [ ] Define interfaces at the CONSUMER, not the provider ("accept interfaces, return structs")
- [ ] Keep interfaces small — 1-3 methods max
- [ ] No "God interfaces" with 5+ methods — split them
- [ ] Zero exported globals — use constructor functions

## Phase 3: Error Handling

Every error must be handled correctly:
- [ ] Always wrap errors with context: `fmt.Errorf("fetching user %d: %w", id, err)`
- [ ] Define domain-specific error variables: `var ErrNotFound = errors.New("not found")`
- [ ] Use `errors.Is()` and `errors.As()` for checking — never string comparison
- [ ] Never return bare `err` — always add context
- [ ] Document deliberately ignored errors with `//nolint` or comment explaining why

## Phase 4: Concurrency (if applicable)

If the code uses goroutines or channels:
- [ ] Bounded concurrency — use semaphores or worker pools, NEVER unbounded goroutine spawning
- [ ] Context propagation — every async operation accepts `context.Context` as first parameter
- [ ] Graceful shutdown — respect context cancellation
- [ ] Channel ownership — the goroutine that creates a channel should close it

## Phase 5: Verification

| Standard | Check | Status |
|----------|-------|--------|
| Layout | Standard cmd/internal/pkg structure? | [ ] |
| Interfaces | Defined at consumer, 1-3 methods? | [ ] |
| Errors | Wrapped with context using %w? | [ ] |
| Errors | Domain error variables defined? | [ ] |
| Functions | < 20 lines, < 3 params? | [ ] |
| Naming | No utils/helpers/common packages? | [ ] |
| Concurrency | Bounded goroutines with context? | [ ] |
| Exports | No exported globals? | [ ] |

## Phase 6: Self-Check

1. "Are all interfaces defined at the consumer side?" → Must be YES
2. "Does every error return include wrapping context with %w?" → Must be YES
3. "Are there any unbounded goroutine spawns?" → Must be NO
4. "Do all public functions have GoDoc comments?" → Must be YES
5. "Would `go vet` and `golint` pass?" → Must be YES
