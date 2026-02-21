You are writing or improving Go unit tests. Follow these phases IN ORDER.
Do not skip phases. Each phase has a GATE — complete it before proceeding.

## Pre-Flight: Decision Context

Before starting, check for these files in the current project root (read them if they exist, skip if not):
1. `DECISIONS.md` — prior decisions for this project. Do NOT re-decide what's already settled.
2. `GUARDRAILS.md` — anti-patterns from past incidents. These are hard constraints.

Also check: $HOME/code/aspora/ai-velocity/golang-unit-testing/ for DECISIONS.md and GUARDRAILS.md

## Phase 0: Analyze Target Code

Before writing any tests:
- Read the function/struct/package you are testing
- Identify all dependencies (interfaces it accepts, external calls it makes)
- Identify edge cases: nil inputs, empty slices, zero values, context cancellation
- Check for existing tests — extend, don't duplicate
- Identify the existing test patterns in the project (testify? gomock? table-driven?)

GATE: List: (1) functions to test, (2) dependencies to mock, (3) edge cases identified. Present to user.

## Phase 1: Test Structure

Set up the test file correctly:
- [ ] File name: `{source}_test.go` in the same package
- [ ] Use `package {name}` for unit tests (access unexported), `package {name}_test` for black-box tests
- [ ] Import: `testing`, plus `testify/assert` or `testify/require` for assertions
- [ ] Mock setup: `testify/mock` or `gomock` for interface mocks

## Phase 2: Write Table-Driven Tests

Every function with multiple scenarios MUST use table-driven tests:

```go
func TestFunctionName(t *testing.T) {
    tests := []struct {
        name    string
        input   InputType
        want    OutputType
        wantErr bool
    }{
        {"happy path", validInput, expectedOutput, false},
        {"empty input", emptyInput, zeroValue, true},
        {"nil context", nilCtx, zeroValue, true},
    }
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            got, err := FunctionUnderTest(tt.input)
            if tt.wantErr {
                assert.Error(t, err)
                return
            }
            assert.NoError(t, err)
            assert.Equal(t, tt.want, got)
        })
    }
}
```

Read the skill reference for test patterns:
$HOME/code/aspora/ai-velocity/golang-unit-testing/SKILL.md

## Phase 3: Coverage Requirements

Every test suite must cover:
- [ ] Happy path — normal successful operation
- [ ] Empty/zero inputs — empty strings, nil pointers, zero-length slices
- [ ] Error paths — every error return branch exercised
- [ ] Boundary values — max int, empty collections, single-element collections
- [ ] Concurrent access — if the code is used concurrently, test with `t.Parallel()` and `-race`

## Phase 4: Mock Verification

If using mocks:
- [ ] Mock only interfaces you OWN — wrap third-party code first, mock the wrapper
- [ ] Verify expected calls: `mock.AssertExpectations(t)`
- [ ] Test both success and error returns from mocked dependencies
- [ ] No mocking of the thing under test — only its dependencies

## Phase 5: Verification

| Check | Status |
|-------|--------|
| Table-driven tests for multi-case functions? | [ ] |
| AAA pattern (Arrange-Act-Assert) in every test? | [ ] |
| Happy path covered? | [ ] |
| Error paths covered? | [ ] |
| Edge cases (nil, empty, zero) covered? | [ ] |
| Mocks only on owned interfaces? | [ ] |
| Tests pass: `go test ./... -race`? | [ ] |
| No test logic duplication (DRY helpers)? | [ ] |

## Phase 6: Self-Check

1. "Do all multi-case functions use table-driven tests?" → Must be YES
2. "Does every test follow AAA (Arrange-Act-Assert)?" → Must be YES
3. "Are we mocking interfaces we own, not third-party code directly?" → Must be YES
4. "Would these tests survive a refactor of internal implementation?" → Must be YES (test behavior, not implementation)
5. "Did I run tests with -race flag?" → Must be YES
