---
name: golang-unit-testing
description: "Generate high-quality unit test cases for Go code using table-driven tests, mocks, and Go testing best practices. Follows SOLID principles, DRY principle, and comprehensive test coverage patterns. Use when writing tests for Go functions, methods, or services."
version: 1.0.0
author: AI Velocity Team
---

# Go Unit Testing Skill

## Purpose
This skill enables AI agents to generate high-quality unit test cases for Go code, following modern best practices including SOLID principles, DRY principle, table-driven tests, and comprehensive test coverage. Specifically designed for Go's testing conventions with `testing` package, mocks, and test helpers.

## When to Trigger This Skill
- **Mandatory: Every code change** — Whenever you add, modify, or remove production code, you must add or update unit tests for the changed behavior. No code change is complete without corresponding tests.
- When user requests: "Write test cases for this Go function/struct/method..."
- After code changes or pull requests to validate new/modified functionality
- When no tests exist for a function, method, or package
- When reviewing code and test coverage is needed
- For new features, bug fixes, or any logical/computational changes
- When user asks: "Add tests", "Write unit tests", "Test coverage", "Generate tests"
- When implementing TDD (Test-Driven Development)

## Critical: Automatic Codebase Analysis

**ALWAYS analyze the codebase first before asking questions or making assumptions.**

1. **Read the codebase automatically**:
   - Read the function/struct/method to be tested
   - Identify all exported and testable functions
   - Understand dependencies from function parameters and struct fields
   - Identify edge cases from conditional logic and error handling
   - Check existing test patterns in the codebase (test file naming, structure)
   - Identify interfaces that need mocking

2. **Do NOT ask the user to describe the code**:
   - The codebase contains all necessary information
   - Understand behavior from function implementations
   - Identify dependencies from imports and function signatures
   - Discover edge cases from conditional logic and error paths
   - Understand test patterns from existing `*_test.go` files

3. **Only ask clarifying questions if**:
   - Code is unclear or ambiguous
   - Multiple interpretations are possible
   - Critical information is missing from codebase
   - Test requirements are unclear (unit vs integration)

## Core Principles

### SOLID Principles in Testing

1. **Single Responsibility Principle (SRP)**
   - Each test function should test one specific behavior or scenario
   - Avoid testing multiple behaviors in a single test
   - One test file per source file (e.g., `service.go` → `service_test.go`)

2. **Open/Closed Principle**
   - Tests should be extensible without modifying existing tests
   - Use table-driven tests for similar scenarios with different inputs
   - Add new test cases rather than rewriting existing ones

3. **Liskov Substitution Principle**
   - Use interfaces and abstractions in test code
   - Mock dependencies using interfaces, not concrete implementations
   - Test implementations should satisfy interface contracts

4. **Interface Segregation Principle**
   - Mock only what you need, not entire interfaces
   - Use specific mock configurations for each test case
   - Create focused mock implementations

5. **Dependency Inversion Principle**
   - Dependencies should be mocked/injected, not hardcoded
   - Test code should depend on abstractions (interfaces), not concrete implementations
   - Use dependency injection patterns in test setup

### DRY (Don't Repeat Yourself) Principle

- Extract common test setup into helper functions
- Use table-driven tests for similar scenarios
- Create reusable test utilities and builders
- Use test fixtures for complex data structures
- Avoid duplicating test logic across multiple test functions

### Table-Driven Tests (Go Idiom)

Table-driven tests are the Go standard for testing multiple scenarios:

```go
func TestFunctionName(t *testing.T) {
    tests := []struct {
        name    string
        input   InputType
        want    OutputType
        wantErr error
    }{
        {
            name:    "success case",
            input:   validInput,
            want:    expectedOutput,
            wantErr: nil,
        },
        {
            name:    "error case",
            input:   invalidInput,
            want:    zeroValue,
            wantErr: ErrInvalidInput,
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            got, err := FunctionName(tt.input)
            if !errors.Is(err, tt.wantErr) {
                t.Errorf("FunctionName() error = %v, wantErr %v", err, tt.wantErr)
                return
            }
            if !reflect.DeepEqual(got, tt.want) {
                t.Errorf("FunctionName() = %v, want %v", got, tt.want)
            }
        })
    }
}
```

## Test Structure & Organization

### File Naming
- Test files must end with `_test.go`
- Test file should be in the same package as the code being tested
- Example: `service.go` → `service_test.go`

### Package Naming
- Use same package name for unit tests (package-level access)
- Use `package_test` for integration tests (external access only)

### Test Function Naming
- Test functions must start with `Test`
- Follow pattern: `TestFunctionName` or `TestMethodName_Scenario`
- Use descriptive names: `TestValidatePayment_NegativeAmount_ReturnsError`

### Test Organization
```go
// Test helpers and fixtures at top
func newTestService(t *testing.T) *Service { ... }
func paymentBuilder() *Payment { ... }

// Test functions
func TestService_ProcessOrder(t *testing.T) { ... }
func TestService_ProcessOrder_ErrorCases(t *testing.T) { ... }
```

## Test Patterns

### 1. Table-Driven Tests (Standard Pattern)

**Use for:** Multiple test cases with same structure

```go
func TestValidatePayment(t *testing.T) {
    tests := []struct {
        name    string
        payment *Payment
        wantErr error
    }{
        {
            name:    "valid payment",
            payment: &Payment{Amount: 100, Currency: "USD"},
            wantErr: nil,
        },
        {
            name:    "negative amount",
            payment: &Payment{Amount: -10, Currency: "USD"},
            wantErr: ErrInvalidAmount,
        },
        {
            name:    "zero amount",
            payment: &Payment{Amount: 0, Currency: "USD"},
            wantErr: ErrInvalidAmount,
        },
        {
            name:    "missing currency",
            payment: &Payment{Amount: 100, Currency: ""},
            wantErr: ErrCurrencyRequired,
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            err := ValidatePayment(tt.payment)
            if !errors.Is(err, tt.wantErr) {
                t.Errorf("ValidatePayment() error = %v, wantErr %v", err, tt.wantErr)
            }
        })
    }
}
```

### 2. Mocking Dependencies

**Simple Mock Implementation:**
```go
type mockRepository struct {
    saveFunc func(ctx context.Context, entity *Entity) error
    getFunc  func(ctx context.Context, id string) (*Entity, error)
}

func (m *mockRepository) Save(ctx context.Context, entity *Entity) error {
    if m.saveFunc != nil {
        return m.saveFunc(ctx, entity)
    }
    return nil
}

func (m *mockRepository) Get(ctx context.Context, id string) (*Entity, error) {
    if m.getFunc != nil {
        return m.getFunc(ctx, id)
    }
    return nil, ErrNotFound
}

func TestService_ProcessOrder(t *testing.T) {
    mockRepo := &mockRepository{
        saveFunc: func(ctx context.Context, entity *Entity) error {
            if entity.ID == "" {
                return ErrInvalidID
            }
            return nil
        },
    }
    
    service := NewService(mockRepo, nil, nil)
    err := service.ProcessOrder(context.Background(), &Order{ID: "123"})
    if err != nil {
        t.Errorf("ProcessOrder() error = %v, want nil", err)
    }
}
```

**Using testify/mock (if project uses it):**
```go
import (
    "github.com/stretchr/testify/mock"
    "github.com/stretchr/testify/assert"
)

type mockRepository struct {
    mock.Mock
}

func (m *mockRepository) Save(ctx context.Context, entity *Entity) error {
    args := m.Called(ctx, entity)
    return args.Error(0)
}

func TestService_ProcessOrder(t *testing.T) {
    mockRepo := new(mockRepository)
    mockRepo.On("Save", mock.Anything, mock.AnythingOfType("*Entity")).Return(nil)
    
    service := NewService(mockRepo, nil, nil)
    err := service.ProcessOrder(context.Background(), &Order{ID: "123"})
    
    assert.NoError(t, err)
    mockRepo.AssertExpectations(t)
}
```

### 3. Testing with Context

**Test context cancellation and timeouts:**
```go
func TestService_ProcessOrder_ContextCancelled(t *testing.T) {
    service := newTestService(t)
    
    ctx, cancel := context.WithCancel(context.Background())
    cancel()  // Cancel immediately
    
    err := service.ProcessOrder(ctx, &Order{ID: "123"})
    if !errors.Is(err, context.Canceled) {
        t.Errorf("ProcessOrder() error = %v, want context.Canceled", err)
    }
}

func TestService_ProcessOrder_ContextTimeout(t *testing.T) {
    service := newTestService(t)
    
    ctx, cancel := context.WithTimeout(context.Background(), 1*time.Nanosecond)
    defer cancel()
    
    time.Sleep(2 * time.Nanosecond)  // Ensure timeout
    
    err := service.ProcessOrder(ctx, &Order{ID: "123"})
    if !errors.Is(err, context.DeadlineExceeded) {
        t.Errorf("ProcessOrder() error = %v, want context.DeadlineExceeded", err)
    }
}
```

### 4. Test Helpers and Builders

**Test Helpers:**
```go
func newTestService(t *testing.T) (*Service, *mockRepository) {
    mockRepo := &mockRepository{}
    service := NewService(mockRepo, nil, nil)
    return service, mockRepo
}

func paymentBuilder() *Payment {
    return &Payment{
        Amount:   100,
        Currency: "USD",
        Status:   StatusPending,
    }
}

func paymentWithAmount(amount int) *Payment {
    p := paymentBuilder()
    p.Amount = amount
    return p
}
```

**Test Fixtures:**
```go
func loadTestData(t *testing.T, filename string) []byte {
    data, err := os.ReadFile(filepath.Join("testdata", filename))
    if err != nil {
        t.Fatalf("failed to load test data: %v", err)
    }
    return data
}
```

### 5. Testing Error Cases

**Always test error paths:**
```go
func TestService_ProcessOrder_ErrorCases(t *testing.T) {
    tests := []struct {
        name      string
        setupMock func(*mockRepository)
        wantErr   error
    }{
        {
            name: "repository save fails",
            setupMock: func(m *mockRepository) {
                m.saveFunc = func(ctx context.Context, entity *Entity) error {
                    return ErrDatabaseConnection
                }
            },
            wantErr: ErrDatabaseConnection,
        },
        {
            name: "invalid order",
            setupMock: func(m *mockRepository) {},
            wantErr: ErrInvalidOrder,
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            mockRepo := &mockRepository{}
            tt.setupMock(mockRepo)
            service := NewService(mockRepo, nil, nil)
            
            err := service.ProcessOrder(context.Background(), &Order{})
            if !errors.Is(err, tt.wantErr) {
                t.Errorf("ProcessOrder() error = %v, wantErr %v", err, tt.wantErr)
            }
        })
    }
}
```

### 6. Testing Concurrent Code

**Always test concurrency when the code has shared state, goroutines, or may be called from multiple goroutines.**

- **Run tests with `-race`**: `go test -race ./...` to detect data races. Add concurrency tests that stress shared components (circuit breakers, caches, processors).
- **Use thread-safe mocks**: When testing components that are called from multiple goroutines, use mocks that protect shared slices/maps with `sync.Mutex` and assert final counts or state after `sync.WaitGroup` completes.
- **Assert invariants, not exact order**: After concurrent execution, assert that final state is one of the valid states (e.g. circuit breaker is Closed, Open, or HalfOpen), total count matches expected, or no panic occurred.
- **Test both success and contention**: Test concurrent success paths and paths that may trigger failure (e.g. circuit breaker opening under concurrent RecordFailure).

**Test goroutines and channels:**
```go
func TestProcessItems_Concurrent(t *testing.T) {
    items := []Item{{ID: "1"}, {ID: "2"}, {ID: "3"}}
    results := ProcessItems(items)
    
    var got []Result
    for result := range results {
        got = append(got, result)
    }
    
    if len(got) != len(items) {
        t.Errorf("ProcessItems() returned %d results, want %d", len(got), len(items))
    }
}

func TestSafeCounter_Concurrent(t *testing.T) {
    counter := &SafeCounter{}
    var wg sync.WaitGroup
    iterations := 1000
    goroutines := 10
    
    for i := 0; i < goroutines; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for j := 0; j < iterations; j++ {
                counter.Increment()
            }
        }()
    }
    
    wg.Wait()
    
    if got := counter.Value(); got != goroutines*iterations {
        t.Errorf("SafeCounter.Value() = %d, want %d", got, goroutines*iterations)
    }
}
```

**Concurrent stress test with race detector:**
```go
// Run with: go test -race ./internal/mypkg/...
func TestCircuitBreaker_Concurrent_Stress(t *testing.T) {
    cb := NewCircuitBreaker(5, 20*time.Millisecond)
    var wg sync.WaitGroup
    for i := 0; i < 100; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for j := 0; j < 10; j++ {
                if cb.Allow() {
                    cb.RecordSuccess()
                } else {
                    cb.RecordFailure()
                }
            }
        }()
    }
    wg.Wait()
    state := cb.State()
    if state != StateClosed && state != StateOpen && state != StateHalfOpen {
        t.Errorf("invalid state after concurrent stress: %v", state)
    }
}
```

**Same-key / same-entity concurrency (read-modify-write):**

Concurrency tests that use **different** keys per goroutine (e.g. `user-0`, `user-1`, …) only stress throughput; they **do not** catch races where multiple goroutines update the **same** entity. Lost updates (e.g. one goroutine’s write overwriting another’s) are a **logical** race, not always a data race the `-race` detector will report.

- **When updates are keyed by entity** (e.g. cache or DB row per `user_id`, `order_id`): Add a test that runs concurrent operations for the **same** key/entity and asserts either (a) serialization (critical sections do not interleave), or (b) deterministic final state (e.g. final document reflects all updates).
- **Test the path that does read-modify-write**: Testing only a pure function (e.g. transformer) from many goroutines with different inputs does not exercise the full update path. If production uses a per-entity lock around read→transform→write, add a test that verifies the lock (e.g. same-key serialization) or the full path with a mock store.
- **Per-entity lock tests**: When code uses per-entity or per-key locks (e.g. sharded mutex by `user_id`), add a test that concurrent lock/unlock for the **same** key serializes (e.g. record order of “start”/“end” of each goroutine’s critical section and assert no interleaving).

**Example: same-key serialization (per-entity lock):**
```go
// Verifies that concurrent lock/unlock for the SAME key serializes (no interleaving).
// Run with: go test -race ./internal/cache/...
func TestUserLocks_SerializesSameUser(t *testing.T) {
    var l userLocks
    const sameUser = "user-1"
    const numGoroutines = 20

    var orderMu sync.Mutex
    var order []int // each goroutine appends id, then id+1000; serialization => no other id between them
    var wg sync.WaitGroup
    for i := 0; i < numGoroutines; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            l.lock(sameUser)
            orderMu.Lock()
            order = append(order, id)
            orderMu.Unlock()
            time.Sleep(1 * time.Millisecond)
            orderMu.Lock()
            order = append(order, id+1000)
            orderMu.Unlock()
            l.unlock(sameUser)
        }(i)
    }
    wg.Wait()

    // With serialization: last "start" (value < 1000) before each "end" (value >= 1000) must match (end - 1000).
    orderMu.Lock()
    defer orderMu.Unlock()
    lastStart := -1
    for _, v := range order {
        if v < 1000 {
            lastStart = v
        } else {
            if lastStart != v-1000 {
                t.Errorf("critical section interleaved: last start=%d but end=%d", lastStart, v-1000)
            }
        }
    }
}
```

### 7. Testing Out-of-Order and Unhappy Paths

**Do not test only happy paths. Explicitly cover out-of-order events, duplicates, and edge cases.**

- **Out-of-order events**: Test behavior when events arrive in wrong order (e.g. refund before order, UPDATE before CREATE). Document expected behavior: e.g. aggregates may go negative, or duplicates are idempotent.
- **Duplicate events**: Test same entity + same status + no field change → filtered or idempotent. Test duplicate CREATE or UPDATE and assert no double-count.
- **Missing or invalid data**: Test DELETE op not processed, nil/empty fields, before_status empty (out-of-order UPDATE). Assert filtering or safe fallback.
- **Event sequences**: Apply multiple events in a fixed order (order → refund → order) and assert final aggregate state. Covers in-order processing semantics.
- **Negative/zero totals**: Test refund-before-order or over-refund; assert either negative totals are allowed or logic rejects/prevents them.

**Example: out-of-order and sequence tests:**
```go
func TestTransform_OutOfOrder_RefundBeforeOrder(t *testing.T) {
    // Refund arrives before any order (replay or out-of-order)
    current := map[string]interface{}{"total_spent_gbp": 0.0}
    updates, _ := transformer.Transform(refundEvent, current, nil)
    if updates["total_spent_gbp"] != -25.0 {
        t.Errorf("out-of-order refund: total_spent_gbp = %v, want -25.0", updates["total_spent_gbp"])
    }
}

func TestShouldProcess_Duplicate_UpdateSameStatus_Filtered(t *testing.T) {
    event := updateEventWithStatus("COMPLETED", "COMPLETED", noFieldChange)
    got := shouldProcessEvent(event, cfg)
    if got {
        t.Error("duplicate UPDATE (same status, no field change) should be filtered")
    }
}

func TestShouldProcess_Delete_NotProcessed(t *testing.T) {
    event := rawEventWithOp("d", "COMPLETED")
    if shouldProcessEvent(event, cfg) {
        t.Error("DELETE op should not be processed")
    }
}
```

## Test Coverage Guidelines

### What to Test

**Must Test:**
- All public functions and methods
- All error paths and edge cases (not just happy path)
- Boundary conditions (zero, negative, max values)
- Nil pointer handling
- Context cancellation and timeouts
- **Concurrent operations** when the code has shared state or is used from multiple goroutines; run with `-race`
- **Out-of-order and unhappy paths**: out-of-order events, duplicates, DELETE/edge ops, event sequences, negative/zero aggregates where applicable

**Should Test:**
- Complex business logic
- Validation functions
- Transformation functions
- State transitions
- Event ordering and idempotency semantics

**May Skip:**
- Simple getters/setters (unless they have logic)
- Trivial wrapper functions
- Generated code

### Coverage Goals

- Aim for 80%+ coverage for business logic
- 100% coverage for critical paths (payment processing, authentication)
- Focus on quality over quantity (better to have fewer, better tests)

## Common Patterns

### Testing HTTP Handlers

```go
func TestHandler_CreateOrder(t *testing.T) {
    service := newTestService(t)
    handler := NewHandler(service)
    
    req := httptest.NewRequest("POST", "/orders", strings.NewReader(`{"amount":100}`))
    req.Header.Set("Content-Type", "application/json")
    w := httptest.NewRecorder()
    
    handler.CreateOrder(w, req)
    
    if w.Code != http.StatusCreated {
        t.Errorf("CreateOrder() status = %d, want %d", w.Code, http.StatusCreated)
    }
}
```

### Testing with Time

```go
func TestService_ProcessOrder_WithTime(t *testing.T) {
    // Use time.Now() in tests, but make it deterministic if needed
    now := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
    
    service := &Service{clock: func() time.Time { return now }}
    // ... test logic
}
```

### Testing JSON Marshaling

```go
func TestPayment_MarshalJSON(t *testing.T) {
    payment := &Payment{Amount: 100, Currency: "USD"}
    
    data, err := json.Marshal(payment)
    if err != nil {
        t.Fatalf("MarshalJSON() error = %v", err)
    }
    
    var got Payment
    if err := json.Unmarshal(data, &got); err != nil {
        t.Fatalf("UnmarshalJSON() error = %v", err)
    }
    
    if got.Amount != payment.Amount {
        t.Errorf("Amount = %d, want %d", got.Amount, payment.Amount)
    }
}
```

## Common Mistakes

### ❌ Testing Implementation Details
```go
// BAD: Testing internal state
func TestService(t *testing.T) {
    s := NewService()
    if s.repo == nil {  // Testing private field
        t.Error("repo should not be nil")
    }
}

// GOOD: Testing behavior
func TestService_ProcessOrder(t *testing.T) {
    s := NewService()
    err := s.ProcessOrder(ctx, order)
    if err != nil {
        t.Errorf("ProcessOrder() error = %v", err)
    }
}
```

### ❌ Not Using Table-Driven Tests
```go
// BAD: Multiple separate test functions
func TestValidatePayment_Valid(t *testing.T) { ... }
func TestValidatePayment_InvalidAmount(t *testing.T) { ... }
func TestValidatePayment_InvalidCurrency(t *testing.T) { ... }

// GOOD: Table-driven test
func TestValidatePayment(t *testing.T) {
    tests := []struct { ... } { ... }
    for _, tt := range tests { ... }
}
```

### ❌ Ignoring Errors in Tests
```go
// BAD: Ignoring error
result, _ := operation()

// GOOD: Check error
result, err := operation()
if err != nil {
    t.Fatalf("operation() error = %v", err)
}
```

### ❌ Concurrency tests with different keys only (miss same-entity race)
```go
// BAD: Different user ID per goroutine - does NOT catch same-user race (e.g. cache overwrites)
for i := 0; i < numGoroutines; i++ {
    go func(id int) {
        userID := fmt.Sprintf("user-%d", id)
        UpdateCache(eventFor(userID))
    }(i)
}
// GOOD: Also add a test where all goroutines use the SAME key and assert serialization or deterministic final state.
```

### ❌ Not Testing Error Cases
```go
// BAD: Only testing success
func TestService_ProcessOrder(t *testing.T) {
    err := service.ProcessOrder(ctx, validOrder)
    if err != nil {
        t.Errorf("error = %v", err)
    }
}

// GOOD: Testing both success and error cases
func TestService_ProcessOrder(t *testing.T) {
    tests := []struct {
        name    string
        order   *Order
        wantErr error
    }{
        {name: "success", order: validOrder, wantErr: nil},
        {name: "invalid order", order: invalidOrder, wantErr: ErrInvalidOrder},
    }
    // ...
}
```

## Rule: Test Every Code Change

**Whenever you change production code (new feature, config option, refactor, or bug fix):**

1. **Add or update tests in the same change** — Do not leave test updates for "later." Include test additions/edits in the same work as the code change.
2. **Cover the new or changed behavior** — At minimum: one test that demonstrates the new path or config (e.g. new filter, new aggregate type, new config field).
3. **Cover edge and error cases** — If you add a new branch (e.g. `NotifyEventTypes`), add tests for "in list" and "not in list"; if you add a new helper (e.g. `getNestedFromMap`), add tests for direct key, nested path, and missing key.
4. **Place tests in the right file** — Same package, `*_test.go` next to the code (e.g. `config.go` → `config_test.go`, `transformers.go` → `transformers_test.go`).

If the codebase has no test file for the package, create one. Prefer table-driven tests for multiple scenarios.

## Quick Reference Checklist

When generating tests, verify:

- [ ] Test file named `*_test.go` in same package
- [ ] Table-driven tests for multiple scenarios
- [ ] All error paths tested (not just happy path)
- [ ] Edge cases and boundary conditions covered
- [ ] Dependencies mocked via interfaces
- [ ] Test helpers extracted for common setup
- [ ] Context cancellation/timeout tested (if applicable)
- [ ] **Concurrent operations tested** when code has shared state or may be called from multiple goroutines; use thread-safe mocks and run `go test -race`
- [ ] **Same-key/same-entity concurrency tested** when updates are keyed by entity (e.g. per-user cache, per-order state): add a test that runs concurrent operations for the **same** key and asserts serialization or deterministic final state (different keys per goroutine do not catch same-entity races)
- [ ] **Out-of-order and unhappy paths tested**: out-of-order events (e.g. refund before order), duplicate events (same status, no field change), DELETE/edge ops, event sequences
- [ ] **Tests added/updated for every code change**: new config fields, new filters, new helpers, or refactors have at least one test covering the new behavior
- [ ] Test names are descriptive
- [ ] No testing of implementation details
- [ ] Proper use of `t.Run()` for subtests
- [ ] Error comparison using `errors.Is()`

## Additional Resources

- [Go Testing Package](https://pkg.go.dev/testing)
- [Table-Driven Tests (Go Blog)](https://go.dev/blog/subtests)
- [testify/mock](https://github.com/stretchr/testify) - Popular testing toolkit
- [gomock](https://github.com/golang/mock) - Mock generation tool
- [Go Code Review Comments - Testing](https://go.dev/wiki/CodeReviewComments#tests)
