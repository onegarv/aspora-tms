---
name: golang-coding-standards
description: "Use when bootstrapping new Go services, implementing functionality in existing Go repositories, or reviewing Go code. Enforces production-grade Go standards following SOLID, DRY, clean code principles, and patterns from highly scalable repositories like Kubernetes and Pinot. Includes OpenAPI specification requirements for REST APIs."
version: 1.2.0
author: AI Velocity Team
---

# Go Coding Standards and Best Practices

## Purpose
This skill enables AI agents to write production-grade Go code following first principles, SOLID, DRY, and clean code principles. Standards are derived from highly scalable repositories like Kubernetes and Pinot, and proven production examples like ardanlabs/service, ensuring code quality, maintainability, and scalability from the start.

**Key Patterns Enforced:**
- Domain-specific error variables (Kubernetes pattern)
- Context usage in async operations (prevent goroutine leaks)
- Bounded concurrency with worker pools
- Proper error wrapping with `Unwrap()` and `Is()` methods
- Documented ignored errors (non-blocking operations)

## When to Trigger This Skill
- When bootstrapping a new Go service or project
- When implementing new functionality in existing Go repositories
- When user requests: "Create a new Go service", "Add this feature", "Implement this in Go"
- When creating REST APIs (also create OpenAPI spec)
- During code reviews for Go code quality
- When refactoring or improving existing Go code
- When designing Go packages, interfaces, or modules

## Critical: Automatic Codebase Analysis

**ALWAYS analyze the codebase first before asking questions or making assumptions.**

1. **Read the codebase automatically**:
   - Understand existing package structure and organization
   - Identify established patterns, interfaces, and abstractions
   - Map dependencies and relationships from imports
   - Recognize domain entities and business logic from code structure
   - Discover error handling patterns and conventions

2. **Do NOT ask the user to describe their architecture**:
   - The codebase contains all necessary information
   - Understand patterns from existing code
   - Identify conventions from code style and structure
   - Discover abstractions from interface definitions

3. **Only ask clarifying questions if**:
   - Code is unclear or ambiguous
   - Multiple valid approaches exist and context is needed
   - Critical business requirements are missing from codebase

## Core Principles

### 1. Package Design and Organization

**Single Responsibility per Package**
- Each package should have one clear purpose
- Package names should be descriptive and domain-focused
- Avoid generic names like `utils`, `helpers`, `common` (be specific)

**Package Structure (Standard Go Layout)**
```
cmd/
  app/
    main.go             # Application entry point
internal/
  domain/
    service.go          # Business logic
    repository.go       # Data access interface
    model.go            # Domain models
    errors.go           # Domain-specific errors
  domain/
    repository/
      postgres.go       # Implementation
pkg/
  public/               # Public library code (if needed)
```

**Key Directories (from golang-standards/project-layout):**
- `cmd/` - Main applications (one per executable)
- `internal/` - Private application/library code (enforced by Go compiler)
- `pkg/` - Library code safe for external use
- `api/` - API definitions (OpenAPI, protocol buffers)
- `configs/` - Configuration file templates

**Visibility Rules**
- Unexported (lowercase) by default - only export what's needed
- Export interfaces, not implementations (when possible)
- Use internal packages for implementation details

**Example:**
```go
// ✅ GOOD: Clear package purpose, minimal exports
package payment

// Service interface - exported for dependency injection
type Service interface {
    ProcessPayment(ctx context.Context, req *PaymentRequest) (*Payment, error)
}

// Internal implementation
type service struct {
    repo Repository
}

// ✅ GOOD: Domain-specific errors
package payment

var (
    ErrInsufficientFunds = errors.New("payment: insufficient funds")
    ErrInvalidAmount     = errors.New("payment: invalid amount")
)
```

### 2. Interface Design (Interface Segregation)

**Small, Focused Interfaces**
- Prefer many small interfaces over few large ones
- Interfaces should define behavior, not data
- Follow the "accept interfaces, return structs" principle

**Example:**
```go
// ✅ GOOD: Small, focused interfaces
type Reader interface {
    Read(ctx context.Context, id string) (*Entity, error)
}

type Writer interface {
    Write(ctx context.Context, entity *Entity) error
}

type Repository interface {
    Reader
    Writer
}

// ❌ BAD: Large interface with unrelated methods
type Repository interface {
    Read(ctx context.Context, id string) (*Entity, error)
    Write(ctx context.Context, entity *Entity) error
    SendEmail(ctx context.Context, email string) error  // Wrong abstraction
    ProcessPayment(ctx context.Context, amount int) error  // Wrong abstraction
}
```

**Accept Interfaces, Return Structs**
```go
// ✅ GOOD: Function accepts interface
func ProcessOrder(service OrderService, order *Order) error {
    return service.Process(order)
}

// ✅ GOOD: Constructor returns concrete type
func NewOrderService(repo Repository) *OrderService {
    return &OrderService{repo: repo}
}
```

### 3. Dependency Injection and Composition

**Constructor Injection**
- All dependencies passed via constructors
- No global state or singletons
- Use dependency injection frameworks (Wire, fx) for complex graphs

**Example:**
```go
// ✅ GOOD: Dependencies injected via constructor
type Service struct {
    repo     Repository
    producer EventProducer
    logger   Logger
}

func NewService(repo Repository, producer EventProducer, logger Logger) *Service {
    return &Service{
        repo:     repo,
        producer: producer,
        logger:   logger,
    }
}

// ❌ BAD: Global dependencies
var globalRepo Repository

func NewService() *Service {
    return &Service{repo: globalRepo}  // Hidden dependency
}
```

**Composition over Inheritance**
- Use embedding for code reuse
- Prefer composition for behavior extension

### 4. Error Handling

**Explicit Error Handling**
- Never ignore errors (use `_ =` only when absolutely necessary with comment)
- Return errors, don't panic (except for programming errors)
- Wrap errors with context using `fmt.Errorf` with `%w` verb
- Always document why errors are ignored (non-blocking operations, best-effort cleanup)

**Domain-Specific Error Variables (Kubernetes Pattern)**
```go
// ✅ GOOD: Define domain-specific error variables
package api

var (
    ErrInvalidRequest      = errors.New("api: invalid request")
    ErrQueryValidationFailed = errors.New("api: query validation failed")
    ErrQueryExecutionFailed  = errors.New("api: query execution failed")
    ErrResultTooLarge       = errors.New("api: result too large")
    ErrCacheUnavailable      = errors.New("api: cache unavailable")
    ErrTimeout               = errors.New("api: operation timeout")
)

// Usage with error wrapping
func (s *Service) ExecuteQuery(ctx context.Context, sql string) error {
    if err := s.validate(sql); err != nil {
        return fmt.Errorf("query validation: %w", ErrQueryValidationFailed)
    }
    
    result, err := s.pinotClient.Query(ctx, sql)
    if err != nil {
        return fmt.Errorf("query execution: %w", ErrQueryExecutionFailed)
    }
    
    return nil
}
```

**Error Types with Unwrap() and Is()**
```go
// ✅ GOOD: Domain-specific error types with Unwrap() and Is()
type QueryError struct {
    Code    string
    Message string
    Cause   error
}

func (e *QueryError) Error() string {
    if e.Cause != nil {
        return fmt.Sprintf("query error [%s]: %s: %v", e.Code, e.Message, e.Cause)
    }
    return fmt.Sprintf("query error [%s]: %s", e.Code, e.Message)
}

func (e *QueryError) Unwrap() error {
    return e.Cause
}

// Is() enables errors.Is() and errors.As() checks
func (e *QueryError) Is(target error) bool {
    if t, ok := target.(*QueryError); ok {
        return t.Code == e.Code
    }
    return false
}

// Usage
func NewQueryError(code, message string, cause error) *QueryError {
    return &QueryError{
        Code:    code,
        Message: message,
        Cause:   cause,
    }
}
```

**Validation Errors with Is() Method**
```go
// ✅ GOOD: Validation errors that support errors.Is()
type ValidationError struct {
    Field   string
    Message string
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("validation error [%s]: %s", e.Field, e.Message)
}

func (e *ValidationError) Is(target error) bool {
    if t, ok := target.(*ValidationError); ok {
        return t.Field == e.Field
    }
    return false
}

// Usage in tests
if errors.Is(err, &ValidationError{Field: "sql"}) {
    // Handle SQL validation error
}
```

**Error Wrapping with Context**
```go
// ✅ GOOD: Error wrapping with context
func (s *Service) ProcessPayment(ctx context.Context, req *PaymentRequest) error {
    if err := s.validate(req); err != nil {
        return fmt.Errorf("payment validation failed: %w", err)
    }
    
    payment, err := s.repo.Create(ctx, req)
    if err != nil {
        return fmt.Errorf("failed to create payment for order %s: %w", req.OrderID, err)
    }
    
    return nil
}
```

**Ignored Errors (Best Effort Operations)**
```go
// ✅ GOOD: Explicitly document why errors are ignored
_, err := s.cacheClient.UpdateOne(ctx, filter, update)
if err != nil {
    // Explicitly ignored: cache writes are non-blocking (best effort)
    // In production, use structured logging: logger.Warn("cache write failed", "error", err)
    _ = err
}

// ✅ GOOD: Response already committed
func (h *Handler) writeJSON(w http.ResponseWriter, status int, data interface{}) {
    w.WriteHeader(status)
    if err := json.NewEncoder(w).Encode(data); err != nil {
        // Explicitly ignored: response already committed
        _ = err
    }
}
```

**Error Checking Pattern**
```go
// ✅ GOOD: Explicit error handling
result, err := operation()
if err != nil {
    return fmt.Errorf("operation failed: %w", err)
}
// Use result

// ❌ BAD: Ignored error without comment
result, _ := operation()  // Dangerous!

// ✅ ACCEPTABLE: Explicitly ignored with comment
result, err := operation()
if err != nil {
    // Explicitly ignored: non-critical operation, best effort
    _ = err
}
```

### 5. Context Usage

**Context Propagation**
- Always accept `context.Context` as first parameter in functions that do I/O
- Propagate context through call chains
- Use context for cancellation, timeouts, and request-scoped values
- Never use `context.Background()` in async operations without timeout

**Context in Synchronous Operations**
```go
// ✅ GOOD: Context as first parameter
func (s *Service) ProcessOrder(ctx context.Context, order *Order) error {
    // Pass context through
    if err := s.repo.Save(ctx, order); err != nil {
        return err
    }
    
    // Use context for timeouts
    ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
    defer cancel()
    
    return s.producer.Publish(ctx, order)
}

// ❌ BAD: Missing context
func (s *Service) ProcessOrder(order *Order) error {
    return s.repo.Save(order)  // No cancellation, no timeout control
}
```

**Context in Async Operations (Critical Pattern)**
```go
// ❌ BAD: Using context.Background() in goroutines (can leak)
go func() {
    s.cacheResult(context.Background(), sql, result, ttl)  // No timeout!
}()

// ✅ GOOD: Use context with timeout in async operations
go func() {
    cacheCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()
    s.cacheResult(cacheCtx, sql, result, ttl)
}()

// ✅ GOOD: Cleanup operations with timeout
if time.Now().After(cachedDoc.ExpiresAt) {
    go func() {
        cleanupCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
        defer cancel()
        _, err := s.cacheClient.DeleteOne(cleanupCtx, filter)
        if err != nil {
            // Non-critical: cleanup failure is acceptable
            _ = err // Explicitly ignored: cleanup is best effort
        }
    }()
    return nil, nil
}
```

**Context in HTTP Clients**
```go
// ✅ GOOD: Respect timeout parameter with context
func (c *HTTPClient) ExecuteQuery(ctx context.Context, sql string, timeout time.Duration) (interface{}, error) {
    // Create context with timeout (respect the provided timeout)
    queryCtx, cancel := context.WithTimeout(ctx, timeout)
    defer cancel()
    
    req, err := http.NewRequestWithContext(queryCtx, "POST", url, body)
    if err != nil {
        return nil, fmt.Errorf("create request: %w", err)
    }
    
    resp, err := c.httpClient.Do(req)
    // ...
}
```

### 6. Resource Management

**Defer for Cleanup**
- Always use `defer` for resource cleanup (files, connections, locks)
- Defer in the same function where resource is acquired

**Example:**
```go
// ✅ GOOD: Defer cleanup
func ProcessFile(path string) error {
    file, err := os.Open(path)
    if err != nil {
        return err
    }
    defer file.Close()  // Always closes
    
    // Use file...
    return nil
}

// ✅ GOOD: Defer with error handling
func (s *Service) Process(ctx context.Context) error {
    tx, err := s.db.BeginTx(ctx, nil)
    if err != nil {
        return err
    }
    defer func() {
        if err != nil {
            tx.Rollback()
        } else {
            tx.Commit()
        }
    }()
    
    // Use tx...
    return nil
}
```

### 7. Concurrency Safety

**Immutable by Default**
- Prefer immutable data structures
- Use channels for communication between goroutines
- Protect shared mutable state with mutexes
- Use bounded concurrency (worker pools) for parallel operations

**Per-entity / per-key updates (read-modify-write):**
- When multiple goroutines can update the **same** entity (e.g. cache or DB row keyed by `user_id`, `order_id`), use **per-entity or per-key locking** so concurrent updates for the same entity are serialized. A single global lock serializes everything; a sharded lock (e.g. `hash(entity_id) % N` mutexes) serializes only same-entity updates and preserves throughput across entities.
- Without per-entity locking, concurrent read-modify-write for the same key can cause lost updates (one goroutine’s write overwrites another’s). This is a **logical** race; the `-race` detector may not report it.
- Design so that concurrency tests can exercise **same-key** scenarios: tests that use different keys per goroutine do not catch same-entity races.

**Example:**
```go
// ✅ GOOD: Immutable struct
type Config struct {
    Host string
    Port int
}

// ✅ GOOD: Thread-safe with mutex
type SafeCounter struct {
    mu    sync.RWMutex
    count int
}

func (c *SafeCounter) Increment() {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.count++
}

func (c *SafeCounter) Value() int {
    c.mu.RLock()
    defer c.mu.RUnlock()
    return c.count
}

// ✅ GOOD: Channel-based communication
func ProcessItems(items []Item) <-chan Result {
    results := make(chan Result)
    go func() {
        defer close(results)
        for _, item := range items {
            results <- process(item)
        }
    }()
    return results
}

// ✅ GOOD: Bounded concurrency with worker pool (Kubernetes pattern)
func (e *Executor) ExecuteParallel(ctx context.Context, tasks []Task, timeout time.Duration) ([]Result, error) {
    queryCtx, cancel := context.WithTimeout(ctx, timeout)
    defer cancel()
    
    taskChan := make(chan Task, len(tasks))
    resultChan := make(chan Result, len(tasks))
    
    // Send tasks
    for _, task := range tasks {
        taskChan <- task
    }
    close(taskChan)
    
    // Start workers (bounded concurrency pattern)
    var wg sync.WaitGroup
    workerCount := e.maxWorkers
    if workerCount > len(tasks) {
        workerCount = len(tasks) // Don't create more workers than tasks
    }
    
    for i := 0; i < workerCount; i++ {
        wg.Add(1)
        go func(workerID int) {  // ✅ Pass workerID to avoid closure issues
            defer wg.Done()
            for task := range taskChan {
                result := processTask(queryCtx, task)
                resultChan <- result
            }
        }(i)
    }
    
    // Wait for all workers to complete
    go func() {
        wg.Wait()
        close(resultChan)
    }()
    
    // Collect results
    results := make([]Result, 0, len(tasks))
    for result := range resultChan {
        results = append(results, result)
    }
    
    return results, nil
}

// ❌ BAD: Goroutine closure issue (captures loop variable)
for i := 0; i < workerCount; i++ {
    go func() {
        // i is captured by closure - all goroutines see same value!
        process(i)
    }()
}
```

### 8. Designing for Testability

**Testability is a First-Class Design Concern**

Code should be designed to be easily testable from the start. This means:
- Dependencies injected via interfaces (not hardcoded)
- Functions are pure when possible (no hidden state)
- Side effects are explicit and mockable
- Business logic separated from infrastructure concerns

**Example: Testable Design**
```go
// ✅ GOOD: Testable - dependencies injected via interface
type Service struct {
    repo     Repository  // Interface, not concrete type
    producer EventProducer
    logger   Logger
}

func NewService(repo Repository, producer EventProducer, logger Logger) *Service {
    return &Service{repo: repo, producer: producer, logger: logger}
}

func (s *Service) ProcessOrder(ctx context.Context, order *Order) error {
    // Business logic that can be tested with mocked dependencies
    if err := s.repo.Save(ctx, order); err != nil {
        return err
    }
    return s.producer.Publish(ctx, order)
}

// ❌ BAD: Not testable - hardcoded dependencies
type Service struct {
    db *sql.DB  // Concrete type, hard to mock
}

func (s *Service) ProcessOrder(order *Order) error {
    // Cannot test without real database
    _, err := s.db.Exec("INSERT INTO orders...")
    return err
}
```

**Interface-Based Design for Testing**
- Define interfaces for external dependencies (databases, APIs, file systems)
- Use small, focused interfaces (easier to mock)
- Accept interfaces, return structs

**Example:**
```go
// ✅ GOOD: Small interface, easy to mock
type PaymentProcessor interface {
    Process(ctx context.Context, amount Money) error
}

// In tests, create mock implementation
type mockPaymentProcessor struct {
    processFunc func(ctx context.Context, amount Money) error
}

func (m *mockPaymentProcessor) Process(ctx context.Context, amount Money) error {
    return m.processFunc(ctx, amount)
}
```

**Separate Business Logic from Infrastructure**
- Business logic should be pure functions when possible
- Infrastructure concerns (DB, HTTP, file I/O) should be in separate layers
- Test business logic without infrastructure

**Example:**
```go
// ✅ GOOD: Business logic separated
func ValidateOrder(order *Order) error {
    // Pure function - easy to test
    if order.Amount <= 0 {
        return ErrInvalidAmount
    }
    return nil
}

// Infrastructure layer uses business logic
func (s *Service) CreateOrder(ctx context.Context, order *Order) error {
    if err := ValidateOrder(order); err != nil {
        return err
    }
    return s.repo.Save(ctx, order)  // Infrastructure concern
}
```

### 9. Testing Practices

**Table-Driven Tests**
- Use table-driven tests for multiple test cases
- Test names should be descriptive
- Test both success and error cases
- Include edge cases and boundary conditions

**Example:**
```go
// ✅ GOOD: Table-driven test
func TestValidatePayment(t *testing.T) {
    tests := []struct {
        name    string
        payment *Payment
        wantErr error
    }{
        {
            name: "valid payment",
            payment: &Payment{Amount: 100, Currency: "USD"},
            wantErr: nil,
        },
        {
            name: "negative amount",
            payment: &Payment{Amount: -10, Currency: "USD"},
            wantErr: ErrInvalidAmount,
        },
        {
            name: "zero amount",
            payment: &Payment{Amount: 0, Currency: "USD"},
            wantErr: ErrInvalidAmount,
        },
        {
            name: "missing currency",
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

**Mocking Dependencies**
- Use interfaces for dependencies
- Create simple mock implementations for tests
- Consider using testing libraries (testify/mock, gomock) for complex mocks

**Example:**
```go
// ✅ GOOD: Mock implementation
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

**Test Helpers**
- Extract common test setup into helpers
- Use test fixtures for complex data
- Create builder functions for test data

**Example:**
```go
// ✅ GOOD: Test helper
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
```

**Testing Error Cases**
- Always test error paths
- Test both expected errors and unexpected errors
- Verify error messages and error types

**Example:**
```go
func TestService_ProcessOrder_ErrorCases(t *testing.T) {
    tests := []struct {
        name        string
        setupMock   func(*mockRepository)
        wantErr     error
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
            name: "context cancelled",
            setupMock: func(m *mockRepository) {},
            wantErr: context.Canceled,
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            mockRepo := &mockRepository{}
            tt.setupMock(mockRepo)
            service := NewService(mockRepo, nil, nil)
            
            ctx, cancel := context.WithCancel(context.Background())
            cancel()  // Cancel immediately
            
            err := service.ProcessOrder(ctx, &Order{})
            if !errors.Is(err, tt.wantErr) {
                t.Errorf("ProcessOrder() error = %v, wantErr %v", err, tt.wantErr)
            }
        })
    }
}
```

### 10. Code Organization and Naming

**Naming Conventions**
- Use short, descriptive names
- Package names: lowercase, single word
- Exported names: PascalCase
- Unexported names: camelCase
- Interfaces: end with `-er` when appropriate (Reader, Writer)

**Function Names**
- Should be verbs or verb phrases
- Getters: no `Get` prefix (use `Name()`, not `GetName()`)
- Setters: `Set` prefix is acceptable

**Example:**
```go
// ✅ GOOD: Clear naming
type User struct {
    name  string
    email string
}

func (u *User) Name() string { return u.name }  // Getter
func (u *User) SetName(name string) { u.name = name }  // Setter

// ✅ GOOD: Verb-based function names
func ProcessOrder(ctx context.Context, order *Order) error
func ValidatePayment(payment *Payment) error
func CalculateTotal(items []Item) Money
```

### 11. DRY (Don't Repeat Yourself)

**Extract Common Logic**
- Identify repeated patterns and extract functions
- Use helper functions for common operations
- Avoid copy-paste code

**Example:**
```go
// ❌ BAD: Repeated validation logic
func CreateOrder(order *Order) error {
    if order.Amount <= 0 {
        return errors.New("amount must be positive")
    }
    if order.Currency == "" {
        return errors.New("currency required")
    }
    // ... more validation
}

func UpdateOrder(order *Order) error {
    if order.Amount <= 0 {
        return errors.New("amount must be positive")
    }
    if order.Currency == "" {
        return errors.New("currency required")
    }
    // ... same validation repeated
}

// ✅ GOOD: Extracted validation
func ValidateOrder(order *Order) error {
    if order.Amount <= 0 {
        return ErrInvalidAmount
    }
    if order.Currency == "" {
        return ErrCurrencyRequired
    }
    return nil
}

func CreateOrder(order *Order) error {
    if err := ValidateOrder(order); err != nil {
        return err
    }
    // ... create logic
}
```

### 12. SOLID Principles in Go

**Single Responsibility**
- Each type/function should have one reason to change
- Separate concerns: business logic, data access, presentation

**Open/Closed**
- Open for extension via interfaces
- Closed for modification of existing code

**Liskov Substitution**
- Implementations must satisfy interface contracts completely
- No surprises in behavior

**Interface Segregation**
- Small, focused interfaces (see Interface Design section)

**Dependency Inversion**
- Depend on abstractions (interfaces), not concretions
- High-level modules don't depend on low-level modules

**Example:**
```go
// ✅ GOOD: Dependency inversion
type Repository interface {
    Save(ctx context.Context, entity *Entity) error
}

type Service struct {
    repo Repository  // Depends on interface, not implementation
}

// Can swap implementations without changing Service
type PostgresRepository struct {}
type InMemoryRepository struct {}
```

### 13. Avoid Overengineering

**Keep It Simple**
- Start with the simplest solution that works
- Don't add complexity until it's actually needed
- Avoid building abstractions "just in case"
- Prefer explicit code over clever abstractions
- Don't add features that aren't required

**When to Add Complexity:**
- When you have a concrete, current need (not hypothetical future needs)
- When the simple solution is causing real problems
- When complexity reduces maintenance burden (not increases it)

**Example:**
```go
// ✅ GOOD: Simple, direct solution
func LoadTableConfigs(configPath string) ([]TableConfig, error) {
    if configPath != "" {
        return loadTableConfigsFromFile(configPath)
    }
    return loadTableConfigsFromEnv()
}

// ❌ BAD: Overengineered - auto-detection, multiple formats, env var fallback
func LoadTableConfigs(configPath string) ([]TableConfig, error) {
    if configPath == "" {
        configPath = getEnv("TABLE_CONFIG_PATH", "")
    }
    if configPath != "" {
        // Try YAML, then JSON, then auto-detect...
        if strings.HasSuffix(path, ".yaml") || strings.HasSuffix(path, ".yml") {
            // ...
        } else if strings.HasSuffix(path, ".json") {
            // ...
        } else {
            // Try both...
        }
    }
    return loadTableConfigsFromEnv()
}
```

**YAGNI Principle (You Aren't Gonna Need It)**
- Don't implement features until they're actually needed
- Don't create abstractions for hypothetical future requirements
- Don't add configuration options "just in case"
- Don't support multiple formats when one is sufficient

**KISS Principle (Keep It Simple, Stupid)**
- Prefer straightforward implementations
- Use standard library when possible
- Avoid unnecessary dependencies
- Write code that's easy to understand and maintain

### 14. Performance Considerations

**Avoid Premature Optimization**
- Write clear, correct code first
- Profile before optimizing
- Optimize hot paths only
- Follow "mechanical sympathy" principles (understand hardware behavior)

**Efficient Patterns**
- Prefer `make()` with capacity for slices when size is known
- Use `strings.Builder` for string concatenation in loops
- Reuse buffers when possible
- Use nil slices instead of empty slices to avoid unnecessary allocation
- Preallocate maps when size is known: `make(map[string]int, size)`

**Example:**
```go
// ✅ GOOD: Pre-allocate slice capacity
items := make([]Item, 0, len(rawItems))
for _, raw := range rawItems {
    items = append(items, parseItem(raw))
}

// ✅ GOOD: Use Builder for string concatenation
var b strings.Builder
for _, s := range strings {
    b.WriteString(s)
}
result := b.String()
```

## Common Mistakes

### ❌ Global State
```go
// BAD
var globalDB *sql.DB

// GOOD: Inject dependencies
type Service struct {
    db *sql.DB
}
```

### ❌ Ignoring Errors
```go
// BAD: Ignored without explanation
result, _ := operation()

// BAD: Ignored without comment
_, err := cache.Write(ctx, data)
if err != nil {
    _ = err  // Why is this ignored?
}

// GOOD: Documented ignored error
result, err := operation()
if err != nil {
    return fmt.Errorf("operation failed: %w", err)
}

// GOOD: Explicitly documented why error is ignored
_, err := cache.Write(ctx, data)
if err != nil {
    // Explicitly ignored: cache writes are non-blocking (best effort)
    // In production, use structured logging: logger.Warn("cache write failed", "error", err)
    _ = err
}
```

### ❌ Panic for Business Logic
```go
// BAD
if amount < 0 {
    panic("invalid amount")
}

// GOOD
if amount < 0 {
    return ErrInvalidAmount
}
```

### ❌ Missing Context
```go
// BAD: Missing context
func Process(data Data) error

// BAD: Using context.Background() in async operations (can leak)
go func() {
    s.cache.Write(context.Background(), data)
}()

// GOOD: Context as first parameter
func Process(ctx context.Context, data Data) error

// GOOD: Context with timeout in async operations
go func() {
    cacheCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()
    s.cache.Write(cacheCtx, data)
}()
```

### ❌ Large Interfaces
```go
// BAD: Too many responsibilities
type Service interface {
    ProcessOrder()
    SendEmail()
    GenerateReport()
    ProcessPayment()
}

// GOOD: Focused interfaces
type OrderProcessor interface {
    ProcessOrder(ctx context.Context, order *Order) error
}
```

### ❌ Exposing Implementation Details
```go
// BAD: Exports implementation
func NewService() *PostgresService {
    return &PostgresService{...}
}

// GOOD: Returns interface
func NewService() Service {
    return &postgresService{...}
}
```

### ❌ Using Generic Package Names
```go
// BAD: Generic, unclear purpose
package utils
package helpers
package common

// GOOD: Specific, domain-focused
package payment
package validation
package metrics
```

### ❌ Missing Context in I/O Operations
```go
// BAD: No cancellation or timeout control
func (r *Repository) Save(entity *Entity) error {
    return r.db.Exec("INSERT ...")
}

// GOOD: Context for cancellation and timeouts
func (r *Repository) Save(ctx context.Context, entity *Entity) error {
    return r.db.ExecContext(ctx, "INSERT ...")
}
```

## Real-World Patterns from Production Repositories

### Domain-Driven, Data-Oriented Architecture (ardanlabs/service)

**Key Principles:**
- Start simple, evolve complexity only when needed
- Domain-driven design with clear boundaries
- Data-oriented design for performance
- Minimal dependencies, maximum clarity
- Explicit about intentions (internal vs pkg)

**Structure Pattern:**
```
app/              # Application initialization and wiring
business/         # Business logic layer
foundation/       # Infrastructure concerns (logging, metrics, etc.)
api/              # API definitions
```

### Kubernetes Coding Conventions

**Package Naming:**
- Avoid uppercase, underscores, or dashes in package names
- Consider parent directory when naming (e.g., `pkg/controllers/autoscaler/foo.go` uses `package autoscaler`)
- Use concise interface names with package context (e.g., `storage.Interface` not `storage.StorageInterface`)

**Locks:**
- Always name locks explicitly: `lock sync.Mutex` (never embed)
- Document lock ordering to prevent deadlocks

**Testing:**
- Table-driven tests for multiple scenarios
- All new packages must include unit tests
- Tests must pass on macOS and Windows
- Integration tests for significant features

### Standard Project Layout Patterns

**When to Use Each Directory:**
- `cmd/` - One directory per executable (e.g., `cmd/api`, `cmd/migrate`)
- `internal/` - Private code (enforced by Go compiler - cannot be imported by external packages)
- `pkg/` - Public library code (use sparingly, prefer `internal/` when possible)
- `api/` - API contracts (OpenAPI specs, protobuf definitions)
- `configs/` - Configuration templates
- `scripts/` - Build and deployment scripts

**Avoid:**
- `/src` directory (Java pattern, not idiomatic Go)
- Generic package names (`utils`, `helpers`, `common`)
- Package sprawl (find appropriate subdirectories)

## Quick Reference Checklist

When implementing Go code, verify:

- [ ] Package has single, clear responsibility
- [ ] Interfaces are small and focused
- [ ] Dependencies injected via constructors
- [ ] All errors handled explicitly
- [ ] Ignored errors are documented with explanation
- [ ] Domain-specific error variables defined (Kubernetes pattern)
- [ ] Error types implement `Unwrap()` and `Is()` methods
- [ ] Context propagated through call chains
- [ ] Context used with timeout in async operations (not `context.Background()`)
- [ ] Resources cleaned up with defer
- [ ] Thread-safe if shared state exists
- [ ] Per-entity or per-key locking used when multiple goroutines update the same entity (e.g. cache/DB keyed by user_id); avoid same-entity lost updates
- [ ] Bounded concurrency used for parallel operations
- [ ] Goroutine closures properly handle loop variables
- [ ] Code is designed for testability (dependencies injected)
- [ ] Tests are table-driven where appropriate
- [ ] Error cases are tested
- [ ] Mocks are used for external dependencies
- [ ] Names are clear and follow conventions
- [ ] No repeated code (DRY)
- [ ] SOLID principles applied
- [ ] No global state or singletons
- [ ] Exported only what's necessary
- [ ] OpenAPI spec created for REST APIs (if applicable)
- [ ] OpenAPI spec includes all endpoints, schemas, examples, and error responses

## Reference Repositories

**Production-Grade Examples:**
- **[ardanlabs/service](https://github.com/ardanlabs/service)** (3.9k stars) - Starter-kit for writing services in Go using Kubernetes. Demonstrates Domain Driven, Data Oriented Architecture with minimal dependencies and idiomatic Go.
- **[golang-standards/project-layout](https://github.com/golang-standards/project-layout)** (55k stars) - Standard Go project layout with common patterns for organizing Go applications.

**Clean Architecture Examples:**
- **[DoWithLogic/golang-clean-architecture](https://github.com/DoWithLogic/golang-clean-architecture)** - Clean architecture with SOLID principles implementation
- **[gothinkster/golang-gin-realworld-example-app](https://github.com/gothinkster/golang-gin-realworld-example-app)** - Real-world application demonstrating production-quality patterns

**Key Patterns from These Repositories:**
- Domain-driven design with clear package boundaries
- Minimal dependencies, maximum clarity
- Explicit error handling throughout
- Domain-specific error variables (not just error types)
- Error types with `Unwrap()` and `Is()` methods
- Context propagation for cancellation and timeouts
- Context with timeout in async operations (not `context.Background()`)
- Bounded concurrency with worker pools
- Proper goroutine closure handling
- Dependency injection via constructors
- Internal packages for private code
- Clear separation of concerns
- Documented ignored errors (non-blocking operations)

### 15. API Documentation with OpenAPI

**OpenAPI Specification (OAS)**
- Always create OpenAPI 3.0+ specifications for REST APIs
- Keep OpenAPI spec in sync with code (update when endpoints change)
- Use OpenAPI for API documentation, client generation, and testing
- Place OpenAPI specs in `api/` or `docs/api/` directory

**When to Create OpenAPI Spec:**
- When creating a new REST API service
- When adding new endpoints to existing APIs
- When modifying request/response schemas
- Before API review or client integration

**OpenAPI Best Practices:**
```yaml
# ✅ GOOD: Complete OpenAPI spec with examples
openapi: 3.0.3
info:
  title: Service Name API
  version: 1.0.0
  description: |
    Clear description of what the API does.
    Include key features, authentication, rate limiting.
paths:
  /api/v1/resource:
    post:
      summary: Clear, concise summary
      description: Detailed description with examples
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/RequestSchema'
            examples:
              example1:
                summary: Example 1
                value: {...}
      responses:
        '200':
          description: Success response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ResponseSchema'
              examples:
                success:
                  value: {...}
        '400':
          description: Validation error
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
components:
  schemas:
    RequestSchema:
      type: object
      required: [field1]
      properties:
        field1:
          type: string
          description: Clear field description
          example: "example value"
```

**Required OpenAPI Elements:**
- **Info**: Title, version, description, contact
- **Servers**: Base URLs for different environments
- **Paths**: All endpoints with HTTP methods
- **Schemas**: Request/response models with validation rules
- **Examples**: At least one example per endpoint
- **Error Responses**: All possible error codes (400, 401, 404, 500, etc.)
- **Security**: Authentication schemes (even if not implemented yet)

**Schema Definition Best Practices:**
```yaml
# ✅ GOOD: Detailed schema with validation
components:
  schemas:
    UserRequest:
      type: object
      required:
        - email
        - name
      properties:
        email:
          type: string
          format: email
          description: User email address
          example: "user@example.com"
        name:
          type: string
          minLength: 1
          maxLength: 100
          description: User full name
          example: "John Doe"
        age:
          type: integer
          minimum: 0
          maximum: 150
          description: User age (optional)
          example: 30
```

**Response Documentation:**
- Document all HTTP status codes (200, 201, 400, 401, 403, 404, 500, etc.)
- Include error response schemas
- Provide examples for both success and error cases
- Document response headers (X-Request-ID, pagination headers, etc.)

**Code Generation:**
- Use OpenAPI spec to generate client SDKs
- Use OpenAPI spec for server stub generation (optional)
- Use OpenAPI spec for API testing and validation

**File Organization:**
```
api/
  openapi.yaml          # Main OpenAPI spec
  openapi.prod.yaml     # Production-specific overrides (optional)
docs/
  api/
    README.md           # API documentation
    examples.md         # Additional examples
```

**Example: Complete OpenAPI Endpoint**
```yaml
/api/v1/users:
  post:
    tags: [Users]
    summary: Create a new user
    description: |
      Creates a new user account.
      
      **Validation:**
      - Email must be unique
      - Name must be between 1-100 characters
      
      **Response:**
      - 201: User created successfully
      - 400: Validation error
      - 409: Email already exists
    operationId: createUser
    requestBody:
      required: true
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/CreateUserRequest'
          examples:
            valid_user:
              summary: Valid user creation
              value:
                email: "user@example.com"
                name: "John Doe"
                age: 30
    responses:
      '201':
        description: User created successfully
        headers:
          Location:
            description: URL of the created user
            schema:
              type: string
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UserResponse'
      '400':
        description: Validation error
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ErrorResponse'
            examples:
              invalid_email:
                value:
                  success: false
                  error: "validation error [email]: invalid email format"
      '409':
        description: Email already exists
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ErrorResponse'
```

**Tools and Resources:**
- [OpenAPI Generator](https://openapi-generator.tech/) - Generate clients/servers from spec
- [Swagger UI](https://swagger.io/tools/swagger-ui/) - Interactive API documentation
- [Redoc](https://github.com/Redocly/redoc) - Beautiful API documentation
- [Stoplight](https://stoplight.io/) - API design and documentation platform

**Integration with Go Code:**
- Keep request/response structs in sync with OpenAPI schemas
- Use struct tags for JSON serialization matching OpenAPI
- Document struct fields with comments matching OpenAPI descriptions
- Generate Go structs from OpenAPI (optional, using openapi-generator)

## Additional Resources

**Official Go Documentation:**
- [Effective Go](https://go.dev/doc/effective_go) - The foundational guide to writing idiomatic Go
- [Go Code Review Comments](https://go.dev/wiki/CodeReviewComments) - Common review comments and style issues
- [Organizing a Go module](https://go.dev/doc/modules/layout) - Official guidance on module structure

**Community Standards:**
- [Kubernetes Coding Conventions](https://kubernetes.dev/docs/guide/coding-convention) - Patterns from Kubernetes codebase
- [Standard Go Project Layout](https://github.com/golang-standards/project-layout) - Community-driven project structure
- [Go Style Guide (Google)](https://google.github.io/styleguide/go/decisions) - Extended style guide
- [OpenAPI Specification](https://swagger.io/specification/) - Official OpenAPI 3.0 specification

**Best Practices Talks:**
- [GopherCon EU 2018: Best Practices for Industrial Programming](https://www.youtube.com/watch?v=PTE4VJIdHPg)
- [GopherCon 2018: How Do You Structure Your Go Apps](https://www.youtube.com/watch?v=oL6JBUk6tj0)
