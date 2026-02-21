# 🚨 Metrics Instrumentation Guardrails

**CRITICAL: Read this BEFORE adding any metrics instrumentation.**

---

## The Zero Mutation Rule

**Instrumentation is PURELY ADDITIVE. It NEVER changes functional behavior.**

If removing all metrics would change your program's behavior, your instrumentation is **broken**.

---

## The 4 Commandments

When adding metrics to ANY method:

### 1. ✅ ONLY Add Metrics Calls
- Add `metrics.record*()` calls
- Add `var timerSample = metrics.start()` and timer stop calls
- Add try-catch blocks ONLY for metrics recording
- **DO NOT** modify existing business logic, validation, error handling, or control flow

### 2. ✅ Preserve ALL Existing Behavior
- Method signatures must remain identical
- Return values must be identical
- Exception types and messages must be identical
- Side effects must be identical
- All existing try-catch blocks must be preserved

### 3. ✅ Wrap, Don't Replace
- Wrap existing code in try-catch for metrics
- Preserve all existing exception handling
- **DO NOT** change exception types
- **DO NOT** add new validation or early returns
- **DO NOT** modify existing control flow

### 4. ✅ Verification Checklist
Before submitting ANY instrumentation change:
- [ ] All existing business logic preserved
- [ ] All existing exception handling preserved
- [ ] Method signatures unchanged
- [ ] Return values unchanged
- [ ] Only metrics calls added
- [ ] No new validation or early returns
- [ ] No changes to control flow
- [ ] No changes to exception types

---

## Common Mistakes (From RCA)

### ❌ MISTAKE #1: Changed Exception Handling
```java
// WRONG: Wrapping exception in different type
public Order createOrder(CreateOrderRequest request) {
    try {
        Order order = doCreateOrder(request);
        metrics.recordOrderCreate(orderType, timerSample, true);
        return order;
    } catch (IllegalArgumentException e) {
        // ❌ WRONG: Changed exception type
        throw new AppServerException(e);
    }
}
```

**Detection:** If you see `throw new SomeException(e)` and it wasn't there before, you violated the rule.

---

### ❌ MISTAKE #2: Added Validation Logic
```java
// WRONG: Added null check that didn't exist
public Order createOrder(CreateOrderRequest request) {
    if (request == null) {
        metrics.recordOrderCreate(orderType, timerSample, false);
        return null; // ❌ WRONG: Changed behavior
    }
    // ...
}
```

**Detection:** If you added `if (request == null)`, `if (isEmpty())`, or ANY business validation, you violated the rule.

---

### ❌ MISTAKE #3: Changed Method Signature
```java
// WRONG: Added parameter
public Order createOrder(CreateOrderRequest request, MeterRegistry registry) {
    // ❌ WRONG: Added 'registry' parameter
}
```

**Detection:** If the number or type of parameters changed, you violated the rule.

---

### ❌ MISTAKE #4: Mid-Method Business Logic Outcomes
```java
// WRONG: Recording business outcomes mid-method
public void processTransaction() {
    if (businessCondition1) {
        result = doSomething();
        metrics.recordBusinessOutcome("success"); // ❌ WRONG
    } else {
        metrics.recordBusinessOutcome("failed"); // ❌ WRONG
    }
}
```

**Detection:** If you're recording metrics based on business conditions (not just success/failure of the entire method), you violated the rule.

---

## ✅ The CORRECT Pattern

**BEFORE (existing code):**
```java
public Order createOrder(CreateOrderRequest request) {
    Order order = doCreateOrder(request);
    return order;
}
```

**AFTER (instrumented code):**
```java
public Order createOrder(CreateOrderRequest request) {
    var timerSample = metrics.start();                    // ADDED
    try {
        Order order = doCreateOrder(request);             // UNCHANGED (exact same line)
        metrics.recordOrderCreate(orderType, timerSample, true); // ADDED
        return order;                                     // UNCHANGED
    } catch (Exception e) {
        metrics.recordOrderCreate(orderType, timerSample, false); // ADDED
        throw e;                                          // PRESERVED (same exception type)
    }
}
```

**Key observations:**
- The line `Order order = doCreateOrder(request);` is **byte-for-byte identical**
- The original method had NO try-catch → we added one ONLY for metrics
- The exception is re-thrown with **the same type** (`throw e`, not `throw new AppServerException(e)`)
- Return value is **identical** (`return order`, unchanged)
- Method signature is **identical** (same parameters, same return type)

---

## Quick Self-Check

Before you submit instrumentation changes, ask yourself:

1. **"If I deleted all the metrics.record*() lines, would the code behave identically to the original?"**
   - ✅ YES → Correct
   - ❌ NO → You violated the Zero Mutation Rule

2. **"Did I add ANY if-statements, validation checks, or early returns?"**
   - ✅ NO → Correct
   - ❌ YES → You violated the Zero Mutation Rule

3. **"Did I change ANY exception types being thrown?"**
   - ✅ NO → Correct
   - ❌ YES → You violated the Zero Mutation Rule

4. **"Did I add ANY parameters to method signatures?"**
   - ✅ NO → Correct
   - ❌ YES → You violated the Zero Mutation Rule

---

## When In Doubt

**If you're unsure whether a change violates the Zero Mutation Rule:**

1. Show the BEFORE and AFTER code side-by-side
2. Ask: "Would a unit test that tested the original method fail with my instrumented version?"
3. If YES (test would fail) → You violated the rule
4. If NO (test still passes, same inputs → same outputs → same exceptions) → You're good

---

## Reference

This guardrail enforces the **Zero Mutation Rule** from [Aspora Cortex](https://github.com/Vance-Club/aspora-cortex):

> When adding instrumentation, observability, logging, or any cross-cutting concern: **it is purely additive.** Instrumentation NEVER changes functional behavior, control flow, exception handling, or business logic. If adding a metric could alter how an error propagates, you've done it wrong.

---

**🔴 ABSOLUTE REQUIREMENT:** Every AI agent (Claude, Cursor, Codex) MUST read this file before adding metrics to ANY codebase.
