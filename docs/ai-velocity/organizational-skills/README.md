# Organizational Skills

**Base layer** skills for the engineering team. Claude Code should reference these as foundational guidance before applying language- or domain-specific skills.

## Purpose

These skills define:

- **Coding standards** — How we write code
- **Architectural principles** — How we design systems
- **Testing practices** — How we verify correctness
- **Robustness** — How we handle edge cases
- **Scalability** — How we build for growth
- **Security** — Input validation, secrets, auth
- **Operational readiness** — Health checks, graceful shutdown, config
- **Code review** — What to check, review culture
- **Technical debt** — When to take it, how to pay it down
- **Documentation** — When and what to document

## Skills

| Skill | Description |
|-------|-------------|
| [Engineering Foundations](engineering-foundations/) | Coding standards, architecture, testing, edge cases, scalability, security, operational readiness, code review, technical debt, documentation |
| [Testing Fundamentals](testing-fundamentals/) | Test pyramid, mocking strategy, when to use unit vs integration vs E2E, avoiding over-mocking |

## Usage

Reference as base layer before task-specific skills:

```
@engineering-foundations/SKILL.md Design a new payment service
@engineering-foundations/SKILL.md @golang-coding-standards/SKILL.md Implement this feature in Go
```

## Relationship to Other Skills

```
organizational-skills/     ← Base layer (this directory)
  engineering-foundations/
  testing-fundamentals/

golang-coding-standards/   ← Language-specific (extends foundations)
java-testing/              ← Extends testing-fundamentals
golang-unit-testing/
prometheus-instrumentation/
...
```
