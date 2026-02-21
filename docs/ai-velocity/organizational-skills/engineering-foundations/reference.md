# Engineering Foundations — Reference

Detailed reference for the engineering-foundations skill. Use when deeper context is needed.

## Architectural Decision Records (ADR)

When making significant architectural decisions:

1. **Context**: What is the situation?
2. **Decision**: What did we decide?
3. **Consequences**: What are the trade-offs?

## Trade-off Framework

| Dimension | Favor simplicity when | Favor complexity when |
|-----------|------------------------|-------------------------|
| Performance | Not a bottleneck | Proven hot path |
| Flexibility | Requirements stable | Multiple consumers, evolving needs |
| Consistency | Single domain | Cross-service, compliance |
| Coupling | Same team, same deploy | Clear ownership boundaries |

## Common Anti-Patterns

- **God object**: One class/module doing too much → split by responsibility
- **Shotgun surgery**: One change requires edits in many places → extract shared logic
- **Feature envy**: Method uses another object's data more than its own → move behavior
- **Primitive obsession**: Using primitives where value objects would clarify → introduce domain types
- **Anemic domain**: Domain objects are data bags with logic elsewhere → move logic into domain

## Testing Heuristics

**Right-BICEP** (for unit tests):
- **R**ight: Are results correct?
- **B**oundary: Are boundaries correct?
- **I**nverse: Can we verify inverse relationships?
- **C**ross-check: Can we verify with another method?
- **E**rror: Are error conditions handled?
- **P**erformance: Are performance constraints met?

**CORRECT** (for boundary conditions):
- **C**onformance: Does value conform to expected format?
- **O**rdering: Is the ordered collection ordered correctly?
- **R**ange: Is value within permissible range?
- **R**eference: Does code reference external state?
- **E**xistence: Does value exist (null, zero, empty)?
- **C**ardinality: Are there exactly right number of values?
- **T**ime: Is timing/order of operations correct?

## Scale Readiness Signals

Add scaling considerations when:
- User/request count exceeds 10x current
- Data volume doubles within 6 months
- P99 latency matters for UX
- Multi-region or multi-tenant is on roadmap

## Data and Privacy

### PII Handling

- **Identify PII**: Names, emails, phone numbers, addresses, IDs that link to individuals.
- **Minimize**: Collect only what's needed; avoid logging PII.
- **Protect**: Encrypt at rest and in transit; restrict access.
- **Retention**: Define how long data is kept; support deletion when required.

### Compliance

- **GDPR, CCPA, etc.**: Know applicable regulations for your users and regions.
- **Right to deletion**: Design for data deletion and export.
- **Consent**: Record and respect user consent for data use.

## Dependencies and Upgrades

### Choosing Dependencies

- **Maturity**: Active maintenance, reasonable release cadence.
- **License**: Compatible with your use (Apache, MIT, etc.).
- **Security**: Known vulnerabilities; response to CVEs.
- **Fit**: Solves a real problem; avoid "nice to have" deps.

### Upgrade Cadence

- **Patch**: Apply security patches promptly.
- **Minor**: Upgrade periodically; test before production.
- **Major**: Plan; check breaking changes and migration path.

### Supply Chain

- Scan for known vulnerabilities (e.g., Snyk, Dependabot).
- Pin versions in lockfiles; review dependency updates in PRs.
