# AI Velocity

**Accelerate developer productivity** with production-grade AI skills for Claude and Cursor.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Skills](https://img.shields.io/badge/skills-8+-green.svg)](#skills)

## What is AI Velocity?

AI Velocity is a curated collection of **production-grade AI skills** that enable developers to work faster and smarter with Claude and Cursor. Each skill is battle-tested, well-documented, and follows industry best practices.

**Velocity = Speed × Direction.** These skills give you both.

## Why AI Velocity?

### The Problem

- **Repetitive tasks** consume developer time
- **Inconsistent implementations** across projects
- **Knowledge silos** prevent best practices from spreading
- **Onboarding friction** slows new team members

### The Solution

AI Velocity provides **standardized, reusable skills** that:

- ✅ **Accelerate development** by automating repetitive tasks
- ✅ **Ensure consistency** across all projects
- ✅ **Share best practices** organization-wide
- ✅ **Reduce onboarding time** with clear, documented workflows

### The Impact

- **10x faster** observability setup
- **Consistent quality** across all services
- **Reduced MTTR** with production-grade instrumentation
- **Faster onboarding** for new team members

## Skills

| Skill | Purpose | Status |
|-------|---------|--------|
| [Engineering Foundations](organizational-skills/engineering-foundations/) | Base layer: coding standards, architecture, testing, edge cases, scalability | ✅ Production |
| [Testing Fundamentals](organizational-skills/testing-fundamentals/) | Test pyramid, mocking strategy, E2E approach | ✅ Production |
| [Go Coding Standards](golang-coding-standards/) | Production-grade Go standards, SOLID, error handling, concurrency | ✅ Production |
| [Java Coding Standards](java-coding-standards/) | Production-grade Java/Spring Boot standards, SOLID, exception handling | ✅ Production |
| [Java Application Development](java-application-development/) | Domain models, SOLID, DRY, layered architecture | ✅ Production |
| [Prometheus Metrics](prometheus-instrumentation/) | Instrument Java/Go services with production-grade metrics | ✅ Production |
| [Datadog Dashboards](datadog-dashboard-creation/) | Create SRE-grade observability dashboards | ✅ Production |
| [Java Testing](java-testing/) | Unit, integration, and E2E tests; mocking strategy; JUnit 5, Mockito, RestAssured, Testcontainers | ✅ Production |
| [Alerting](alert-generation/) | Set up intelligent alerting rules | ✅ Production |
| [Git Workflow](git-workflow/) | Commit conventions, branch naming, merge vs rebase, PR checklist | ✅ Production |
| [Hackathon Advisor](hackathon-advisor/) | Grounded hackathon runbooks: stack, hour-by-hour plan, anti-patterns, skills map | ✅ Production |

## Quick Start

### For Claude.ai

1. Click the skill icon (🧩) in your chat interface
2. Upload the skill folder (e.g., `prometheus-metrics/`)
3. Start using: "Add Prometheus metrics to my service"

### For Claude Code / Cursor

1. Place skills in your config directory:
   ```bash
   mkdir -p ~/.config/claude-code/skills/
   cp -r prometheus-metrics ~/.config/claude-code/skills/
   ```

2. Reference skills or use slash-style commands:
   ```markdown
   @prometheus-instrumentation/SKILL.md Add metrics to my service
   @java-application-development/SKILL.md Design domain model for orders
   @java-testing/SKILL.md Generate tests for this class
   ```

3. **Slash-style commands** (use these phrases or reference the command docs in `commands/`):
   | Command | Use for |
   |---------|--------|
   | `/observability:metrics` | Prometheus instrumentation |
   | `/observability:dashboard` | Datadog dashboards |
   | `/observability:alerts` | Alerting rules |
   | `/dev:java` | Java app design (domain models, SOLID, DRY) |
   | `/test:java` | Java unit/integration tests |
   | `/test:java-fixtures` | Test builders and fixtures |
   | `/test:go` | Go unit tests |
   | `/standards:go` | Go coding standards |

### For Teams

1. Clone this repository
2. Share skills via your internal documentation
3. Reference skills in project READMEs

See [Usage Guide](docs/usage-guide.md) for detailed instructions.

## Architecture

```
ai-velocity/
├── README.md                    # This file
├── .claude-plugin/
│   └── plugin.json              # Plugin manifest (Claude Code / Cowork)
├── .mcp.json                    # Optional: MCP server template (copy to your config)
├── organizational-skills/       # Base layer for engineering teams
│   ├── README.md
│   ├── engineering-foundations/ # Coding standards, architecture, testing, scalability
│   │   ├── SKILL.md
│   │   └── reference.md
│   └── testing-fundamentals/    # Test pyramid, mocking strategy, E2E
│       ├── SKILL.md
│       └── reference.md
├── golang-coding-standards/     # Go coding standards
│   └── SKILL.md
├── java-coding-standards/       # Java/Spring Boot coding standards
│   ├── SKILL.md
│   └── reference.md
├── java-application-development/  # Domain models, SOLID, DRY
├── prometheus-instrumentation/  # Metrics instrumentation skill
│   └── SKILL.md
├── datadog-dashboard-creation/  # Dashboard creation skill
│   └── SKILL.md
├── java-testing/                # Unit, integration, E2E testing skill
│   ├── SKILL.md
│   └── README.md
├── alert-generation/            # Alerting setup skill
│   ├── SKILL.md
│   └── COMPLETE_EXAMPLE.md
├── git-workflow/                # Git commit conventions, branches, PR workflow
│   ├── SKILL.md
│   └── reference.md
├── hackathon-advisor/           # Hackathon project runbooks and planning
│   ├── SKILL.md
│   ├── DECISIONS.md
│   ├── GUARDRAILS.md
│   ├── README.md
│   └── references/
│       └── runbook-template.md
└── docs/                        # Documentation
    ├── usage-guide.md
    └── contributing.md
```

## Standards

All skills follow [**Aspora Cortex**](https://github.com/Vance-Club/aspora-cortex) — our shared engineering operating system. Cortex defines the principles (SOLID, Clean Architecture, Four Golden Signals, Zero Logic Mutation) that every skill enforces.

To create new skills, see the SAGE principles and skill design framework in Cortex.

## Principles

AI Velocity skills follow these principles:

1. **Production-Grade**: Battle-tested in real production environments
2. **Best Practices**: Follow industry standards (SRE, RED method, etc.)
3. **Clear Documentation**: Comprehensive guides with examples
4. **Consistent**: Standardized patterns across all skills
5. **Maintainable**: Easy to update and extend

## Contributing

We welcome contributions! See [Contributing Guide](docs/contributing.md) for:

- How to create new skills
- Quality standards
- Review process
- Code of conduct

**Quick contribution steps:**

1. Fork the repository
2. Create your skill following the [skill template](docs/contributing.md#skill-template)
3. Test across Claude.ai, Claude Code, and API
4. Submit a pull request

## Roadmap

- [ ] Kubernetes deployment skills
- [ ] Database migration skills
- [ ] API documentation skills
- [ ] Security scanning skills
- [ ] Performance optimization skills

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: See [docs/](docs/) folder
- **Issues**: Open an issue on GitHub
- **Questions**: Contact the maintainers

---

**Built with ❤️ by the Vance Engineering Team**

*Accelerate your development. Build faster. Ship better.*
