# Contributing Guide

**How to contribute skills to AI Velocity and maintain our high quality standards.**

## Overview

This guide explains how to create, submit, and maintain AI Velocity skills. We follow strict quality standards to ensure all skills are production-grade and immediately usable.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Skill Structure](#skill-structure)
3. [Quality Standards](#quality-standards)
4. [Development Process](#development-process)
5. [Review Process](#review-process)
6. [Maintenance](#maintenance)

---

## Getting Started

### Prerequisites

- Understanding of the skill's domain
- Experience with Claude.ai, Claude Code, or Cursor
- Ability to write clear, concise documentation
- Commitment to maintain the skill

### Before You Start

1. **Check existing skills** - Avoid duplicates
2. **Validate the need** - Ensure it solves a real problem
3. **Plan the structure** - Design before implementing
4. **Test thoroughly** - Verify across all platforms

---

## Skill Structure

### Required Files

Every skill must have this structure:

```
skill-name/
├── SKILL.md              # Required: Main skill file
├── README.md             # Optional: User-facing docs
├── examples/             # Optional: Example files
│   └── example.md
└── templates/           # Optional: Code templates
    └── template.java
```

### SKILL.md Format

**YAML Frontmatter (Required):**

```yaml
---
name: skill-name
description: Clear, one-sentence description of what this skill does
version: 1.0.0
author: Your Name / Team Name
tags: [tag1, tag2, tag3]
---
```

**Content Structure:**

```markdown
# Skill Name

## Purpose
Clear explanation of what this skill does and why it exists.

## When to Trigger This Skill
- Use case 1
- Use case 2
- Use case 3

## Prerequisites
- Requirement 1
- Requirement 2

## Instructions
[Detailed step-by-step instructions for Claude]

## Examples
[Real-world examples]

## Best Practices
[Guidelines and recommendations]

## Troubleshooting
[Common issues and solutions]
```

### Naming Conventions

- **Folder name:** `kebab-case` (e.g., `prometheus-metrics`)
- **Skill name:** Same as folder name
- **Descriptive:** Clear what the skill does
- **Short:** Maximum 3 words

---

## Quality Standards

### Documentation Quality

**Must have:**
- ✅ Clear purpose statement
- ✅ When to use guidance
- ✅ Step-by-step instructions
- ✅ Real-world examples
- ✅ Error handling guidance
- ✅ Best practices section

**Should have:**
- Examples for different scenarios
- Troubleshooting section
- Performance considerations
- Security implications

### Code Quality

**Generated code must:**
- ✅ Follow language best practices
- ✅ Include error handling
- ✅ Be production-ready
- ✅ Include comments
- ✅ Follow project conventions

### Testing Requirements

**Test across:**
- ✅ Claude.ai web interface
- ✅ Claude Code desktop app
- ✅ Cursor IDE
- ✅ Claude API (if applicable)

**Test scenarios:**
- ✅ Happy path
- ✅ Edge cases
- ✅ Error conditions
- ✅ Different project structures

---

## Development Process

### Step 1: Design

**Before writing code:**

1. **Define the problem** - What does this solve?
2. **Identify use cases** - When will this be used?
3. **Design the solution** - How will it work?
4. **Plan the structure** - What files are needed?

**Document your design:**
- Create a design doc
- Get feedback from team
- Iterate before implementing

### Step 2: Implementation

**Create the skill:**

1. **Set up structure:**
   ```bash
   mkdir -p skill-name/{examples,templates}
   touch skill-name/SKILL.md
   ```

2. **Write SKILL.md:**
   - Start with YAML frontmatter
   - Write clear instructions
   - Add examples
   - Include best practices

3. **Test thoroughly:**
   - Test in Claude.ai
   - Test in Claude Code
   - Test in Cursor
   - Test edge cases

### Step 3: Documentation

**Create supporting docs:**

1. **README.md** (if needed):
   - Quick start guide
   - Common use cases
   - Links to examples

2. **Examples:**
   - Real-world scenarios
   - Before/after comparisons
   - Different configurations

3. **Templates:**
   - Reusable code snippets
   - Configuration files
   - Project structures

### Step 4: Review

**Self-review checklist:**

- [ ] Follows skill structure
- [ ] Has YAML frontmatter
- [ ] Instructions are clear
- [ ] Examples work
- [ ] Tested across platforms
- [ ] No typos or errors
- [ ] Follows style guide

---

## Review Process

### Submission

1. **Fork the repository**
2. **Create a branch:** `git checkout -b skill-name`
3. **Add your skill:** Follow structure guidelines
4. **Test thoroughly:** Verify across platforms
5. **Submit PR:** Include description and examples

### PR Requirements

**Must include:**
- Clear description of the skill
- Use cases and examples
- Testing evidence
- Screenshots (if applicable)

**PR Template:**

```markdown
## Skill: [Name]

### Purpose
[What this skill does]

### Use Cases
- [Use case 1]
- [Use case 2]

### Testing
- [x] Tested in Claude.ai
- [x] Tested in Claude Code
- [x] Tested in Cursor
- [x] Examples verified

### Examples
[Link to examples or screenshots]
```

### Review Criteria

**Reviewers check:**

1. **Structure:**
   - Follows required format
   - Has all required files
   - Naming conventions correct

2. **Quality:**
   - Instructions are clear
   - Examples work
   - Best practices included

3. **Testing:**
   - Works across platforms
   - Handles edge cases
   - Error handling present

4. **Documentation:**
   - Clear and complete
   - No typos
   - Follows style guide

### Approval Process

1. **Automated checks:** Structure, format, links
2. **Peer review:** At least 2 reviewers
3. **Maintainer review:** Final approval
4. **Merge:** After all approvals

---

## Maintenance

### Keeping Skills Updated

**Responsibilities:**
- Monitor for issues
- Update examples
- Fix bugs
- Improve documentation

### Versioning

**Version format:** `MAJOR.MINOR.PATCH`

- **MAJOR:** Breaking changes
- **MINOR:** New features
- **PATCH:** Bug fixes

**Update version in:**
- SKILL.md frontmatter
- CHANGELOG.md (if exists)
- Release notes

### Deprecation

**If a skill is deprecated:**

1. Mark as deprecated in frontmatter
2. Add migration guide
3. Update documentation
4. Announce to users

---

## Style Guide

### Writing Style

- **Clear and concise:** No unnecessary words
- **Action-oriented:** Use imperative mood
- **Consistent:** Follow existing patterns
- **Professional:** Maintain high standards

### Code Style

- **Follow language conventions:** Java, Python, etc.
- **Include comments:** Explain why, not what
- **Error handling:** Always include
- **Production-ready:** No placeholders

### Documentation Style

- **Minto Pyramid:** Main point first, details follow
- **Examples first:** Show, then explain
- **Progressive disclosure:** Basic to advanced
- **Visual hierarchy:** Use headings effectively

---

## Skill Template

Use this template for new skills:

```markdown
---
name: skill-name
description: One-sentence description
version: 1.0.0
author: Your Name
tags: [tag1, tag2]
---

# Skill Name

## Purpose
[What this skill does and why it exists]

## When to Trigger This Skill
- [Use case 1]
- [Use case 2]

## Prerequisites
- [Requirement 1]
- [Requirement 2]

## Instructions

[Step-by-step instructions for Claude]

## Examples

### Example 1: [Scenario]
[Example code or usage]

### Example 2: [Scenario]
[Example code or usage]

## Best Practices
- [Practice 1]
- [Practice 2]

## Troubleshooting

### Issue: [Problem]
**Solution:** [Fix]

## References
- [Link 1]
- [Link 2]
```

---

## Getting Help

- **Questions:** Open a discussion
- **Issues:** Open an issue
- **Feedback:** Submit via PR comments
- **Contact:** Reach out to maintainers

---

**Thank you for contributing to AI Velocity!** 🚀
