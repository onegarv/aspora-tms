# Usage Guide

**How to use AI Velocity skills to accelerate your development.**

## Overview

This guide explains how to use AI Velocity skills in Claude.ai, Claude Code, and Cursor. Each skill is designed to be self-contained and immediately usable.

### Quick Start for Cursor Users

**Two ways to use skills:**

**Method 1: Natural Language (if ai-velocity is accessible)**
- Press `Cmd+L` (Mac) or `Ctrl+L` (Windows/Linux) to open chat
- Type: `Add Prometheus metrics to my service following SRE best practices`
- Cursor searches your workspace and finds the skill automatically

**Method 2: Explicit Reference (always works)**
- Press `Cmd+L` to open chat
- Type: `@/Users/apple/code/aspora/ai-velocity/prometheus-instrumentation/SKILL.md Add metrics`
- Replace the path with your actual `ai-velocity` location

**Note:** For natural language to work, `ai-velocity` must be in your workspace or explicitly added. Otherwise, use explicit `@` paths.

See [Using Skills in Cursor](#using-skills-in-cursor) for detailed instructions.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Using Skills in Claude.ai](#using-skills-in-claudeai)
3. [Using Skills in Claude Code](#using-skills-in-claude-code)
4. [Using Skills in Cursor](#using-skills-in-cursor)
5. [Skill Reference](#skill-reference)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Prerequisites

- **Claude.ai**: Free account
- **Claude Code / Cursor**: Installed and configured
- **Basic familiarity** with your development environment

### Installation

**For Claude.ai:** No installation - upload skills directly in the interface.

**For Claude Code / Cursor:** One-time global installation required. Skills are then available across all projects.

---

## Using Skills in Claude.ai

### Step 1: Access Skills

1. Open [Claude.ai](https://claude.ai)
2. Click the **skill icon (🧩)** in the chat interface
3. Select **"Add Custom Skill"**

### Step 2: Upload Skill

1. Navigate to the skill folder (e.g., `prometheus-instrumentation/`)
2. Upload the entire folder or just `SKILL.md`
3. Claude will automatically detect and load the skill

### Step 3: Use the Skill

Simply ask Claude to use the skill:

```
Add Prometheus metrics to my Java service
```

Claude will automatically:
- Analyze your codebase
- Identify what needs instrumentation
- Add metrics following best practices
- Explain what was done

### Example Workflows

**Observability Setup:**
```
1. "Add Prometheus metrics to my service"
2. "Create a Datadog dashboard for my service"
3. "Set up alerting rules for my metrics"
```

**Code Review:**
```
"Review my service for missing observability"
```

---

## Using Skills in Claude Code

### Overview

**Skills work globally** - Install once, use everywhere. No need to copy skills into each repository.

### Step 1: Install Skills Globally

```bash
# Create global skills directory
mkdir -p ~/.config/claude-code/skills/

# Copy skills from ai-velocity (replace with your actual path)
cp -r /path/to/ai-velocity/prometheus-instrumentation ~/.config/claude-code/skills/
cp -r /path/to/ai-velocity/datadog-dashboard-creation ~/.config/claude-code/skills/
cp -r /path/to/ai-velocity/alert-generation ~/.config/claude-code/skills/
```

### Step 2: Invoke Skills

**In Chat:**
```
@prometheus-instrumentation/SKILL.md Add metrics to my service
```

**In Code:**
```java
// TODO: Add Prometheus metrics
// See: @prometheus-instrumentation/SKILL.md
```

### Step 3: Verify

```bash
ls ~/.config/claude-code/skills/
# Should show: prometheus-instrumentation, datadog-dashboard-creation, alert-generation
```

---

## Using Skills in Cursor

### Overview

**How Cursor Uses Skills:**

Cursor's AI can use skills in two ways:

1. **Automatic Detection**: Cursor searches your **current workspace** for relevant skill files when you describe a task
2. **Explicit Reference**: Use `@` syntax to explicitly reference a skill file (works from anywhere)

**Important:** For auto-detection to work, `ai-velocity` must be accessible to Cursor:
- In your current workspace
- In a sibling directory that Cursor can access
- Or explicitly referenced with `@` syntax

**Example - Automatic Detection (if ai-velocity is accessible):**
```
Add Prometheus metrics to my Java service
```
If `ai-velocity` is in your workspace, Cursor will:
- Search your workspace for relevant files
- Find `prometheus-instrumentation/SKILL.md` in `ai-velocity`
- Understand you need observability
- Apply the skill to your codebase

**Example - Explicit Reference (always works):**
```
@/Users/apple/code/aspora/ai-velocity/prometheus-instrumentation/SKILL.md Add metrics
```
This explicitly tells Cursor which skill file to use, regardless of workspace location.

### Installation Options

**Option 1: Keep ai-velocity Accessible (Recommended for Auto-Detection)**

For natural language to work, `ai-velocity` must be in Cursor's searchable area:

**A. In Your Workspace:**
```bash
# If ai-velocity is in your workspace root or subdirectory
# Cursor will automatically find it when you use natural language
```

**B. As Sibling Directory:**
```bash
# Clone ai-velocity as a sibling to your project
cd /Users/apple/code/aspora/
git clone <ai-velocity-repo-url> ai-velocity

# Now when you work in goblin-service, Cursor can access ../ai-velocity
```

**C. Add to Workspace:**
- In Cursor, use "File > Add Folder to Workspace"
- Add the `ai-velocity` directory
- Now Cursor can search it automatically

**How Auto-Detection Works:**
1. You type: "Add Prometheus metrics to my service"
2. Cursor searches your workspace for files matching "prometheus", "metrics", "SKILL.md"
3. Finds `ai-velocity/prometheus-instrumentation/SKILL.md`
4. AI reads the skill file and applies it

**Option 2: Explicit Reference (Always Works - No Setup)**

If `ai-velocity` isn't accessible, use explicit paths:

```
@/Users/apple/code/aspora/ai-velocity/prometheus-instrumentation/SKILL.md Add metrics
```

This works from anywhere, regardless of workspace configuration.

**Option 3: Project-Specific (Per Repository)**

Copy skills to your project for guaranteed access:

```bash
# In your project directory
mkdir -p .claude-skills/
cp -r /path/to/ai-velocity/prometheus-instrumentation .claude-skills/
```

Then reference: `@.claude-skills/prometheus-instrumentation/SKILL.md`

**Recommendation:** 
- If `ai-velocity` is accessible: Use natural language (auto-detection)
- Otherwise: Use explicit `@` paths (always reliable)

### How to Use Skills in Cursor

**Method 1: Natural Language (Recommended - AI Auto-Detection)**

Just describe what you want in natural language. Cursor's AI will automatically find and use the relevant skill:

1. Open Cursor
2. Press `Cmd+L` (Mac) or `Ctrl+L` (Windows/Linux) to open chat
3. Type naturally:
   ```
   Add Prometheus metrics to my Java service following SRE best practices
   ```
   
   Or:
   ```
   Create a Datadog dashboard for my service with the four golden signals
   ```

Cursor's AI will:
- Understand your intent
- Search for relevant skills in your workspace
- Automatically apply the appropriate skill
- Follow the skill's best practices

**Method 2: Explicit Reference (When You Need a Specific Skill)**

Use `@` syntax to explicitly reference a skill file:

```
@/Users/apple/code/aspora/ai-velocity/prometheus-instrumentation/SKILL.md Add metrics to my service
```

**When to use explicit reference:**
- You want to use a specific skill version
- Multiple skills could apply and you want to be explicit
- The skill is in a non-standard location

**Method 3: In Composer**

1. Press `Cmd+I` (Mac) or `Ctrl+I` (Windows/Linux) to open composer
2. Type naturally or with explicit reference:
   ```
   Add Prometheus metrics to OrderService.java
   ```
   Or:
   ```
   @/Users/apple/code/aspora/ai-velocity/prometheus-instrumentation/SKILL.md Add metrics to OrderService.java
   ```

### How Cursor Discovers Skills

**Method 1: Workspace Search (Auto-Detection)**

When you use natural language, Cursor searches your **current workspace**:

1. **You type:** "Add Prometheus metrics to my service"
2. **Cursor searches:** Your workspace for files matching keywords ("prometheus", "metrics", "SKILL.md")
3. **Cursor finds:** `ai-velocity/prometheus-instrumentation/SKILL.md` (if accessible)
4. **AI reads:** The skill file and applies its guidelines

**Where Cursor Searches:**
- ✅ Current workspace root and all subdirectories
- ✅ Files explicitly added to workspace (via "Add Folder to Workspace")
- ❌ Sibling directories (usually NOT searched unless added to workspace)
- ❌ Global config directories (may not be searched)

**Important:** For auto-detection to work, `ai-velocity` must be:
- Inside your current workspace, OR
- Added to workspace via "Add Folder to Workspace", OR
- Explicitly referenced with `@` syntax

**Method 2: Explicit Reference (Always Works)**

Use `@` syntax to explicitly tell Cursor where to look:

```
@/Users/apple/code/aspora/ai-velocity/prometheus-instrumentation/SKILL.md Add metrics
```

This works regardless of workspace configuration.

**Best Practice:** 
- If `ai-velocity` is in your workspace: Use natural language
- Otherwise: Use explicit `@` paths (more reliable)

### Complete Example: Adding Metrics in Cursor

**Scenario:** Add Prometheus metrics to your Java service

**Option A: Natural Language (if ai-velocity is in workspace)**

1. **Open Cursor** in your project directory
2. **Ensure `ai-velocity` is accessible:**
   - Either in your workspace, OR
   - Added via "File > Add Folder to Workspace"
3. **Open Chat:** `Cmd+L` / `Ctrl+L`
4. **Type naturally:**
   ```
   Add Prometheus metrics to all my service classes following SRE best practices. Include L0 and L1 metrics with proper tag sanitization.
   ```
5. **Cursor's AI will:**
   - Search your workspace for files matching "prometheus", "metrics", "SKILL.md"
   - Find `ai-velocity/prometheus-instrumentation/SKILL.md` (if accessible)
   - Read the skill file
   - Understand you need observability metrics
   - Analyze your codebase
   - Identify service classes
   - Add metrics instrumentation following the skill's guidelines
   - Show you the changes

**If auto-detection doesn't work:** Use Option B (explicit reference)

**Option B: Explicit Reference**

1. **Open Cursor** in your project directory
2. **Open Chat:** `Cmd+L` / `Ctrl+L`
3. **Type with explicit path:**
   ```
   @/Users/apple/code/aspora/ai-velocity/prometheus-instrumentation/SKILL.md Analyze my codebase and add Prometheus metrics to all service classes
   ```

**Recommendation:** Start with natural language. Cursor's AI is designed to understand context and find the right tools automatically.

### Cursor-Specific Tips

**Keyboard Shortcuts:**
- `Cmd+L` / `Ctrl+L`: Open chat
- `Cmd+I` / `Ctrl+I`: Open composer
- `Cmd+K` / `Ctrl+K`: Quick edit

**Auto-completion:**
- Start typing `@prometheus` and Cursor will suggest available skills
- Use Tab to autocomplete skill names

**Multi-file Editing:**
- Skills can modify multiple files at once
- Cursor shows all changes in a unified diff view

### Troubleshooting Cursor

**Issue: Skill not found**

**Solution:** Use absolute path instead:
```
@/Users/apple/code/aspora/ai-velocity/prometheus-instrumentation/SKILL.md
```

**Issue: @ syntax not working**

- Ensure Cursor is up to date
- Use absolute path (always works): `@/full/path/to/ai-velocity/prometheus-instrumentation/SKILL.md`
- Verify the file exists: `ls /path/to/ai-velocity/prometheus-instrumentation/SKILL.md`

**Issue: Short names not working (e.g., `@prometheus-instrumentation`)**

- Cursor may not support global skill discovery
- Use absolute paths instead: `@/full/path/to/ai-velocity/prometheus-instrumentation/SKILL.md`
- This method always works regardless of Cursor's configuration

---

## Skill Reference

### Prometheus Metrics

**Purpose:** Instrument Java services with production-grade metrics

**When to use:**
- Setting up observability for a new service
- Adding metrics to existing code
- Reviewing metrics implementation

**Usage:**
```
@prometheus-instrumentation/SKILL.md Add metrics to my service
```

**What it does:**
- Analyzes your service classes
- Adds L0 and L1 metrics
- Creates MetricsUtil helper class
- Configures Prometheus endpoint
- Documents all metrics

### Datadog Dashboards

**Purpose:** Create SRE-grade observability dashboards

**When to use:**
- Creating dashboards for new services
- Improving existing dashboards
- Following SRE best practices

**Usage:**
```
@datadog-dashboard-creation/SKILL.md Create a dashboard for my service
```

**What it does:**
- Analyzes your metrics
- Creates dashboard JSON
- Organizes by Golden Signals
- Follows RED method
- Includes error handling

### Alerting

**Purpose:** Set up intelligent alerting rules

**When to use:**
- Configuring alerts for services
- Setting up on-call workflows
- Defining SLOs and SLIs

**Usage:**
```
@alert-generation/SKILL.md Set up alerts for my service
```

**What it does:**
- Creates alert rules
- Defines SLOs
- Sets up notification channels
- Includes runbooks

---

## Best Practices

### 1. Start with Metrics

Always instrument metrics before creating dashboards or alerts:

```
1. Add Prometheus metrics
2. Create Datadog dashboard
3. Set up alerts
```

### 2. Use Skills Consistently

Reference skills in your project documentation:

```markdown
## Observability

This service uses AI Velocity skills for observability:
- Metrics: @prometheus-instrumentation/SKILL.md
- Dashboards: @datadog-dashboard-creation/SKILL.md
- Alerts: @alert-generation/SKILL.md
```

### 3. Review Generated Code

Always review what AI generates:
- Verify metrics are correct
- Check dashboard queries
- Validate alert thresholds

### 4. Customize for Your Needs

Skills provide defaults, but customize:
- Metric names to match your conventions
- Dashboard layouts for your team
- Alert thresholds for your SLOs

### 5. Share Knowledge

Document your customizations:
- Update team wiki
- Share in code reviews
- Present in team meetings

---

## Troubleshooting

### Skill Not Loading

**Problem:** Skill doesn't appear in Claude

**Solution:**
1. Verify skill folder structure
2. Check `SKILL.md` has YAML frontmatter
3. Ensure file encoding is UTF-8

### Incorrect Results

**Problem:** AI generates wrong code

**Solution:**
1. Provide more context in your prompt
2. Reference specific files: `@file.java`
3. Check skill version matches your needs

### Missing Dependencies

**Problem:** Generated code has missing imports

**Solution:**
1. Verify your project structure
2. Check build configuration
3. Review skill prerequisites

### Performance Issues

**Problem:** Skill execution is slow

**Solution:**
1. Limit scope: "Add metrics to OrderService only"
2. Use smaller codebases for testing
3. Check Claude API rate limits

---

## Advanced Usage

### Combining Skills

Use multiple skills together:

```
1. @prometheus-instrumentation/SKILL.md Add metrics
2. @datadog-dashboard-creation/SKILL.md Create dashboard using those metrics
3. @alert-generation/SKILL.md Set up alerts based on dashboard
```

### Custom Prompts

Create custom prompts that reference skills:

```markdown
# My Service Setup

Follow these steps:
1. @prometheus-instrumentation/SKILL.md - Add metrics
2. Review generated code
3. @datadog-dashboard-creation/SKILL.md - Create dashboard
4. Deploy and verify
```

### Team Workflows

Integrate skills into your team processes:

- **Code Review:** Reference skills in PR templates
- **Onboarding:** Include skills in new hire docs
- **Standards:** Document which skills to use when

---

## Next Steps

- Read [Contributing Guide](contributing.md) to create your own skills
- Explore existing skills for patterns
- Share your customizations with the team

---

**Questions?** Open an issue or contact the maintainers.
