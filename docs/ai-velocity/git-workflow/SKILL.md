---
name: git-workflow
description: "Use when writing commit messages, creating branches, signing commits, or managing git history. Defines Conventional Commits format, commit signing, branch naming, merge vs rebase guidelines, and PR checklist. Use when the user asks for git help, commit conventions, how to sign commits, or how to structure commits and branches."
version: 1.0.0
author: AI Velocity Team
---

# Git Workflow Standards

Production-grade git conventions for commits, branches, and PR workflows. Complements [engineering-foundations](organizational-skills/engineering-foundations/) for team consistency.

## When to Apply

- Writing or amending commit messages
- Setting up or verifying commit signing
- Creating branches for features or fixes
- Merging, rebasing, or resolving conflicts
- When user asks: "how do I commit this?", "write a commit message", "merge vs rebase?"
- Before opening PRs or pushing changes
- Undoing or cleaning up git history

---

## 1. Commit Conventions (Conventional Commits)

### Format

```
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

- **Type**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
- **Scope**: Optional; module or area (e.g., `auth`, `api`, `payment`)
- **Subject**: Imperative mood, lowercase start, no period. Max ~72 chars.

### Examples

```
feat(auth): add JWT token validation
fix(api): handle null response in order service
chore(deps): upgrade Spring Boot to 3.2
docs(readme): update installation steps
refactor(payment): extract validation logic
```

### Rules

- **One logical change per commit**: Easier to review and revert.
- **Imperative mood**: "add validation" not "added validation".
- **Body when needed**: Explain *why* for non-obvious changes.
- **Footer for breaking changes**: `BREAKING CHANGE: ...` or `fix!: ...`

### Commit Signing

- **Sign commits** when required by org or to verify authenticity.
- Signed commits show a "Verified" badge in GitHub/GitLab.
- If signing is not configured: See [reference.md](reference.md#commit-signing-setup) for setup.

```bash
git config --get commit.gpgsign   # Check if signing is enabled
git log --show-signature -1       # Verify last commit is signed
```

---

## 2. Branch Naming and Workflow

### Branch Prefixes

| Prefix | Use |
|--------|-----|
| `feature/` | New features |
| `bugfix/` | Bug fixes |
| `hotfix/` | Urgent production fixes (from main) |
| `refactor/` | Code refactoring |
| `docs/` | Documentation only |

### Naming Pattern

```
<prefix>/<short-description>
```

Examples: `feature/order-payment`, `bugfix/null-check-in-api`, `hotfix/security-patch`

### Workflow (Trunk-Based Style)

- **Main** (`main` or `master`): Production-ready; protected.
- **Short-lived branches**: Merge within days, not weeks.
- **Sync often**: Rebase or merge from main to avoid large conflicts.

---

## 3. Merge vs Rebase

### When to Rebase

- **Local, unpushed commits**: Clean up history before pushing.
- **Feature branch**: Rebase onto main to keep linear history.
- **Before PR**: Rebase onto target branch for clean diff.

### When to Merge

- **Shared branches**: Never rebase after others have pulled.
- **Merging PR**: Use merge commit or squash as team prefers.
- **Release branches**: Merge, don't rebase.

### Rule of Thumb

- Rebase **your own** local commits.
- Merge when integrating **others'** work or when branch is shared.

```bash
# Before pushing feature branch
git fetch origin main
git rebase origin/main
```

---

## 4. Common Operations

### Stash

```bash
git stash                    # Stash changes
git stash pop                # Apply and remove
git stash apply              # Apply, keep in stash
git stash list               # List stashes
```

### Undo

| Situation | Command |
|-----------|---------|
| Unstage file | `git restore --staged <file>` |
| Discard working changes | `git restore <file>` |
| Undo last commit (keep changes) | `git reset --soft HEAD~1` |
| Undo last commit (discard changes) | `git reset --hard HEAD~1` |
| Revert a commit (new commit) | `git revert <commit>` |

### Check Status

```bash
git status
git log --oneline -10
git diff
git diff --staged
```

---

## 5. Before Opening a PR

### Checklist

- [ ] Commits follow Conventional Commits format
- [ ] Commits are signed (if required by org)
- [ ] Branch is rebased/merged from latest main
- [ ] No merge commits from main in middle of branch (rebase instead)
- [ ] Changes are logical and reviewable (not one giant commit)
- [ ] No debug code, console logs, or commented-out code
- [ ] Tests pass locally

### Suggested Commands

```bash
git fetch origin main
git rebase origin/main
git log --oneline -5   # Verify commit messages
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Create branch | `git checkout -b feature/my-feature` |
| Commit | `git add .` then `git commit -m "feat(scope): message"` |
| Amend last commit | `git commit --amend` |
| Sync with main | `git fetch origin main && git rebase origin/main` |
| View history | `git log --oneline` |

---

## Additional Resources

- For conflict resolution, interactive rebase, and **commit signing setup**: See [reference.md](reference.md)
