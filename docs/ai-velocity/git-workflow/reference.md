# Git Workflow — Reference

Advanced operations for the git-workflow skill.

## Commit Signing Setup

Signed commits verify authorship. Use **SSH** (simpler) or **GPG** (traditional). GitHub and GitLab support both.

### Check if Already Configured

```bash
git config --get commit.gpgsign        # Should return "true" if enabled
git config --get user.signingkey        # GPG key ID or SSH key path
git config --get gpg.format             # "ssh" or "openpgp"
git log --show-signature -1             # Verify last commit
```

### Option A: SSH Signing (Recommended, Git 2.34+)

Uses your existing SSH key. No separate GPG setup.

```bash
# 1. Check for existing SSH key
ls -la ~/.ssh/id_*.pub

# 2. If none, generate one
ssh-keygen -t ed25519 -C "your_email@example.com" -f ~/.ssh/id_ed25519

# 3. Configure Git to use SSH for signing
git config --global gpg.format ssh
git config --global user.signingkey ~/.ssh/id_ed25519
git config --global commit.gpgsign true

# 4. Add SSH key to GitHub/GitLab (Settings → SSH Keys)
# For signing: Add the same key under "Signing keys" if separate; or use the same key for both auth and signing.
```

**GitHub**: Settings → SSH and GPG keys → New SSH key (paste `~/.ssh/id_ed25519.pub`). The same key works for auth and signing.

**GitLab**: Preferences → SSH Keys. Add your public key.

### Option B: GPG Signing

```bash
# 1. Install GPG (macOS: brew install gnupg, Linux: usually pre-installed)

# 2. Generate key
gpg --default-new-key-algo rsa4096 --gen-key
# Follow prompts; use your work email

# 3. Get key ID
gpg --list-secret-keys --keyid-format=long
# Use the ID after "sec   rsa4096/" (e.g., 3AA5C34371567BD2)

# 4. Configure Git
git config --global user.signingkey <KEY_ID>
git config --global commit.gpgsign true

# 5. Export public key and add to GitHub/GitLab
gpg --armor --export <KEY_ID>
# Paste in GitHub: Settings → SSH and GPG keys → New GPG key
# Paste in GitLab: Preferences → GPG Keys
```

### Verify Setup

```bash
git commit --allow-empty -m "test: verify signing"
git log --show-signature -1
# Should show "Good signature" or "Valid signature"
```

### Troubleshooting

- **"gpg failed to sign"**: Ensure GPG is in PATH; on macOS, add `export GPG_TTY=$(tty)` to `~/.zshrc` or `~/.bashrc`.
- **SSH key not found**: Use full path in `user.signingkey`; ensure key has correct permissions (`chmod 600 ~/.ssh/id_ed25519`).
- **GitHub/GitLab not showing verified**: Add the public key to your account; email in commit must match account email.

---

## Conflict Resolution

### When Conflicts Occur

1. **During merge**: `git merge main` reports conflicts.
2. **During rebase**: `git rebase main` stops at conflicting commit.

### Resolving Merge Conflicts

```bash
# Find conflicted files
git status

# Edit files - look for <<<<<<<, =======, >>>>>>>
# Remove markers, keep intended code

# After resolving
git add <resolved-files>
git commit   # for merge
# OR
git rebase --continue   # for rebase
```

### Abort if Needed

```bash
git merge --abort      # During merge
git rebase --abort    # During rebase
```

---

## Interactive Rebase

### Squash Commits

```bash
git rebase -i HEAD~3   # Last 3 commits
```

In editor, change `pick` to `squash` (or `s`) for commits to combine. Save and edit commit message.

### Reorder or Edit Commits

```bash
git rebase -i HEAD~5
```

Reorder lines to change commit order. Change `pick` to `edit` to pause and amend that commit.

### Reword Commit Message

```bash
git rebase -i HEAD~1
# Change 'pick' to 'reword', save
# Edit message in next screen
```

---

## Revert vs Reset

| Use | Command | Safe for shared branches? |
|-----|---------|---------------------------|
| Undo commit, keep changes staged | `git reset --soft HEAD~1` | No (rewrites history) |
| Undo commit, discard changes | `git reset --hard HEAD~1` | No |
| Create new commit that undoes another | `git revert <commit>` | Yes |

**Rule**: Use `git revert` for commits already pushed and shared.

---

## Useful Aliases

```bash
git config --global alias.lg "log --oneline --graph -20"
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
```
