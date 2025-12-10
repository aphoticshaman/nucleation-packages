# SESSION_CONTINUITY.skill.md

## Protocol for Maintaining State Across Claude Code Conversations

**Version**: 1.0
**Domain**: Development Workflow, Context Management, Git Operations
**Purpose**: Eliminate context loss, repo confusion, and work duplication between sessions

---

## 1. EXECUTIVE SUMMARY

Claude Code sessions are stateless - each new conversation starts from zero. This creates catastrophic failure modes:
- Pushing to wrong branches
- Confusing multiple repositories
- Duplicating completed work
- Losing critical decisions and context
- Breaking user's local state

This skill file provides protocols for **seamless session continuity** - capturing state at session end and reconstructing it at session start.

**Core Principle**: Treat each session like a shift handoff between developers working on the same codebase.

---

## 2. THE SESSION HANDOFF PROBLEM

### 2.1 What Gets Lost Between Sessions

**Git State**:
- Current branch name
- Commits made this session
- Push/pull status with remotes
- Uncommitted changes
- Stash contents

**Work Context**:
- What problem we were solving
- Decisions made and their rationale
- Failed approaches (don't retry)
- Pending tasks
- User preferences discovered

**Repository Topology**:
- Multiple repos (mirrors, forks)
- Remote names and URLs
- Sync status between repos
- Which repo is source of truth

**Code Context**:
- Files modified
- Architecture decisions
- Patterns established
- Naming conventions used

### 2.2 Common Failure Modes

**Repo Confusion**:
- User has `lunatiq` and `HungryOrca` - which am I working in?
- User's local is ahead/behind of Claude's
- Pushing to wrong remote

**Branch Confusion**:
- Multiple claude/* branches from different sessions
- Not knowing which branch has latest work
- Accidentally working on old branch

**Work Duplication**:
- Re-implementing something already done
- Re-investigating already-answered questions
- Re-making already-made decisions

**State Desync**:
- Claude's repo state differs from user's
- User made changes Claude doesn't know about
- Merge conflicts from parallel work

---

## 3. SESSION START PROTOCOL

### 3.1 Immediate Context Gathering

When starting a session on an existing project, IMMEDIATELY gather:

```bash
# 1. Current branch and status
git branch --show-current
git status -sb

# 2. Recent commits (shows what was done)
git log --oneline -10

# 3. Remote configuration
git remote -v

# 4. Sync status with all remotes
git fetch --all
git branch -vv

# 5. Any uncommitted work
git diff --stat
git stash list
```

### 3.2 Context Reconstruction Questions

If the user says "continue from last session" or similar:

1. **What branch should I be on?**
   - Check gitStatus in system prompt
   - Ask if unclear

2. **What was the last task?**
   - Check recent commits for context
   - Ask user to summarize if needed

3. **Are there multiple repos?**
   - Check remotes
   - Clarify which is source of truth

4. **What's the sync status?**
   - Is user's local ahead/behind?
   - Do repos need syncing?

### 3.3 Red Flags at Session Start

**STOP and clarify if you see**:
- Multiple remotes with different URLs
- Branch name doesn't match expected pattern
- Recent commits from unknown authors
- Uncommitted changes you didn't make
- Merge conflicts

---

## 4. SESSION END PROTOCOL

### 4.1 Git State Capture

Before ending a session, ALWAYS:

```bash
# 1. Commit all work
git add -A
git status  # Verify what's being committed
git commit -m "Description of work done"

# 2. Push to remote
git push -u origin <branch-name>

# 3. Capture final state
git log --oneline -5
git status -sb
git branch -vv
```

### 4.2 Handoff Summary Template

Provide this at session end:

```markdown
## SESSION HANDOFF - [Date]

### Git State
- **Branch**: `claude/feature-name-sessionId`
- **Last Commit**: `abc1234` - "Commit message"
- **Push Status**: Pushed to origin
- **Sync Commands for User**:
  ```bash
  git pull origin <branch>
  git push origin <branch>  # if user has local changes
  ```

### Work Completed
1. [Task 1] - files modified, approach taken
2. [Task 2] - files modified, approach taken

### Pending Tasks
1. [Task] - why it's pending, what's needed

### Key Decisions Made
1. [Decision] - rationale
2. [Decision] - rationale

### Failed Approaches (Don't Retry)
1. [Approach] - why it failed

### Files Modified This Session
- `path/to/file1.js` - what changed
- `path/to/file2.ts` - what changed

### User Preferences Discovered
- Prefers X over Y
- Uses Z workflow

### Next Session Should
1. Start by doing X
2. Continue with Y
3. Test Z
```

### 4.3 Multi-Repo Sync Instructions

If working with multiple repos (e.g., lunatiq + HungryOrca):

```markdown
### Repo Sync Status
- **Primary (source of truth)**: HungryOrca
- **Mirror**: lunatiq

### Sync Commands for User
```bash
# Pull from primary
git pull hungryorca claude/<branch>

# Push to mirror
git push origin claude/<branch>
```

### Current Sync State
- HungryOrca: `abc1234`
- lunatiq: `abc1234`
- Status: IN SYNC ✓
```

---

## 5. MULTI-REPO MANAGEMENT

### 5.1 Identifying Repo Topology

Common patterns:
- **Mirror**: Same code, different locations (backup)
- **Fork**: Diverged from upstream, may sync
- **Monorepo subdir**: Multiple projects in one repo

**Detection**:
```bash
# List all remotes
git remote -v

# Check if remotes point to same repo
# Look for: origin, upstream, fork names
```

### 5.2 Establishing Source of Truth

Pick ONE repo as source of truth. All others sync FROM it.

**Protocol**:
1. Claude pushes to source of truth
2. User pulls from source of truth
3. User pushes to mirrors
4. OR Claude pushes to both (if access)

**Document in every session**:
- Which repo is source of truth
- Sync direction
- User's responsibility vs Claude's

### 5.3 Sync Conflict Resolution

If repos diverge:

```bash
# 1. Fetch both
git fetch origin
git fetch other-remote

# 2. Compare
git log origin/branch..other-remote/branch --oneline
git log other-remote/branch..origin/branch --oneline

# 3. Decide which has the "real" work
# 4. Reset the other to match
# 5. Force push if necessary (with user permission)
```

---

## 6. BRANCH MANAGEMENT

### 6.1 Branch Naming Convention

Pattern: `claude/<feature>-<sessionId>`

- `claude/` prefix identifies Claude-created branches
- `<feature>` describes the work
- `<sessionId>` links to specific conversation

**Examples**:
- `claude/oracle-llm-integration-01Pb2hy2J5P1`
- `claude/fix-race-condition-01QhjkR1Dqkk`

### 6.2 Branch Lifecycle

1. **Create**: At session start if new work
2. **Work**: All commits go here
3. **Push**: Before session end
4. **Merge**: User merges to main when ready
5. **Delete**: After merge (optional)

### 6.3 Dealing with Old Branches

If you see multiple `claude/*` branches:
1. Check which has most recent commits
2. Ask user which is current
3. Don't assume - wrong branch = wasted work

---

## 7. CONTEXT PRESERVATION PATTERNS

### 7.1 Decision Documentation

When making significant decisions, document inline:

```javascript
// DECISION [2024-11-19]: Using queue instead of mutex for LLM requests
// RATIONALE: llama.rn context is single-threaded, queue ensures sequential
// ALTERNATIVES CONSIDERED: Lock/mutex (adds complexity), disable button (poor UX)
```

### 7.2 Failed Approach Documentation

When something doesn't work:

```javascript
// FAILED APPROACH [2024-11-19]: Tried using AsyncStorage for model cache
// REASON: 6MB limit exceeded by 2.4GB model
// SOLUTION: Using FileSystem.cacheDirectory instead
```

### 7.3 TODO Markers for Continuity

Leave breadcrumbs for next session:

```javascript
// TODO [NEXT SESSION]: Implement card synergy detection
// CONTEXT: User wants hybrid interpretation, AGI provides facts, LLM synthesizes
// FILES INVOLVED: agiInterpretation.js, oracleActor.ts
// BLOCKER: Need to understand cardSynergyMatrix.js first
```

---

## 8. USER STATE MANAGEMENT

### 8.1 Tracking User's Environment

Note and remember:
- OS (Windows/Mac/Linux)
- Shell (cmd/PowerShell/bash/zsh)
- IDE preferences
- Build system (npm/yarn/pnpm)
- Emulator/device setup

### 8.2 User's Local vs Claude's Environment

**They are different!** Claude's environment:
- Linux-based
- May have different installed packages
- Different file paths
- Different line endings potentially

**Protocol**:
- Give commands appropriate to USER's OS
- Note when behavior might differ
- Test on user's environment when possible

### 8.3 User Preference Learning

Track preferences discovered during session:
- Code style preferences
- Communication style
- Risk tolerance
- Workflow patterns

Document in handoff for next session.

---

## 9. ERROR RECOVERY

### 9.1 Desync Recovery

If user's repo and Claude's diverge:

```bash
# On user's machine:
# 1. Stash any local work
git stash

# 2. Fetch Claude's changes
git fetch origin claude/<branch>

# 3. Reset to Claude's state
git reset --hard origin/claude/<branch>

# 4. Reapply local work if needed
git stash pop
```

### 9.2 Wrong Branch Recovery

If work was done on wrong branch:

```bash
# 1. Note the commits made
git log --oneline -n

# 2. Switch to correct branch
git checkout correct-branch

# 3. Cherry-pick the commits
git cherry-pick <commit1> <commit2> ...

# 4. Clean up wrong branch if needed
```

### 9.3 Push Rejection Recovery

If push fails (non-fast-forward):

```bash
# 1. Fetch remote state
git fetch origin

# 2. Check what's different
git log HEAD..origin/branch --oneline
git log origin/branch..HEAD --oneline

# 3. Either rebase or merge
git rebase origin/branch  # if linear history preferred
# OR
git merge origin/branch   # if preserving both

# 4. Push
git push origin branch
```

---

## 10. COMMUNICATION PROTOCOLS

### 10.1 Asking for Context

When context is missing, ask specifically:

**BAD**: "Can you give me more context?"
**GOOD**: "I see you're on branch X. What was the last task we were working on? Are there changes on your local I should know about?"

### 10.2 Confirming Shared Understanding

Before making changes:

"I'm going to [action] on [branch] and push to [remote]. This will [effect]. Sound right?"

### 10.3 Sync Instructions

Always provide copy-paste commands:

```bash
# Pull my changes
git pull hungryorca claude/init-git-repo-01QhjkR1DqkkRBaVktUfUd2E

# Push to your mirror
git push origin claude/init-git-repo-01QhjkR1DqkkRBaVktUfUd2E
```

Not: "Pull from the branch and push to origin."

---

## 11. TESTING CONTINUITY

### 11.1 Session Start Checklist

- [ ] Identified current branch
- [ ] Checked recent commits
- [ ] Verified remote configuration
- [ ] Checked sync status
- [ ] Confirmed source of truth repo
- [ ] Understood pending work
- [ ] No uncommitted changes from unknown source

### 11.2 Session End Checklist

- [ ] All work committed
- [ ] Pushed to remote
- [ ] Provided sync commands for user
- [ ] Documented decisions made
- [ ] Listed pending tasks
- [ ] Noted failed approaches
- [ ] Captured user preferences
- [ ] Clear next steps

### 11.3 Continuity Verification

At session start, verify with user:
- "Last session we were working on X, correct?"
- "You're on branch Y?"
- "Any changes since we last talked?"

---

## 12. FAILURE MODE CATALOG

### 12.1 The Repo Mixup
**Symptom**: Pushing to wrong repo or branch
**Prevention**: Always check `git remote -v` and `git branch` at session start
**Recovery**: Fetch correct remote, cherry-pick or rebase

### 12.2 The Zombie Branch
**Symptom**: Working on old branch, missing recent work
**Prevention**: Check commit dates, ask user which branch is current
**Recovery**: Find correct branch, merge or rebase

### 12.3 The Phantom Changes
**Symptom**: User's local has changes Claude doesn't know about
**Prevention**: Always ask "any changes since last session?"
**Recovery**: User commits and pushes, Claude pulls

### 12.4 The Duplicate Work
**Symptom**: Re-implementing already completed features
**Prevention**: Check git log for recent work, ask what's done
**Recovery**: Git history shows what was done, don't redo

### 12.5 The Decision Amnesia
**Symptom**: Re-making decisions, contradicting previous approach
**Prevention**: Document decisions inline, reference in handoff
**Recovery**: Search git log and code comments for prior decisions

---

## 13. TOOLING INTEGRATION

### 13.1 Claude Code Specific

**gitStatus in system prompt**: Always check this first - it's the starting state snapshot

**Stop hooks**: If present, they provide automated checks before session end

**Todo tracking**: Use TodoWrite to maintain task state

### 13.2 Git Hooks

If repo has hooks:
- pre-commit: May modify files
- post-commit: May trigger notifications
- pre-push: May run tests

Account for these in workflow.

### 13.3 CI/CD Awareness

If repo has CI/CD:
- Commits may trigger builds
- Push may trigger deployments
- Consider implications before push

---

## 14. ADVANCED PATTERNS

### 14.1 Concurrent Sessions

If user might run multiple Claude sessions:
- Use unique branch names per session
- Coordinate before merging
- Be aware of potential conflicts

### 14.2 Long-Running Features

For multi-session features:
1. Create tracking issue/doc
2. Maintain persistent TODO
3. Regular sync points
4. Clear milestone markers

### 14.3 Handoff to Different Claude Instance

When handing off to fresh context:
1. Provide comprehensive summary
2. Include all file paths
3. Link to relevant docs/issues
4. Explicit "do this first" instruction

---

## 15. TEMPLATES

### 15.1 Session Start Message

```
Starting session on [project].

**Current Git State:**
- Branch: X
- Last commit: Y
- Status: Clean/Uncommitted changes

**Remotes:**
- origin: [url]
- [other]: [url]

**Questions:**
1. What should I focus on this session?
2. Any changes since last time I should know about?
3. [Specific question about state]
```

### 15.2 Session End Message

```
## Session Summary - [Date]

### Completed
- [x] Task 1
- [x] Task 2

### Git Status
- Branch: `X`
- Commits: `abc1234`, `def5678`
- Pushed: Yes ✓

### Sync Commands
```bash
git pull origin X
```

### Next Session
1. Start with X
2. Then do Y
3. Test Z

### Key Files Changed
- `path/file1.js` - Description
- `path/file2.ts` - Description
```

---

## 16. VERSION HISTORY

- **v1.0** (2024-11): Initial protocol specification

---

*This skill should be applied at the start and end of EVERY session working on an existing project. Consistency in these protocols eliminates the majority of context loss issues.*
