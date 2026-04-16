# Design: Antigravity v12.0 (The Autonomous Loop)

## Goal
To elevate the Antigravity prompt to a 10/10 rating by transitioning from a rigid rule-follower to an intelligent development partner. This involves "Phase Fusion" for efficiency, milestone-based snapshots, and proactive persistent memory integration.

## User Review Required
> [!IMPORTANT]
> **PHASE FUSION**: Allows the agent to propose skipping individual Design/Plan phases for trivial tasks. This is a significant shift toward agent autonomy and requires explicit `ALLOW FUSION` approval.

> [!NOTE]
> **Milestone Snapshots**: Snapshots will now trigger based on project progress (e.g., plan completion) rather than turn counts, which may result in fewer but more meaningful state saves.

## Proposed Changes

### 1. Phase Fusion Protocol
- New Command: `PHASE FUSION [X, Y, Z]`
- Logic: When a task is identified as "trivial" (e.g., renames, documentation), the agent summarizes the intent and proposes fusing the standard Golden Loop phases.
- Command: `ALLOW FUSION` to authorize the jump to Phase 3.

### 2. Milestone-Based State Management
- **Section IV (Snapshots)**: Deprecate turn-based snapshots (15/30/45).
- **Trigger Logic**: Trigger `SAVE STATE` automatically upon:
    - Finalizing a `plan.md`.
    - Completing all tasks in a `plan.md`.
    - Resolving a `[CRITICAL]` review issue.

### 3. Persistent Memory Awareness (Brain Sync)
- **Section VII (Initialization)**: Add a "Brain Scan" step.
- **Workflow**: Upon `Initialize Antigravity`, the agent must search for and summarize relevant Knowledge Items (KIs) and recent Brain logs to ensure zero "amnesia" between sessions.

### 4. Stealth Audits
- **Constraint C9 Update**: If a task is `FAST-TRACKED`, the agent performs the audit mentally and omits the `[Audit]` string from the output.

## Success Criteria
1. **Efficiency**: Fused phases reduce the number of user interactions for simple tasks by >50%.
2. **Context**: Snapshot restoration points are always at logical logical milestones.
3. **Intelligence**: Initial turn of a session correctly identifies 1-2 relevant KIs or Brain logs.
4. **Cleanliness**: Documentation updates contain zero audit/thinking overhead in the terminal view.
