# Master Prompt v11.0 Implementation Plan

> **For Antigravity:** REQUIRED WORKFLOW: Use `.agent/workflows/execute-plan.md` to execute this plan in single-flow mode.

**Goal:** Update `master-prompt.md` to version 11.0, aligning it with the Antigravity persona and streamlining operational gates.

**Architecture:** Systematic text modification of the core system prompt, focused on identity, efficiency, and skill integration.

**Tech Stack:** Markdown

---

### Task 1: Environment Verification

**Files:**
- Modify: `docs/plans/task.md`

**Step 1: Verify current prompt state**
Run: `ls master-prompt.md`
Expected: File exists and is readable.

**Step 2: Initialize Git (if not already)**
Run: `git init`
Expected: Git repo initialized.

**Step 3: Commit current state as baseline**
Run: `git add master-prompt.md; git commit -m "chore: baseline master-prompt v10.0"`

---

### Task 2: Identity & Branding Update

**Files:**
- Modify: `master-prompt.md`

**Step 1: Replace "Superpowers" with "Antigravity"**
Modify: `master-prompt.md:1-10`, `master-prompt.md:337-365`
Target: Change Title, Persona Identity, Intake Form, and Termination Protocol.

**Step 2: Update Version to 11.0**
Modify: `master-prompt.md:3`

**Step 3: Commit changes**
Run: `git add master-prompt.md; git commit -m "feat: rebrand to Antigravity v11.0"`

---

### Task 3: Fast-Track Protocol Implementation

**Files:**
- Modify: `master-prompt.md`

**Step 1: Add Fast-Track Command to Reference**
Modify: `master-prompt.md:305-324`
TargetContent:
```markdown
| `FAST-TRACK <reason>` | Skip TDD constraints for documentation, comments, or CSS |
```

**Step 2: Update Constraint C4 (No Test, No Code) with Fast-Track exception**
Modify: `master-prompt.md:68-71`
TargetContent:
```markdown
> **Permitted exception:** Purely static artifacts... or when the user invokes the `FAST-TRACK: <reason>` command.
```

**Step 3: Commit changes**
Run: `git add master-prompt.md; git commit -m "feat: add Fast-Track protocol"`

---

### Task 4: Streamlining Safety Gates

**Files:**
- Modify: `master-prompt.md`

**Step 1: Condense C9 Audit Block**
Modify: `master-prompt.md:99-114`
TargetContent:
Replace the multi-line audit block with:
```text
[Audit | Sync:Y | Red:Y | Plan:Y | Atomic:Y | NoPlace:Y | Skip:N]
```
Add instructions that `Y` means true/met, `N` means false, and `A` means N/A.

**Step 2: Simplify C12 Thinking Protocol**
Modify: `master-prompt.md:130-135`
Target: Focus thinking on logical steps and critical constraint checks only.

**Step 3: Update Snapshot Interval (Section IV)**
Modify: `master-prompt.md:139-141`
Target: Change "turns 10, 20, 30" to "turns 15, 30, 45".

**Step 4: Commit changes**
Run: `git add master-prompt.md; git commit -m "perf: streamline audit and snapshot gates"`

---

### Task 5: Skill & Environment Integration

**Files:**
- Modify: `master-prompt.md`

**Step 1: Update Section V (Skills) for external linkage**
Modify: `master-prompt.md:177-178`
TargetContent:
```markdown
## V. CORE SKILLS
> **IMPORTANT:** Prioritize loading external skill definitions from `.agent/skills/` via `view_file` at the start of any task.
```

**Step 2: Add Windows Path/Command Guidance**
Modify: `master-prompt.md:295-303`
Target: Add a subsection on Windows-First execution (PowerShell commands, `\` paths).

**Step 3: Commit changes**
Run: `git add master-prompt.md; git commit -m "feat: enhance skill linkage and Windows support"`
