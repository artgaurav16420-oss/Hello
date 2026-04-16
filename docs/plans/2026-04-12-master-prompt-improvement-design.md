# Design: Master Prompt v11.0 (Antigravity Edition)

## Goal
To evolve the `master-prompt.md` from the "Superpowers v10.0" era into the "Antigravity v11.0" era, focusing on identity alignment, external skill integration, and operational efficiency through the introduction of a Fast-Track protocol and streamlined safety gates.

## User Review Required
> [!IMPORTANT]
> The introduction of the `FAST-TRACK` command allows skipping TDD for static assets. This is a deliberate trade-off of "integrity-by-default" for "speed-on-demand."

## Proposed Changes

### 1. Identity & Persona
- **Persona**: All references to "Superpowers" changed to "Antigravity".
- **Version**: Updated to `v11.0`.
- **Standby Message**: Updated termination protocol to reflect Antigravity.

### 2. Operational Efficiency
- **Fast-Track Protocol**: New command `FAST-TRACK <reason>` and corresponding Skill exception in Section III.
- **Audit Compactness (C9)**: Redesign the audit table into a single-line format:
  `[Audit | Sync:Y | Red:Y | Plan:Y | Atomic:Y | NoPlace:Y | Skip:N]`
- **Thinking Protocol (C12)**: Focused on the logic of the change, removing redundant state headers.
- **Snapshot Interval**: Updated `SAVE STATE` frequency to 15 turns.

### 3. Skill & Env Integration
- **Context Awareness**: Explicitly prioritize `.agent/skills/` over internal Section V summaries when ambiguity exists.
- **Windows Support**: Added instruction to prefer PowerShell-compatible syntax and Windows pathing (`\`) within workspace commands.

## Architecture & Data Flow
No major architectural changes; this is a state-machine and instruction-set refinement.

## Success Criteria
1.  **Identity**: Agent correctly identifies as Antigravity v11.0.
2.  **Fast-Track**: Successfully bypasses TDD for a CSS or Doc change when invoked.
3.  **Conciseness**: Multi-line audit blocks are replaced by the compact one-liner.
4.  **Skills**: Agent correctly prioritizes external skill definitions.
