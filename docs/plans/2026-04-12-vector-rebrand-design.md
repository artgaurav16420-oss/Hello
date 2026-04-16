# Design: Persona Rebrand — Vector v12.0

## Goal
Rebrand the "Antigravity" persona to **Vector** to improve intuitive understanding of the protocol's directional and disciplined nature.

## User Review Required
> [!IMPORTANT]
> **Command Change**: The initialization command will change from `Initialize Antigravity` to **`Initialize Vector`**. Any existing scripts or macros using the old command must be updated.

## Proposed Changes

### 1. Identify Re-branding
- Replace all instances of `Antigravity` with `Vector` in `master-prompt.md`.
- Handle cases for `Vector` (Proper), `VECTOR` (All-Caps Intake), and `vector` (Lower-case standby).

### 2. Header & Command Reference
- Update the Section I identity definition.
- Update Section VI command reference table.
- Update Section VII initialization trigger and response.
- Update Section VIII termination summary.

## Success Criteria
1. Grep search for "Antigravity" returns zero results in `master-prompt.md`.
2. Initialization command `Initialize Vector` works correctly in Section VII logic.
3. Termination label shows `Vector v12.0 Standby`.
