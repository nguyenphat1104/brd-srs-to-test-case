# Direction approved

- User selection: "b" (2026-08-17)
- Direction: Timeline canvas — source selection, live generation, and final result progress down one page; artifacts stay in a side panel.
- Reviewed prototypes: `design-demos/seamless-flow-guided-workspace.html`, `design-demos/seamless-flow-timeline.html`, and `design-demos/seamless-flow-console.html`.
- Assumption: retain the existing white/slate/blue theme, real agent activity, task details, model assignments, and artifact payloads; do not show private model reasoning.
- Iteration: published artifacts are highlighted within their agent update and open in an overlay drawer; no empty fixed artifact panel.

## Redesign iteration (2026-08-21)

- User instruction (verbatim): "use huashu design to refactor/redesign the whole 'brd-srs-to-test-case/src' ui/ux more intuitive. you decide dont ask me" — then: "you always stuck at cli run. please use ui-ux-promax to redesign".
- Exemption from the three-direction gate: user explicitly delegated the decision ("you decide dont ask me") and asked to skip the CLI/screenshot loop. Recorded here per the gate-file protocol.
- Note: no `ui-ux-promax` skill is installed; the loaded `huashu-design` skill drives this redesign.
- Scope reality: `src/brd_srs_testgen/` is the pure Python research core (no UI). The entire UI/UX lives in `app.py` (Streamlit). Redesign therefore targets `app.py`'s presentation layer only; the `src/` core and all behavior are untouched.
- Chosen direction (decided by the designer, per user delegation): **refine the approved Timeline canvas in place**, keeping the white/slate/blue theme and every test-asserted label/key/structure. Concrete UX upgrades:
  1. **Real timeline spine** — the 3 run stages (Select source → Live generation → Validated result) are connected by a vertical spine with numbered/checkpoint nodes (from the approved `seamless-flow-timeline.html` prototype), replacing the current disconnected stage rows.
  2. **App bar** — brand mark + product name + Settings in a proper top bar, replacing two floating buttons.
  3. **Runs home hero** — kicker + headline + primary CTA; history table gains colored status pills (Completed / Failed / Interrupted).
  4. **Detail page pipeline strip** — Requirements → Scenarios → Test cases shown as a connected flow with counts, so the artifact structure is visible at a glance.
  5. **Settings dialog** — sectioned (Provider / Centralized agents / Limits) with clearer hierarchy.
- Hard constraint: every label, widget key, session-state key, and structure asserted in `tests/test_app.py` stays byte-identical; verification is `pytest` (no CLI screenshot loop).
