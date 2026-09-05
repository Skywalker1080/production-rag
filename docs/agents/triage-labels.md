# Triage Labels

The skills speak in terms of five canonical triage roles. This file maps those roles to the actual label strings used in this repo's issue tracker.

| Label in mattpocock/skills | Label in our tracker | Meaning                                  |
| -------------------------- | -------------------- | ---------------------------------------- |
| `needs-triage`             | `To Do`              | Maintainer needs to evaluate this issue  |
| `needs-info`               | `needs-info`         | Waiting on reporter for more information |
| `ready-for-agent`          | `ready-for-agent`    | Fully specified, ready for an AFK agent  |
| `ready-for-human`          | `Production Hardening` | Requires human implementation          |
| `wontfix`                  | `Done`               | Will not be actioned                     |

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), use the corresponding label string from this table.

Edit the right-hand column to match whatever vocabulary you actually use.

> Note: user-provided workflow labels were `To Do, Production Hardening, Done`. Mapped to closest roles above; `needs-info` and `ready-for-agent` kept as defaults since no override was given. `Done` doubles as closed-state for `wontfix` — split into separate `Done` / `wontfix` labels if you need to distinguish completed vs abandoned.
