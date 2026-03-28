## Commit Hygiene

- Keep commits clean, bounded, and purpose-specific.
- Prefer one logical change per commit. Do not mix unrelated fixes, refactors, docs updates, and test rewrites unless they are tightly coupled.
- When work spans multiple concerns, split it into a short chain of commits with readable messages.
- Before committing, check that the diff matches the stated purpose of the commit and does not include unrelated workspace noise.
- If a change is exploratory or lower confidence, prefer using a separate branch until it is trusted.