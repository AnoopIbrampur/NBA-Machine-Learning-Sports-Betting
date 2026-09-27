# Results before the 2024-26 leakage fix (superseded)

These files are the results as reported before `Get_Data.py` was fixed. At that point
`dataset_2012-26` rows for 2024-25 and 2025-26 included the game being predicted, so the
**baseline's** validation numbers, and 156 of its test games, are inflated here.

- Pre-game models: unaffected by the leak. Their numbers differ from the corrected run only
  because the regenerated dataset no longer has rows for opening-night games, which have no
  prior-game snapshot.
- Numbers on test games from never-affected seasons (2021-22 to 2023-24) remain valid.

Corrected results are in the parent directory. See `docs/findings.md`.
