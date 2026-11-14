# Contributing to PyCapital

PyCapital is a Python library that gives unified, validated access to Indian market data (NSE, BSE, AMFI, SEBI). Everyone follows these rules, so every pull request looks the same and review stays quick.

## Git rules

1. Never work directly on `main`. `main` must always be the working, demo-ready version.
2. Each member has one branch: `a-work`, `b-work`, `c-work`. Commit often with short messages.
3. Commit messages start with a type: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`.
4. When a chunk of work is finished, open a pull request from your branch into `main`. One teammate reviews and approves it. Then use **Squash and merge**.
5. After every merge, everyone updates their branch:
   `git checkout <your-branch>` then `git pull origin main`.
6. Review each other's work, not your own. Review within one day so nobody is blocked.

## File ownership

Each file has one owner, to avoid merge conflicts. If you need a change in someone else's file, ask the owner or mention it in your PR description. Do not edit it quietly.

- **A:** `nse.py`, `bse.py`, `_session_utils.py`, `symbol_map.py`, `pyproject.toml`
- **B:** `amfi.py`, `sebi.py`, `schema.py`, `tests/fixtures/`
- **C:** `fno.py`, `corporate.py`, `indices.py`

## Connector standard

Every connector in `src/pycapital/connectors/` follows these rules:

- Dates passed in are always `YYYY-MM-DD`. Convert to the exchange's own format inside the connector.
- No `print`. Use `logger = logging.getLogger(__name__)`. The library is silent unless the user enables logging.
- No test or demo code inside connector files. Tests live in `tests/`, demos in `demo/`.
- A connector never raises on a normal failure (network error, bad symbol, empty result). It logs the problem and returns `empty_frame("<dataset>")` from `pycapital.schema`.
- Output columns and dtypes come from `schema.py`, the single source of truth.
- Long date ranges are chunked so exchange row caps never silently truncate data.
- Every public function has a docstring with `Args` and `Returns`.

## Tests

- `tests/unit/`: offline tests, run with `pytest`. They must pass without internet.
- `tests/live/`: tests that call the real NSE and BSE servers. Run them manually before opening a PR.

## Pull request checklist

Copy this into every PR description:

- [ ] No `print`, no `sys.path` hacks, no leftover test or exam edits
- [ ] Dates are `YYYY-MM-DD`
- [ ] Failure returns `empty_frame(...)`, nothing raises
- [ ] Docstring with Args and Returns is present
- [ ] `pytest` passes
- [ ] Live tests were run and the results match the previous run
- [ ] I touched only files I own, or the owner agreed