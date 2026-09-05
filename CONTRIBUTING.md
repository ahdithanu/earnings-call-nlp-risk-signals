# Contributing

## Setup

```bash
pip install -e ".[api,dev]"
pre-commit install   # optional but recommended: runs the CI gates locally
```

## Workflow

- `main` is protected: changes land via PRs with green CI (lint, format,
  types, tests on 3.11/3.12, Docker build).
- `make check` runs exactly what CI gates on.
- If you touch the pipeline or `earnings_signals/universe.py`, rebuild and
  validate the data (`make build`), rerun the analyses (`make analyze`),
  regenerate the site (`make site`), and update the README's numbers —
  results in the README are never hand-invented; they come from `results/`.

## Layout

- `earnings_signals/` — the installable package: scoring, isolation,
  attribution, panel features, and the FastAPI service.
- `scripts/` — pipeline and analysis entry points (`python -m scripts.<name>`).
- `results/` — committed analysis outputs the README and site derive from.
- `web/` — self-contained static explorer (`index.template.html` is the
  source; `index.html` is generated).

## Principles

- Coverage gaps are reported, never papered over (flags + logged fallbacks,
  nulls instead of substituted values).
- Findings are stated with their caveats; negative results are kept.
- The universe, thresholds, and headline stats live in exactly one place
  each — change them there and regenerate.
