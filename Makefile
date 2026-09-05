# Developer entry points. Everything here is what CI runs, so `make check`
# passing locally means the PR gates will pass too.

.PHONY: install test lint format typecheck check build validate analyze site serve docker

install:            ## editable install with api+dev extras
	pip install -e ".[api,dev]"

test:               ## unit tests with coverage
	python -m pytest tests/ --cov --cov-report=term-missing

lint:               ## ruff lint + format check
	ruff check .
	ruff format --check .

format:             ## auto-fix lint + reformat
	ruff check --fix .
	ruff format .

typecheck:          ## mypy over the package
	mypy

check: lint typecheck test   ## everything CI gates on

build:              ## rebuild the feature parquet (downloads dataset on first run)
	python -m scripts.build_features
	python -m scripts.validate_features

validate:           ## data-quality gate on the committed parquet
	python -m scripts.validate_features

analyze:            ## rerun every analysis into results/
	python -m scripts.analyze_uncertainty_growth
	python -m scripts.analyze_execqa_robustness
	python -m scripts.analyze_exec_roles
	python -m scripts.latest_signals

site:               ## regenerate the self-contained explorer
	python -m scripts.export_web_data

serve:              ## run the scoring API locally
	uvicorn earnings_signals.api:app --reload

docker:             ## build and smoke-test the API image
	docker build -t earnings-signals-api .
	docker run -d --rm -p 8000:8000 --name esapi earnings-signals-api
	sleep 5 && curl -sf localhost:8000/healthz && docker stop esapi
