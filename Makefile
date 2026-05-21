PYTHON     := python3
PYTHONPATH := PYTHONPATH=src

.DEFAULT_GOAL := help

.PHONY: help install test lint format crawl clean docker-build docker-run

help: ## Show available commands
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install: ## Install dependencies and pre-commit hooks
	pip install -e ".[dev]"
	pre-commit install
	pre-commit install --hook-type commit-msg

test: ## Run all tests
	$(PYTHONPATH) $(PYTHON) -m pytest tests/ -v

lint: ## Run ruff linter
	ruff check src/ tests/

format: ## Format code with ruff
	ruff format src/ tests/

crawl: ## Run the crawler (default 500 articles per source)
	$(PYTHONPATH) $(PYTHON) -m sri.crawler --max-articles 500

clean: ## Remove cache and compiled files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache

docker-build: ## Build Docker image (requires Dockerfile — Corte 3)
	docker build -t sri-tecnologia .

docker-run: ## Run system in Docker container (requires Dockerfile — Corte 3)
	docker run -p 7860:7860 sri-tecnologia
