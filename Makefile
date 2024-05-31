PORT=7860
WORKERS=1

#
# Installation
#

.PHONY: setup-tesseract
setup-tesseract:
	sudo add-apt-repository ppa:alex-p/tesseract-ocr5 -y
	sudo apt-get update
	sudo apt install -y tesseract-ocr
	sudo apt install tesseract-ocr-eng
	tesseract --version

.PHONY: setup
setup:
	pip install -U --no-cache-dir pip setuptools wheel poetry

.PHONY: install
install:
	poetry install --extras tesseract --extras visualization

.PHONY: install-api
install-api:
	poetry install --extras api --extras tesseract --extras visualization

#
# linter/formatter/typecheck/testing
#

.PHONY: lint
lint: install
	poetry run ruff check --output-format=github .

.PHONY: format
format: install
	poetry run ruff format --check --diff .

.PHONY: typecheck
typecheck: install
	poetry run mypy --cache-dir=/dev/null .

#
# Testing
#

.PHONY: test-package
test-package: install
	poetry run pytest -svx --ignore=tests/api/

.PHONY: test-api
test-api: install-api
	poetry run pytest -svx tests/api/

#
# Development
#

.PHONY: run-api
run: install-api
ifeq ($(WORKERS), 1)
	poetry run uvicorn pyaesthetics.api.run:app --port $(PORT) --reload
else
	poetry run uvicorn pyaesthetics.api.run:app --port $(PORT) --workers $(WORKERS)
endif
