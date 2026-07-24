PYTHON ?= python3

.PHONY: format-check lint typecheck architecture-check dependency-check
.PHONY: dead-code-check security-check complexity-check image-boundaries-check
.PHONY: exceptions-check quality test

format-check:
	$(PYTHON) -m tools.quality.run_check format

lint:
	$(PYTHON) -m tools.quality.run_check lint

typecheck:
	$(PYTHON) -m tools.quality.run_check typecheck

architecture-check:
	$(PYTHON) -m tools.quality.architecture_check
	$(PYTHON) -m pytest -q tests/test_architecture_check.py tests/test_quality_tooling.py

dependency-check:
	$(PYTHON) -m tools.quality.run_check dependency

dead-code-check:
	$(PYTHON) -m tools.quality.run_check dead-code

security-check:
	$(PYTHON) -m tools.quality.run_check security

complexity-check:
	$(PYTHON) -m tools.quality.run_check complexity

image-boundaries-check:
	$(PYTHON) -m tools.quality.image_boundaries_check

exceptions-check:
	$(PYTHON) -m tools.quality.validate_governance

quality: exceptions-check format-check lint typecheck architecture-check dependency-check dead-code-check security-check complexity-check image-boundaries-check

test:
	$(PYTHON) -m pytest
