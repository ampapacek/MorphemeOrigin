# Makefile

.PHONY: all run clean venv venv_recreate agreement

PYTHON_BIN ?= $(shell command -v python3.12 || command -v python3.11 || command -v python3)

# Default target: run the pipeline
all: run

# Check or create the virtual environment, then install requirements
venv:
	@if [ -d "MorphOriginVenv" ]; then \
	  VENV_PY=$$(MorphOriginVenv/bin/python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"); \
	  if [ "$$VENV_PY" = "3.14" ]; then \
	    echo "Existing MorphOriginVenv uses Python $$VENV_PY, which triggers scipy source build (requires gfortran)."; \
	    echo "Run 'make venv_recreate' to rebuild with $(PYTHON_BIN)."; \
	    exit 1; \
	  fi; \
	  . MorphOriginVenv/bin/activate && pip install -r requirements.txt; \
	else \
	  echo "Setting up virtual environment 'MorphOriginVenv' using $(PYTHON_BIN)"; \
	  $(PYTHON_BIN) -m venv MorphOriginVenv; \
	  . MorphOriginVenv/bin/activate && pip install -r requirements.txt; \
	fi

venv_recreate:
	@echo "Recreating virtual environment 'MorphOriginVenv' using $(PYTHON_BIN)"
	rm -rf MorphOriginVenv
	@$(PYTHON_BIN) -m venv MorphOriginVenv
	@. MorphOriginVenv/bin/activate && pip install -r requirements.txt

# Run the main Python script using the MorphOriginVenv environment
run: venv
	@echo "Running Morph Etymology evaluation..."
	@. MorphOriginVenv/bin/activate && python3 src/main.py --enable_all

# Remove generated .tsv files
clean:
	@echo "Cleaning up mistake, stats files and outputs directory..."
	rm -f mistakes*.tsv morphs*stats.tsv languages*stats.tsv annotator_differences.tsv languages*stats_extended.tsv morphs*stats_extended.tsv
	@if [ -d "outputs" ]; then rm -f outputs/*; fi
# Compute Inter-Annotator Agreement
agreement: venv
	@echo "Computing Inter Annotator Agreement on data/annotations/dev.tsv and data/annotations/dev_annotator2.tsv..."
	@. MorphOriginVenv/bin/activate && python3 src/inter_annotator.py
