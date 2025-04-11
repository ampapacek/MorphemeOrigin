# Makefile

.PHONY: all run clean venv agreement

# Default target: run the pipeline
all: run

# Check or create the virtual environment, then install requirements
venv:
	@if [ -d "MorphOriginVenv" ]; then \
	  . MorphOriginVenv/bin/activate; \
	else \
	  echo "Setting up virtual environment 'MorphOriginVenv'"; \
	  python3 -m venv MorphOriginVenv; \
	  . MorphOriginVenv/bin/activate && pip install -r requirements.txt; \
	fi

# Run the main Python script using the MorphOriginVenv environment
run: venv
	@echo "Running Morph Etymology evaluation..."
	@. MorphOriginVenv/bin/activate && python3 main.py --enable_all

# Remove generated .tsv files
clean:
	@echo "Cleaning up mistake and stats files..."
	rm -f mistakes*.tsv morphs*stats.tsv languages*stats.tsv annotator_differences.tsv

# Compute Inter-Annotator Agreement
agreement:
	@echo "Computing Inter Annotator Agreement on annotations/dev.tsv and annotations/dev_annotator2.tsv..."
	@python3 inter_annotator.py