# Makefile

.PHONY: all run clean myvenv

# Default target: run the pipeline
all: run

# Check or create the virtual environment in 'myvenv', then install requirements
myvenv:
	@echo "Setting up virtual environment 'myvenv' if not existing..."
	@if [ ! -d "myvenv" ]; then \
	  python3 -m venv myvenv; \
	fi
	@. myvenv/bin/activate && pip install -r requirements.txt

# Run your main Python script using the myvenv environment
run: myvenv
	@echo "Running Morph Etymology evaluation..."
	@. myvenv/bin/activate && python main.py --enable_all

# Remove generated .tsv files
clean:
	@echo "Cleaning up mistake and stats files..."
	rm -f mistakes*.tsv morphs*stats.tsv languages*stats.tsv annotator_differences.tsv

# Compute Inter-Annotator Agreement
agreement:
	@echo "Computing Inter Annotator Agreement on annotations/dev.tsv and annotations/dev_annotator2.tsv..."
	@python3 inter_annotator.py
