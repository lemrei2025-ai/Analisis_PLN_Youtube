SOURCE ?= sample
PY ?= python

.PHONY: install sample pipeline extract test lint clean

install:
	pip install -r requirements.txt && pip install -e .

sample:
	$(PY) scripts/make_sample_data.py

pipeline:
	$(PY) -m ytnlp.pipeline --source $(SOURCE)

extract:
	$(PY) -m ytnlp.data.youtube_api --from-kaggle

test:
	pytest -q

lint:
	ruff check src tests scripts

clean:
	rm -rf data/interim/* data/processed/* reports/*
	touch data/interim/.gitkeep data/processed/.gitkeep reports/.gitkeep
