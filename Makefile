.PHONY: env train eval plots all test

env:
conda env create -f environment.yml || true

train:
python run.py train --config configs/final.yaml

eval:
python run.py eval --config configs/final.yaml

plots:
python run.py plots --config configs/final.yaml

all:
python run.py all --config configs/final.yaml

test:
pytest -q
