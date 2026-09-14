.DEFAULT_GOAL := help
.PHONY: help install data train-baseline train-solution eval eval-baseline eval-solution prompt analyze analyze-baseline clean

PYTHON     ?= python
DATA_DIR   ?= data
N          ?= 10000
TEST_FRAC  ?= 0.1
SEED       ?= 0
PROMPT     ?= 3344x1119

help:
	@echo "make install             pip install -r requirements.txt"
	@echo "make data                generate train/test data ($(N) examples, seed $(SEED))"
	@echo "make train-baseline      train the plain 'a*b=' -> answer variant"
	@echo "make train-solution      train the detailed long-multiplication scratchpad variant"
	@echo "make eval                evaluate the default checkpoint (checkpoints/solution)"
	@echo "make eval-baseline       evaluate checkpoints/baseline on data/test.jsonl"
	@echo "make eval-solution       evaluate checkpoints/solution on data/test.jsonl"
	@echo "make prompt PROMPT=AxB   run a single prediction, e.g. make prompt PROMPT=3344x1119"
	@echo "make analyze             carry-density + qualitative breakdown for checkpoints/solution"
	@echo "make analyze-baseline    same, for checkpoints/baseline"
	@echo "make clean               remove generated data/ and checkpoints/ (asks first)"

install:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) generate_dataset.py --n $(N) --test-frac $(TEST_FRAC) --seed $(SEED) --out $(DATA_DIR)

train-baseline:
	$(PYTHON) train.py --variant baseline --train $(DATA_DIR)/train.jsonl --test $(DATA_DIR)/test.jsonl

train-solution:
	$(PYTHON) train.py --variant solution --train $(DATA_DIR)/train.jsonl --test $(DATA_DIR)/test.jsonl

eval: eval-solution

eval-baseline:
	MATH_NLP_CKPT_DIR=checkpoints/baseline $(PYTHON) run.py --data $(DATA_DIR)/test.jsonl

eval-solution:
	MATH_NLP_CKPT_DIR=checkpoints/solution $(PYTHON) run.py --data $(DATA_DIR)/test.jsonl

prompt:
	$(PYTHON) run.py --prompt $(PROMPT)

analyze:
	MATH_NLP_CKPT_DIR=checkpoints/solution $(PYTHON) analyze.py --data $(DATA_DIR)/test.jsonl

analyze-baseline:
	MATH_NLP_CKPT_DIR=checkpoints/baseline $(PYTHON) analyze.py --data $(DATA_DIR)/test.jsonl

clean:
	@echo "this deletes $(DATA_DIR)/ and checkpoints/ -- ctrl-c to abort, enter to continue"
	@read _
	rm -rf $(DATA_DIR) checkpoints
