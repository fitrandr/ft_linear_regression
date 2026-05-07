SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

.DEFAULT_GOAL := help

SYS_PYTHON ?= python3
VENV_DIR ?= .venv
VENV_BIN := $(VENV_DIR)/bin
VENV_PYTHON := $(VENV_BIN)/python
VENV_PIP := $(VENV_BIN)/pip
PYTHON := $(VENV_PYTHON)
PIP_PACKAGES ?= matplotlib

DATASET ?= data.csv
MODEL ?= model.json

LEARNING_RATE ?= 0.1
ITERATIONS ?= 10000
TEST_RATIO ?= 0.2
SEED ?= 42
LOG_EVERY ?= 1000
EARLY_STOPPING_PATIENCE ?= 300
EARLY_STOPPING_MIN_DELTA ?= 1e-6
VERBOSITY ?= info

MILEAGE ?= 100000
PREDICT_JSON ?= 1

EVAL_JSON ?= 1
EVAL_REPORT ?= evaluation_report.json
INTERPRET_REPORT ?= interpretation_report.txt

PLOT_OUTPUT ?= regression_plot
PLOT_FORMAT ?= png
PLOT_THEME ?= light
PLOT_X_AXIS ?= raw
PLOT_DPI ?= 150
PLOT_REPORT_DIR ?= report_artifacts
PLOT_SHOW ?= 0

C_RESET := \033[0m
C_BOLD := \033[1m
C_BLUE := \033[34m
C_CYAN := \033[36m
C_GREEN := \033[32m
C_YELLOW := \033[33m

TRAIN_CMD = $(PYTHON) train.py --dataset $(DATASET) --model $(MODEL) --learning-rate $(LEARNING_RATE) --iterations $(ITERATIONS) --test-ratio $(TEST_RATIO) --seed $(SEED) --log-every $(LOG_EVERY) --early-stopping-patience $(EARLY_STOPPING_PATIENCE) --early-stopping-min-delta $(EARLY_STOPPING_MIN_DELTA) --verbosity $(VERBOSITY)

ifeq ($(PREDICT_JSON),1)
PREDICT_JSON_FLAG := --json
else
PREDICT_JSON_FLAG :=
endif
PREDICT_CMD = $(PYTHON) predict.py --model $(MODEL) --mileage $(MILEAGE) $(PREDICT_JSON_FLAG)

ifeq ($(EVAL_JSON),1)
EVAL_JSON_FLAG := --json
else
EVAL_JSON_FLAG :=
endif
EVAL_CMD = $(PYTHON) evaluate.py --dataset $(DATASET) --model $(MODEL) $(EVAL_JSON_FLAG) --report $(EVAL_REPORT) --test-ratio $(TEST_RATIO) --seed $(SEED)

ifeq ($(PLOT_SHOW),1)
PLOT_SHOW_FLAG := --show
else
PLOT_SHOW_FLAG :=
endif
PLOT_CMD = $(PYTHON) plot.py --dataset $(DATASET) --model $(MODEL) --output $(PLOT_OUTPUT) --format $(PLOT_FORMAT) --theme $(PLOT_THEME) --x-axis $(PLOT_X_AXIS) --dpi $(PLOT_DPI) --report-dir $(PLOT_REPORT_DIR) $(PLOT_SHOW_FLAG)
INTERPRET_CMD = $(PYTHON) interpret.py --report $(EVAL_REPORT) --output $(INTERPRET_REPORT)

TEST_CMD = $(PYTHON) -m unittest discover -s tests -p 'test_*.py' -q
LINT_CMD = $(PYTHON) -m py_compile train.py predict.py evaluate.py plot.py interpret.py trainer/*.py predictor/*.py evaluator/*.py plotter/*.py interpreter/*.py tests/*.py


define run
	@printf "$(C_CYAN)[cmd]$(C_RESET) %s\n" "$(1)"
	@$(1)
endef


define title
	@printf "\n$(C_BOLD)$(C_BLUE)==> %s$(C_RESET)\n" "$(1)"
endef

.PHONY: help venv deps doctor lint test train predict evaluate interpret plot artifacts makeup all clean fclean re

all: makeup

help:
	@printf "$(C_BOLD)Available targets$(C_RESET)\n"
	@printf "  $(C_GREEN)help$(C_RESET)      Show this help\n"
	@printf "  $(C_GREEN)venv$(C_RESET)      Create virtual environment in $(VENV_DIR)\n"
	@printf "  $(C_GREEN)deps$(C_RESET)      Install dependencies in virtual environment\n"
	@printf "  $(C_GREEN)doctor$(C_RESET)    Check runtime requirements\n"
	@printf "  $(C_GREEN)lint$(C_RESET)      Run py_compile checks\n"
	@printf "  $(C_GREEN)test$(C_RESET)      Run unit tests\n"
	@printf "  $(C_GREEN)train$(C_RESET)     Train model and save $(MODEL)\n"
	@printf "  $(C_GREEN)predict$(C_RESET)   Predict a price for MILEAGE=$(MILEAGE)\n"
	@printf "  $(C_GREEN)evaluate$(C_RESET)  Evaluate model + save $(EVAL_REPORT)\n"
	@printf "  $(C_GREEN)interpret$(C_RESET) Interpret evaluation report into $(INTERPRET_REPORT)\n"
	@printf "  $(C_GREEN)plot$(C_RESET)      Render regression diagnostics plot\n"
	@printf "  $(C_GREEN)artifacts$(C_RESET) Run train + evaluate + plot\n"
	@printf "  $(C_GREEN)makeup$(C_RESET)    Full pipeline: doctor, lint, test, artifacts, predict\n"
	@printf "  $(C_GREEN)clean$(C_RESET)     Remove Python cache files\n"
	@printf "  $(C_GREEN)fclean$(C_RESET)    clean + generated artifacts\n"
	@printf "  $(C_GREEN)re$(C_RESET)        fclean then makeup\n"
	@printf "\n$(C_BOLD)Common variables$(C_RESET): DATASET MODEL LEARNING_RATE ITERATIONS TEST_RATIO SEED MILEAGE PLOT_FORMAT PLOT_THEME PLOT_SHOW VENV_DIR PIP_PACKAGES INTERPRET_REPORT\n"
	@printf "Example: make makeup ITERATIONS=2000 MILEAGE=85000 PLOT_THEME=dark\n"

$(VENV_PYTHON):
	$(call title,Create virtual environment)
	$(call run,$(SYS_PYTHON) -m venv $(VENV_DIR))

venv: $(VENV_PYTHON)
	$(call title,Virtual environment ready)
	@printf "$(C_GREEN)Using venv:$(C_RESET) %s\n" "$(VENV_DIR)"
	@printf "$(C_GREEN)Python path:$(C_RESET) %s\n" "$(VENV_PYTHON)"

deps: venv
	$(call title,Install dependencies in virtual environment)
	$(call run,$(VENV_PIP) install --upgrade pip)
	@if [[ -f requirements.txt ]]; then \
		printf "$(C_CYAN)[cmd]$(C_RESET) %s\n" "$(VENV_PIP) install -r requirements.txt"; \
		$(VENV_PIP) install -r requirements.txt; \
	else \
		printf "$(C_CYAN)[cmd]$(C_RESET) %s\n" "$(VENV_PIP) install $(PIP_PACKAGES)"; \
		$(VENV_PIP) install $(PIP_PACKAGES); \
	fi

doctor: venv
	$(call title,Environment checks)
	@printf "$(C_CYAN)[cmd]$(C_RESET) %s\n" "$(PYTHON) --version"
	@$(PYTHON) --version
	@printf "$(C_CYAN)[cmd]$(C_RESET) %s\n" "$(PYTHON) -c 'import importlib.util as u; print(\"matplotlib=\" + (\"ok\" if u.find_spec(\"matplotlib\") else \"missing\"))'"
	@$(PYTHON) -c 'import importlib.util as u; print("matplotlib=" + ("ok" if u.find_spec("matplotlib") else "missing"))'

lint: venv
	$(call title,Static syntax checks)
	$(call run,$(LINT_CMD))

test: venv
	$(call title,Unit tests)
	$(call run,$(TEST_CMD))

train: venv
	$(call title,Train linear regression model)
	$(call run,$(TRAIN_CMD))

predict: venv
	$(call title,Run prediction)
	$(call run,$(PREDICT_CMD))

evaluate: venv
	$(call title,Evaluate model)
	$(call run,$(EVAL_CMD))

interpret: venv
	$(call title,Interpret evaluation report)
	$(call run,$(INTERPRET_CMD))

plot: deps
	$(call title,Render diagnostics plot)
	$(call run,$(PLOT_CMD))

artifacts: train evaluate interpret plot
	$(call title,Artifacts ready)
	@printf "$(C_GREEN)Generated:$(C_RESET) %s, %s, %s, %s.%s\n" "$(MODEL)" "$(EVAL_REPORT)" "$(INTERPRET_REPORT)" "$(PLOT_OUTPUT)" "$(PLOT_FORMAT)"

makeup:
	$(call title,ML pipeline started)
	@printf "$(C_YELLOW)[1/7] deps$(C_RESET)\n"
	@$(MAKE) --no-print-directory deps
	@printf "$(C_YELLOW)[2/7] doctor$(C_RESET)\n"
	@$(MAKE) --no-print-directory doctor
	@printf "$(C_YELLOW)[3/7] lint$(C_RESET)\n"
	@$(MAKE) --no-print-directory lint
	@printf "$(C_YELLOW)[4/7] test$(C_RESET)\n"
	@$(MAKE) --no-print-directory test
	@printf "$(C_YELLOW)[5/7] artifacts$(C_RESET)\n"
	@$(MAKE) --no-print-directory artifacts
	@printf "$(C_YELLOW)[6/7] predict$(C_RESET)\n"
	@$(MAKE) --no-print-directory predict
	@printf "$(C_YELLOW)[7/7] done$(C_RESET)\n"
	@printf "$(C_GREEN)Pipeline completed successfully.$(C_RESET)\n"

clean:
	$(call title,Clean Python caches)
	$(call run,find . -type d -name '__pycache__' -prune -exec rm -rf {} +)
	$(call run,find . -type f -name '*.pyc' -delete)

fclean: clean
	$(call title,Clean generated artifacts)
	$(call run,rm -f $(MODEL) $(EVAL_REPORT) $(INTERPRET_REPORT))
	$(call run,rm -f regression_plot.png regression_plot.svg regression_plot.pdf)
	$(call run,rm -f $(PLOT_OUTPUT).png $(PLOT_OUTPUT).svg $(PLOT_OUTPUT).pdf)
	$(call run,rm -rf $(PLOT_REPORT_DIR) report)
	$(call run,rm -rf $(VENV_DIR))

re: fclean makeup
