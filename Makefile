# Makefile for setting up environment
#################### Read Environment
RUNTIME_FILE := ./config/runtime/runtime.env
PYTHON_FILE  := ./config/runtime/python.env
include $(RUNTIME_FILE)
include $(PYTHON_FILE)

#################### Makefile Configuration
GIT_ROOT ?= $(shell git rev-parse --show-toplevel)
# e.g., Darwin for MacOS
PLATFORM_TYPE = $(shell uname)
# dynamically detect shell type as bash or zsh
ifeq ($(shell basename $(SHELL)), zsh)
        SHELL := zsh
		SHELL_CONFIG := $(HOME)/.zshrc
else
        SHELL := bash
		SHELL_CONFIG := $(HOME)/.bashrc
endif

#################### Makefile Context
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := info

.PHONY: help info info_dotfiles
help:
	@echo "Commands  		: "
	@echo "download  		: downloads dependencies distribution"
	@echo "system    		: Installs System Libraries per $(PLATFORM_TYPE)"
	@echo "install   		: create environment for workspace $(PACKAGE_NAME)"
	@echo "format    		: formatting and linting of workspace $(PACKAGE_NAME)"
	@echo "lint      		: lint workspace $(PACKAGE_NAME) with ruff"
	@echo "typecheck 		: type-check workspace $(PACKAGE_NAME) with pyright"
	@echo "clean     		: cleans all files for workspace $(PACKAGE_NAME)"
	@echo "test      		: execute unit testing"
	@echo "install_kernel 	: create a named, persistent Jupyter kernel + venv for extra deps (KERNEL_NAME=, KERNEL_DEPS=)"

info:
	@echo "Git Root:       $(GIT_ROOT)"
	@echo "Workspace:      $(PACKAGE_NAME)"
	@echo "Platform:       ${PLATFORM_TYPE}"
	@echo "Architecture:   $$(uname -m)"
	@echo "Shell:          $(SHELL)"

info_dotfiles:
	@echo "Dotfiles Repo:      $(DOTFILES_REPO)"
	@echo "Dotfiles Remote:    $(DOTFILES_REMOTE)"
	@echo "Dotfiles Branch:    $(BRANCH)"


#################### Installation
.PHONY: install install_setup

install:
	@echo "Installing workspace $(PACKAGE_NAME) for development..."
	$(MAKE) install_setup
	$(MAKE) install_dotfiles
	$(MAKE) install_python

install_setup:
	@echo "Installing Setup for $(PACKAGE_NAME)..."
	mkdir -p .$(PACKAGE_NAME)
	mkdir -p _build config data docs scripts templates examples apps packages
	touch .env.template
	touch docs/.gitkeep

install_dotfiles:
	@echo "Installing Dotfiles from $(DOTFILES_REPO)..."
	@if [ ! -d $(DOTFILES_DIR) ]; then \
		git clone $(DOTFILES_REPO) $(DOTFILES_DIR); \
	else \
		echo "Updating Dotfiles..."; \
		git -C $(DOTFILES_DIR) pull --ff-only; \
	fi
	$(MAKE) link_dotfiles

# links dotfiles contents individually so any existing entries are preserved
link_dotfiles:
	@echo "Linking Dotfiles..."
	@for dir in .vscode; do \
		mkdir -p $(GIT_ROOT)/$$dir; \
		find $(GIT_ROOT)/$(DOTFILES_DIR)/$$dir -maxdepth 1 -mindepth 1 | \
		while read src; do \
			dest="$(GIT_ROOT)/$$dir/$$(basename $$src)"; \
			{ [ -d "$$dest" ] && ! [ -L "$$dest" ]; } || ln -sfn "$$src" "$$dest"; \
		done; \
	done


#################### Python / uv
.PHONY: download_python conda_config uv_download
.PHONY: install_python
.PHONY: uv_install_python
.PHONY: clean_python

install_python:
	$(MAKE) download_python
	$(MAKE) uv_install_python

download_python:
	@echo "Downloading Python version $(PYTHON_VERSION) with UV..."
#	$(MAKE) conda_config
	$(MAKE) uv_download

conda_config:
	@if command -v conda >/dev/null 2>&1; then \
		echo "Configuration of Conda Environment..."; \
		conda config --set ssl_verify false; \
		conda config --set auto_activate_base false; \
		conda deactivate; \
	else \
		echo "Conda Environment not present; skipping conda_config."; \
	fi

uv_download:
	@echo "Installing UV package manager..."
	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv   self update
	@echo "UV version: $$(uv --version)"

uv_install_python:
	@echo "Installing Python environment with uv..."
	uv python install $(PYTHON_VERSION)
	@echo "$(PYTHON_VERSION)" > .python-version
	uv venv $(PYTHON_VENV_DIR) --python $(PYTHON_VERSION) --prompt "$(PYTHON_VENV_KERNEL_NAME)"
	source $(PYTHON_VENV_DIR)/bin/activate && uv sync --all-extras --active
	uv pip install --upgrade pip ipython ipykernel
	uv run python -m ipykernel install --user --name=$(PYTHON_VENV_KERNEL_NAME)
	@echo "UV version: $$(uv --version)"

clean_python:
	@echo "Cleaning Python artifacts for workspace $(PACKAGE_NAME)..."
#	rm -rf $(PYTHON_VENV_DIR)
	rm -rf .pytest_cache dist
	find . -not -path './.git/*' -type d -name "__pycache__" -exec rm -rf {} +
	find . -not -path './.git/*' -type f -name "*.pyc" -delete




#################### Utilties
.PHONY: format lint typecheck test clean run_app install_kernel
APP ?= examples/apps/studio/app.py
# KERNEL_NAME/KERNEL_DEPS/KERNEL_VENV_DIR/KERNEL_ID/KERNEL_DISPLAY defaults come from
# $(RUNTIME_FILE) — override per invocation, e.g.:
#   make install_kernel KERNEL_NAME=nlp KERNEL_DEPS="transformers torch"

format:
	@echo "Formatting $(PACKAGE_NAME)..."
	uv run ruff format .

typecheck:
	@echo "Type checking $(PACKAGE_NAME)..."
	uv run pyright

lint:
	@echo "Linting $(PACKAGE_NAME)..."
	uv run ruff check .

test:
	@echo "Running tests for $(PACKAGE_NAME)..."
	uv run pytest

clean:
	@echo "Cleaning project files for installed package ..."
	$(MAKE) clean_python
	find . -not -path './.git/*' -type d -name "outputs" -exec rm -rf {} +
	find . -not -path './.git/*' -name "*.out" -delete
	find . -not -path './.git/*' -name ".DS_Store" -delete

run_app:
	@echo "Launching Streamlit app with auto-reload..."
	uv run --group apps streamlit run $(APP) \
		--server.runOnSave true \
		--server.fileWatcherType watchdog

install_kernel:
	@echo "Creating persistent venv '$(KERNEL_VENV_DIR)' and Jupyter kernel '$(KERNEL_ID)' with: $(KERNEL_DEPS)..."
	uv venv $(KERNEL_VENV_DIR) --python $(PYTHON_VERSION) --clear
	uv pip install --python $(KERNEL_VENV_DIR)/bin/python \
		-e packages/velari-core -e packages/velari-data $(KERNEL_DEPS) ipykernel
	$(KERNEL_VENV_DIR)/bin/python -m ipykernel install --user --name=$(KERNEL_ID) --display-name="$(KERNEL_DISPLAY)"
	@echo "Select kernel '$(KERNEL_DISPLAY)' in VS Code/Jupyter to run notebooks that need: $(KERNEL_DEPS)"
