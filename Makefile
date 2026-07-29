# Makefile for setting up environment
#################### Read Environment
RUNTIME_FILE := ./config/runtime/runtime.env
PYTHON_FILE  := ./config/runtime/python.env
include $(RUNTIME_FILE)
include $(PYTHON_FILE)

export PACKAGE_INSTALL_NAME

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
	@echo "Commands  : "
	@echo "download  : downloads dependencies distribution"
	@echo "system    : Installs System Libraries per $(PLATFORM_TYPE)"
	@echo "install   : create environment based on project $(PACKAGE_INSTALL_NAME)"
	@echo "format    : formatting and linting of project $(PACKAGE_NAME)"
	@echo "clean     : cleans all files or project $(PACKAGE_INSTALL_NAME)"
	@echo "test      : execute unit testing"

info:
	@echo "Git Root:       $(GIT_ROOT)"
	@echo "Package:        $(PACKAGE_INSTALL_NAME) - $(PACKAGE_NAME)"
	@echo "Platform:       ${PLATFORM_TYPE}"
	@echo "Architecture:   $$(uname -m)"
	@echo "Shell:          $(SHELL)"

info_dotfiles:
	@echo "Dotfiles Repo:      $(DOTFILES_REPO)"
	@echo "Dotfiles Remote:    $(DOTFILES_REMOTE)"
	@echo "Dotfiles Branch:    $(BRANCH)"


#################### Installation
.PHONY: install install_setup install_dotfiles link_dotfiles link_vaultspace

install:
	@echo "Installing package $(PACKAGE_INSTALL_NAME) for development..."
	$(MAKE) install_setup
	$(MAKE) install_dotfiles
	$(MAKE) link_vaultspace

install_setup:
	@echo "Installing Setup for $(PACKAGE_NAME)..."
	mkdir -p .$(PACKAGE_NAME)
	mkdir -p _build config docs templates projects
#	mkdir -p _build config data docs scripts apps projects examples templates
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

# links dotfiles contents individually so existing entries (e.g. .github/workflows) are preserved
link_dotfiles:
	@echo "Linking Dotfiles..."
	@for dir in .vscode .github; do \
		mkdir -p $(GIT_ROOT)/$$dir; \
		find $(GIT_ROOT)/$(DOTFILES_DIR)/$$dir -maxdepth 1 -mindepth 1 | \
		while read src; do \
			dest="$(GIT_ROOT)/$$dir/$$(basename $$src)"; \
			{ [ -d "$$dest" ] && ! [ -L "$$dest" ]; } || ln -sfn "$$src" "$$dest"; \
		done; \
	done

# links to obsidian vaults for contextlib, artifactlib, promptlib
link_vaultspace:
	@echo "Linking Vaultspace..."
	mkdir -p stores
	ln -sfn $(VAULTSPACE_ROOT)/contextlib 	stores/contextlib
	ln -sfn $(VAULTSPACE_ROOT)/artifactlib 	stores/artifactlib
	ln -sfn $(VAULTSPACE_ROOT)/promptlib 	stores/promptlib


#################### Python / uv
.PHONY: conda_config uv_download download_python
.PHONY: install_python uv_sync_project_name
.PHONY: format lint test clean_python run_app

APP ?= examples/apps/studio/app.py

download_python:
	@echo "Downloading Python version $(PYTHON_VERSION) with UV..."
	$(MAKE) conda_config
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

# not currently used
# uv_init:
# 	@echo "Initializing UV for project $(PACKAGE_INSTALL_NAME)..."
# 	@echo "$(PYTHON_VERSION)" > .python-version
# 	@if [ ! -f pyproject.toml ]; then uv init; fi
# 	@uv python install $(PYTHON_VERSION)
# 	@if [ ! -d "$(PYTHON_VENV_DIR)" ]; then uv venv "$(PYTHON_VENV_DIR)" --python $(PYTHON_VERSION); fi
# 	@rm -f main.py

install_python:
	@echo "Installing Python environment with uv..."
	uv python install $(PYTHON_VERSION)
	@echo "$(PYTHON_VERSION)" > .python-version
	$(MAKE) uv_sync_project_name
	uv venv $(PYTHON_VENV_DIR) --python $(PYTHON_VERSION) --prompt "$(PYTHON_VENV_KERNEL_NAME)"
	source $(PYTHON_VENV_DIR)/bin/activate && uv sync --all-extras --active
	uv pip install -e .
	uv pip install --upgrade pip ipython ipykernel
	uv run python -m ipykernel install --user --name=$(PYTHON_VENV_KERNEL_NAME)
	@echo "UV version: $$(uv --version)"

uv_sync_project_name:
	@test -f "$(RUNTIME_FILE)" || { echo "Missing $(RUNTIME_FILE)"; exit 1; }
	@test -f "$(PYTHON_FILE)"  || { echo "Missing $(PYTHON_FILE)"; exit 1; }
	@echo "Using PACKAGE_INSTALL_NAME=$(PACKAGE_INSTALL_NAME)"
	@echo "Before: $$(grep -E '^name[[:space:]]*=' pyproject.toml)"
	@if [ "$(PLATFORM_TYPE)" = "Darwin" ]; then \
		sed -E -i '' "s|^name[[:space:]]*=.*|name = \"$(PACKAGE_INSTALL_NAME)\"|" pyproject.toml; \
	else \
		sed -E -i "s|^name[[:space:]]*=.*|name = \"$(PACKAGE_INSTALL_NAME)\"|" pyproject.toml; \
	fi
	@echo "After : $$(grep -E '^name[[:space:]]*=' pyproject.toml)"

run_app:
	@echo "Launching Streamlit app with auto-reload..."
	uv run --group apps streamlit run $(APP) \
		--server.runOnSave true \
		--server.fileWatcherType watchdog

format:
	@echo "Formatting $(PACKAGE_NAME)..."
	uv run ruff format .

lint:
	@echo "Linting $(PACKAGE_NAME)..."
	uv run ruff check .

test:
	@echo "Running tests for $(PACKAGE_NAME)..."
	uv run pytest

clean_python:
	@echo "Cleaning Python artifacts for $(PACKAGE_INSTALL_NAME)..."
#	rm -rf $(PYTHON_VENV_DIR)
	rm -rf .pytest_cache dist
	find . -not -path './.git/*' -type d -name "__pycache__" -exec rm -rf {} +
	find . -not -path './.git/*' -type f -name "*.pyc" -delete


#################### Minimal Agents and Skills
.PHONY: setup_agent setup_agent_claude
.PHONY: install_agents install_local_skills

install_agents:
	@echo "Installing agents setup..."
	$(MAKE) setup_agent
	$(MAKE) setup_agent_claude
	$(MAKE) install_local_skills
	$(MAKE) -C $(SKILLS_DIR) install_plugin_local PROJECT_DIR=$(GIT_ROOT)
	$(MAKE) -C $(SKILLS_DIR) install_marketplace_claude
	$(MAKE) -C $(SKILLS_DIR) install_plugin_claude PROJECT_DIR=$(GIT_ROOT)

install_local_skills:
	@echo "Installing agent-skills from $(SKILLS_REPO)..."
	@if [ ! -d $(SKILLS_DIR) ]; then \
		git clone $(SKILLS_REPO) $(SKILLS_DIR); \
	fi

setup_agent:
	@echo "Installing agents setup..."
# General setup
	mkdir -p .agents
	touch AGENTS.md

setup_agent_claude:
	@echo "Installing Claude Coding Agent..."
# repopulate claude directories
	mkdir -p .claude
	mkdir -p .claude-plugin
	touch CLAUDE.md
	echo @AGENTS.md > CLAUDE.md
	touch .claude/CLAUDE.md
	touch .claude/settings.json
	touch .claude/settings.local.json


#################### MCP Servers
PROJECT_DIR ?= $(GIT_ROOT)
.PHONY: install_mcp_servers

install_mcp_servers:
	@echo "Installing MCP Servers..."
	@echo "MCP project scope directory: $(PROJECT_DIR)"
	@cd $(PROJECT_DIR); \
	for entry in $(MCP_SERVERS); do \
		name=$$(echo $$entry | cut -d@ -f1); \
		url=$$(echo $$entry | cut -d@ -f2-); \
		claude mcp add --transport http --scope project $$name $$url; \
	done; \
	for entry in $(MCP_SERVERS_STDIO); do \
		name=$$(echo $$entry | cut -d'^' -f1); \
		command=$$(echo $$entry | cut -d'^' -f2); \
		args_csv=$$(echo $$entry | cut -d'^' -f3-); \
		args=$$(echo $$args_csv | tr ',' ' '); \
		claude mcp add --scope project $$name -- $$command $$args; \
	done


#################### Development CLIs
.PHONY: install_github_cli install_codex_cli install_claude_cli

install_github_cli:
	@echo "Installing GitHub CLIs for Development..."
# gitub copilot
	npm install -g @github/copilot
	copilot update
	@echo "Github Copilot Version: $$(copilot --version)"
# github agentic workflows
	gh extension install github/gh-aw
	gh extension upgrade aw
	@echo "Github Agentic Workflow Version: $$(gh aw --version)"
	gh aw add githubnext/agentics/daily-repo-status

install_claude_cli:
	@echo "Installing Claude CLI for Development..."
# npm install -g @anthropic-ai/claude-code
	curl -fsSL https://claude.ai/install.sh | bash
	@echo "Claude CLI Version: $$(claude --version)"

install_codex_cli:
	@echo "Installing OpenAI Codex CLI for Development..."
	npm install -g @openai/codex
	@echo "OpenAI Codex CLI Version: $$(codex --version)"

#################### General
.PHONY: clean
clean:
	@echo "Cleaning project files for installed package ..."
	$(MAKE) clean_python
	find . -not -path './.git/*' -type d -name "outputs" -exec rm -rf {} +
	find . -not -path './.git/*' -name "*.out" -delete
	find . -not -path './.git/*' -name ".DS_Store" -delete
