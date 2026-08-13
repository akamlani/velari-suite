# Makefile for velari-ai package
#################### Read Environment
GIT_ROOT     ?= $(shell git rev-parse --show-toplevel)
RUNTIME_FILE := $(GIT_ROOT)/config/runtime/runtime.env
PYTHON_FILE  := $(GIT_ROOT)/config/runtime/python.env
include $(RUNTIME_FILE)
include $(PYTHON_FILE)

#################### Makefile Configuration
# override runtime.env's shared PACKAGE_NAME=velari with this package's own name
PACKAGE_NAME := velari-ai
PROJECT_DIR  ?= $(GIT_ROOT)
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

.PHONY: help info
help:
	@echo "Commands  		: "
	@echo "info      		: display package/environment info"

info:
	@echo "Git Root:       $(GIT_ROOT)"
	@echo "Package:        $(PACKAGE_NAME)"
	@echo "Platform:       ${PLATFORM_TYPE}"
	@echo "Architecture:   $$(uname -m)"
	@echo "Shell:          $(SHELL)"

#################### MCP
.PHONY: list_mcp install_mcp_servers
list_mcp:
	@echo "Listing MCP Modudles..."
	claude mcp list

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
