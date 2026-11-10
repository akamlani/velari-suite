# Makefile for velari-harness package
#################### Read Environment
GIT_ROOT     ?= $(shell git rev-parse --show-toplevel)
RUNTIME_FILE := $(GIT_ROOT)/config/runtime/runtime.env
PYTHON_FILE  := $(GIT_ROOT)/config/runtime/python.env
include $(RUNTIME_FILE)
include $(PYTHON_FILE)

#################### Makefile Configuration
# override runtime.env's shared PACKAGE_NAME=velari with this package's own name
PACKAGE_NAME := velari-harness
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
