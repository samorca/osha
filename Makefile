SHELL := /bin/bash

APP := otgh
CMD := ./cmd/otgh
BIN_DIR := bin
BIN := $(BIN_DIR)/$(APP)

GO ?= go
DOCKER ?= docker
COMPOSE ?= docker compose

VERSION ?= dev
COMMIT ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
BUILD_DATE ?= $(shell date -u +%Y-%m-%dT%H:%M:%SZ)

IMAGE ?= otgh
TAG ?= dev
FULL_IMAGE := $(IMAGE):$(TAG)

GOFLAGS ?=
LDFLAGS ?= -s -w

.PHONY: all
all: fmt vet test build

.PHONY: help
help:
	@echo "OTGH development targets"
	@echo ""
	@echo "  make fmt             Format Go source"
	@echo "  make vet             Run go vet"
	@echo "  make test            Run tests"
	@echo "  make test-race       Run tests with race detector"
	@echo "  make test-cover      Generate test coverage"
	@echo "  make build           Build OTGH binary"
	@echo "  make run             Run OTGH locally"
	@echo "  make tidy            Run go mod tidy"
	@echo "  make verify          fmt-check + vet + test + race"
	@echo "  make docker-build    Build container image"
	@echo "  make compose-up      Start with Docker Compose"
	@echo "  make compose-down    Stop Docker Compose"
	@echo "  make compose-logs    Follow OTGH container logs"
	@echo "  make clean           Remove generated build files"

.PHONY: fmt
fmt:
	$(GO) fmt ./...

.PHONY: fmt-check
fmt-check:
	@test -z "$$(gofmt -l .)" || { \
		echo "The following Go files require formatting:"; \
		gofmt -l .; \
		exit 1; \
	}

.PHONY: vet
vet:
	$(GO) vet ./...

.PHONY: test
test:
	$(GO) test ./...

.PHONY: test-race
test-race:
	$(GO) test -race ./...

.PHONY: test-cover
test-cover:
	$(GO) test -coverprofile=coverage.out ./...
	$(GO) tool cover -func=coverage.out

.PHONY: tidy
tidy:
	$(GO) mod tidy

.PHONY: download
download:
	$(GO) mod download

.PHONY: build
build:
	mkdir -p $(BIN_DIR)
	CGO_ENABLED=0 $(GO) build \
		$(GOFLAGS) \
		-trimpath \
		-ldflags "$(LDFLAGS)" \
		-o $(BIN) \
		$(CMD)

.PHONY: run
run:
	$(GO) run $(CMD)

.PHONY: verify
verify: fmt-check vet test test-race

.PHONY: docker-build
docker-build:
	$(DOCKER) build \
		--build-arg VERSION=$(VERSION) \
		--build-arg COMMIT=$(COMMIT) \
		--build-arg BUILD_DATE=$(BUILD_DATE) \
		-t $(FULL_IMAGE) \
		.

.PHONY: compose-config
compose-config:
	$(COMPOSE) config

.PHONY: compose-up
compose-up:
	$(COMPOSE) up -d --build

.PHONY: compose-down
compose-down:
	$(COMPOSE) down

.PHONY: compose-logs
compose-logs:
	$(COMPOSE) logs -f otgh

.PHONY: compose-ps
compose-ps:
	$(COMPOSE) ps

.PHONY: clean
clean:
	rm -rf $(BIN_DIR)
	rm -f coverage.out
