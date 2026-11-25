SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c

BINARY ?= manager
BIN_DIR ?= bin

IMG ?= osha-operator-lifecycle:dev
PRODUCTION_IMG ?= ghcr.io/samorca/osha-operator-lifecycle:v0.1.0

NAMESPACE ?= osha-lifecycle-system

CONTAINER_TOOL ?= docker
CONTAINERD_TOOL ?= ctr
SUDO ?= sudo
DOCKER_BUILD_NETWORK ?= host
KUBECTL ?= kubectl
CONTROLLER_GEN ?= controller-gen

.PHONY: all
all: build

.PHONY: help
help:
	@echo ""
	@echo "OSHA Operator Lifecycle Manager"
	@echo ""
	@echo "Development:"
	@echo "  make fmt              Format Go code"
	@echo "  make vet              Run go vet"
	@echo "  make tidy             Update Go modules"
	@echo "  make generate         Generate DeepCopy code"
	@echo "  make manifests        Generate CRDs and RBAC"
	@echo "  make test             Run unit tests"
	@echo "  make build            Build manager binary"
	@echo "  make run              Run manager locally"
	@echo ""
	@echo "Images:"
	@echo "  make docker-build     Build lifecycle image"
	@echo "  make docker-build-k8s Build image and import it into Kubernetes containerd"
	@echo "  make docker-push      Push lifecycle image"
	@echo ""
	@echo "Kubernetes:"
	@echo "  make install          Install lifecycle CRDs"
	@echo "  make render-dev       Render development manifests"
	@echo "  make render-prod      Render production manifests"
	@echo "  make deploy-dev       Deploy development manager"
	@echo "  make deploy-prod      Deploy production manager"
	@echo "  make status           Show deployment state"
	@echo "  make logs             Follow manager logs"
	@echo "  make undeploy-dev     Remove development deployment"
	@echo "  make uninstall        Remove lifecycle CRDs"
	@echo ""

.PHONY: fmt
fmt:
	gofmt -w api cmd internal

.PHONY: vet
vet:
	go vet ./...

.PHONY: tidy
tidy:
	go mod tidy

.PHONY: generate
generate:
	$(CONTROLLER_GEN) object paths="./api/..."

.PHONY: manifests
manifests:
	mkdir -p config/crd/bases config/rbac
	$(CONTROLLER_GEN) \
		rbac:roleName=osha-lifecycle-manager-role \
		crd \
		paths="./api/...;./internal/..." \
		output:crd:artifacts:config=config/crd/bases \
		output:rbac:artifacts:config=config/rbac

.PHONY: test
test:
	go test ./... -cover

.PHONY: test-verbose
test-verbose:
	go test ./... -v

.PHONY: build
build: generate fmt vet
	mkdir -p $(BIN_DIR)
	CGO_ENABLED=0 go build \
		-trimpath \
		-o $(BIN_DIR)/$(BINARY) \
		./cmd/main.go

.PHONY: run
run: generate fmt vet
	go run ./cmd/main.go

.PHONY: docker-build
docker-build:
	$(CONTAINER_TOOL) build \
		--network=$(DOCKER_BUILD_NETWORK) \
		-t $(IMG) \
		.

.PHONY: docker-build-k8s
docker-build-k8s: docker-build
	$(CONTAINER_TOOL) save $(IMG) | \
		$(SUDO) $(CONTAINERD_TOOL) --namespace k8s.io images import -

.PHONY: docker-push
docker-push:
	$(CONTAINER_TOOL) push $(IMG)

.PHONY: install
install: manifests
	$(KUBECTL) apply -k config/crd

.PHONY: uninstall
uninstall:
	$(KUBECTL) delete \
		-k config/crd \
		--ignore-not-found=true

.PHONY: render-dev
render-dev:
	$(KUBECTL) kustomize \
		config/overlays/development

.PHONY: render-prod
render-prod:
	$(KUBECTL) kustomize \
		config/overlays/production

.PHONY: deploy-dev
deploy-dev: manifests
	$(KUBECTL) kustomize \
		config/overlays/development | \
	$(KUBECTL) apply -f -

.PHONY: deploy-prod
deploy-prod: manifests
	$(KUBECTL) kustomize \
		config/overlays/production | \
	$(KUBECTL) apply -f -

.PHONY: undeploy-dev
undeploy-dev:
	$(KUBECTL) kustomize \
		config/overlays/development | \
	$(KUBECTL) delete \
		-f - \
		--ignore-not-found=true

.PHONY: status
status:
	$(KUBECTL) get deployment,pods \
		-n $(NAMESPACE) \
		-o wide
	@echo
	$(KUBECTL) get operatorcatalogs \
		-n $(NAMESPACE) \
		2>/dev/null || true
	$(KUBECTL) get operatorpackages \
		-n $(NAMESPACE) \
		2>/dev/null || true
	$(KUBECTL) get operatorsubscriptions \
		-n $(NAMESPACE) \
		2>/dev/null || true
	$(KUBECTL) get operatorinstallplans \
		-n $(NAMESPACE) \
		2>/dev/null || true

.PHONY: samples
samples:
	$(KUBECTL) apply -k config/samples

.PHONY: logs
logs:
	$(KUBECTL) logs \
		-n $(NAMESPACE) \
		deployment/controller-manager \
		-c manager \
		-f

.PHONY: verify
verify:
	@echo "== Go =="
	go version
	@echo
	@echo "== kubectl =="
	$(KUBECTL) version --client
	@echo
	@echo "== Kubernetes API =="
	$(KUBECTL) cluster-info
	@echo
	@echo "== controller-gen =="
	$(CONTROLLER_GEN) --version
	@echo
	@echo "== Docker =="
	$(CONTAINER_TOOL) --version

.PHONY: clean
clean:
	rm -rf $(BIN_DIR)
	rm -f coverage.out

.PHONY: test-e2e
test-e2e:
	IMAGE=$(IMG) ./test/e2e/run.sh
