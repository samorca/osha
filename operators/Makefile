SHELL := /usr/bin/env bash

BINARY ?= manager
BIN_DIR ?= bin

IMG ?= osha-operator:dev
CONTAINER_TOOL ?= docker
CONTAINERD_TOOL ?= ctr
SUDO ?= sudo

NAMESPACE ?= osha-system

CONTROLLER_GEN ?= controller-gen
KUSTOMIZE ?= kubectl kustomize
KUBECTL ?= kubectl

.PHONY: all
all: build

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
		rbac:roleName=osha-operator-role \
		crd \
		paths="./api/...;./internal/controller/..." \
		output:crd:artifacts:config=config/crd/bases \
		output:rbac:artifacts:config=config/rbac

.PHONY: test
test:
	go test ./... -cover

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
docker-build: build
	$(CONTAINER_TOOL) build \
		-t $(IMG) \
		.

.PHONY: docker-push
docker-push:
	$(CONTAINER_TOOL) push $(IMG)

.PHONY: docker-build-k8s
docker-build-k8s: docker-build
	$(CONTAINER_TOOL) save $(IMG) | \
		$(SUDO) $(CONTAINERD_TOOL) --namespace k8s.io images import -

.PHONY: install
install: manifests
	$(KUBECTL) apply -k config/crd

.PHONY: uninstall
uninstall:
	$(KUBECTL) delete -k config/crd \
		--ignore-not-found=true

.PHONY: render-dev
render-dev:
	$(KUSTOMIZE) config/overlays/development

.PHONY: render-production
render-production:
	$(KUSTOMIZE) config/overlays/production

.PHONY: deploy-dev
deploy-dev: manifests
	$(KUSTOMIZE) config/overlays/development | \
	$(KUBECTL) apply -f -

.PHONY: deploy-production
deploy-production: manifests
	$(KUSTOMIZE) config/overlays/production | \
	$(KUBECTL) apply -f -

.PHONY: undeploy-dev
undeploy-dev:
	$(KUSTOMIZE) config/overlays/development | \
	$(KUBECTL) delete -f - \
	--ignore-not-found=true

.PHONY: status
status:
	$(KUBECTL) get deployment,pods \
		-n $(NAMESPACE) \
		-o wide
	$(KUBECTL) get oshaplatforms \
		-n $(NAMESPACE)

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
	@echo "== Kubernetes cluster =="
	$(KUBECTL) cluster-info
	@echo
	@echo "== controller-gen =="
	$(CONTROLLER_GEN) --version
	@echo
	@echo "== Container runtime =="
	$(CONTAINER_TOOL) --version
	@echo
	@echo "== Kubernetes containerd CLI =="
	$(CONTAINERD_TOOL) --version

.PHONY: clean
clean:
	rm -rf $(BIN_DIR)
	rm -f coverage.out

.PHONY: help
help:
	@echo ""
	@echo "OSHA Operator"
	@echo ""
	@echo "Development:"
	@echo "  make fmt                Format Go source"
	@echo "  make tidy               Resolve Go modules"
	@echo "  make generate           Generate DeepCopy code"
	@echo "  make manifests          Generate CRD and RBAC"
	@echo "  make test               Run tests"
	@echo "  make build              Build manager binary"
	@echo "  make run                Run against current kubeconfig"
	@echo ""
	@echo "Container:"
	@echo "  make docker-build       Build container image"
	@echo "  make docker-build-k8s   Build image and import it into k8s.io containerd"
	@echo "  make docker-push        Push container image"
	@echo ""
	@echo "Kubernetes:"
	@echo "  make install            Install CRDs"
	@echo "  make deploy-dev         Deploy development Operator"
	@echo "  make deploy-production  Deploy production Operator"
	@echo "  make status             Show Operator status"
	@echo "  make logs               Follow Operator logs"
	@echo "  make undeploy-dev       Remove development Operator"
	@echo "  make uninstall          Remove CRDs"
	@echo ""
