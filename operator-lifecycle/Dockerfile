# syntax=docker/dockerfile:1

FROM golang:1.27.3 AS builder

ARG TARGETOS
ARG TARGETARCH

WORKDIR /workspace

COPY go.mod go.sum ./

RUN go mod download

COPY api/ api/
COPY cmd/ cmd/
COPY internal/ internal/

RUN CGO_ENABLED=0 \
    GOOS=${TARGETOS:-linux} \
    GOARCH=${TARGETARCH} \
    go build \
    -trimpath \
    -ldflags="-s -w" \
    -o /manager \
    ./cmd/main.go

FROM gcr.io/distroless/static:nonroot

WORKDIR /

COPY --from=builder /manager /manager

USER 65532:65532

ENTRYPOINT ["/manager"]
