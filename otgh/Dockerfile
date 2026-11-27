# syntax=docker/dockerfile:1

ARG GO_VERSION=1.27.1

FROM golang:${GO_VERSION}-bookworm AS build

WORKDIR /src

COPY go.mod go.sum ./

RUN go mod download

COPY . .

ARG VERSION=dev
ARG COMMIT=unknown
ARG BUILD_DATE=unknown

RUN CGO_ENABLED=0 \
    GOOS=linux \
    go build \
    -trimpath \
    -ldflags="-s -w" \
    -o /out/otgh \
    ./cmd/otgh


FROM gcr.io/distroless/static-debian12:nonroot AS runtime

WORKDIR /app

COPY --from=build /out/otgh /usr/local/bin/otgh

USER nonroot:nonroot

VOLUME ["/var/lib/otgh"]

ENV OTGH_CONFIG=/etc/otgh/otgh.json
ENV OTGH_SPOOL_DIR=/var/lib/otgh/spool

ENTRYPOINT ["/usr/local/bin/otgh"]
