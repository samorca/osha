{{/*
=====================================================================
OSHA Helm template helpers
=====================================================================

This file contains reusable naming, labeling, image,
and ServiceAccount helpers for the OSHA chart.

Workloads:

    API
    Runtime
    Worker

The helpers intentionally keep selector labels stable.
Labels used by Deployment selectors must not change
between upgrades.
*/}}


{{/*
---------------------------------------------------------------------
Chart name
---------------------------------------------------------------------
*/}}

{{- define "osha.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end }}


{{/*
---------------------------------------------------------------------
Fully qualified release name
---------------------------------------------------------------------
*/}}

{{- define "osha.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := include "osha.name" . -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end }}


{{/*
---------------------------------------------------------------------
Chart label
---------------------------------------------------------------------
*/}}

{{- define "osha.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end }}


{{/*
---------------------------------------------------------------------
Version
---------------------------------------------------------------------
*/}}

{{- define "osha.version" -}}
{{- default .Chart.AppVersion .Values.global.image.tag | trunc 63 | trimSuffix "-" -}}
{{- end }}


{{/*
---------------------------------------------------------------------
Base selector labels
---------------------------------------------------------------------

Do not put mutable values here.

These labels participate in workload selectors.
*/}}

{{- define "osha.selectorLabels" -}}
app.kubernetes.io/name: {{ include "osha.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}


{{/*
---------------------------------------------------------------------
Common resource labels
---------------------------------------------------------------------
*/}}

{{- define "osha.labels" -}}
helm.sh/chart: {{ include "osha.chart" . }}
{{ include "osha.selectorLabels" . }}
app.kubernetes.io/part-of: osha
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
osha.atonixcorp.io/system: orchestration-self-healing-ai
osha.atonixcorp.io/environment: {{ .Values.global.environment | quote }}
{{- with .Values.commonLabels }}
{{ toYaml . }}
{{- end }}
{{- end }}


{{/*
=====================================================================
API
=====================================================================
*/}}


{{/*
---------------------------------------------------------------------
API resource name
---------------------------------------------------------------------
*/}}

{{- define "osha.api.fullname" -}}
{{- printf "%s-api" (include "osha.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end }}


{{/*
---------------------------------------------------------------------
API selector labels
---------------------------------------------------------------------
*/}}

{{- define "osha.api.selectorLabels" -}}
{{ include "osha.selectorLabels" . }}
app.kubernetes.io/component: api
{{- end }}


{{/*
---------------------------------------------------------------------
API resource labels
---------------------------------------------------------------------
*/}}

{{- define "osha.api.labels" -}}
{{ include "osha.labels" . }}
app.kubernetes.io/component: api
{{- end }}


{{/*
---------------------------------------------------------------------
API ServiceAccount
---------------------------------------------------------------------
*/}}

{{- define "osha.api.serviceAccountName" -}}
{{- if .Values.serviceAccount.api.create -}}
{{- default (include "osha.api.fullname" .) .Values.serviceAccount.api.name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.api.name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end }}


{{/*
=====================================================================
Runtime
=====================================================================
*/}}


{{/*
---------------------------------------------------------------------
Runtime resource name
---------------------------------------------------------------------
*/}}

{{- define "osha.runtime.fullname" -}}
{{- printf "%s-runtime" (include "osha.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end }}


{{/*
---------------------------------------------------------------------
Runtime selector labels
---------------------------------------------------------------------
*/}}

{{- define "osha.runtime.selectorLabels" -}}
{{ include "osha.selectorLabels" . }}
app.kubernetes.io/component: runtime
{{- end }}


{{/*
---------------------------------------------------------------------
Runtime resource labels
---------------------------------------------------------------------
*/}}

{{- define "osha.runtime.labels" -}}
{{ include "osha.labels" . }}
app.kubernetes.io/component: runtime
{{- end }}


{{/*
---------------------------------------------------------------------
Runtime ServiceAccount
---------------------------------------------------------------------
*/}}

{{- define "osha.runtime.serviceAccountName" -}}
{{- if .Values.serviceAccount.runtime.create -}}
{{- default (include "osha.runtime.fullname" .) .Values.serviceAccount.runtime.name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.runtime.name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end }}


{{/*
=====================================================================
Worker
=====================================================================
*/}}


{{/*
---------------------------------------------------------------------
Worker resource name
---------------------------------------------------------------------
*/}}

{{- define "osha.worker.fullname" -}}
{{- printf "%s-worker" (include "osha.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end }}


{{/*
---------------------------------------------------------------------
Worker selector labels
---------------------------------------------------------------------
*/}}

{{- define "osha.worker.selectorLabels" -}}
{{ include "osha.selectorLabels" . }}
app.kubernetes.io/component: worker
osha.atonixcorp.io/worker-type: {{ .Values.worker.type | quote }}
{{- end }}


{{/*
---------------------------------------------------------------------
Worker resource labels
---------------------------------------------------------------------
*/}}

{{- define "osha.worker.labels" -}}
{{ include "osha.labels" . }}
app.kubernetes.io/component: worker
osha.atonixcorp.io/worker-type: {{ .Values.worker.type | quote }}
{{- end }}


{{/*
---------------------------------------------------------------------
Worker ServiceAccount
---------------------------------------------------------------------
*/}}

{{- define "osha.worker.serviceAccountName" -}}
{{- if .Values.serviceAccount.worker.create -}}
{{- default (include "osha.worker.fullname" .) .Values.serviceAccount.worker.name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.worker.name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end }}


{{/*
=====================================================================
Container image
=====================================================================
*/}}


{{/*
---------------------------------------------------------------------
OSHA image
---------------------------------------------------------------------

Result:

    repository:tag

Example:

    osha:dev
*/}}

{{- define "osha.image" -}}
{{- $repository := required "global.image.repository is required" .Values.global.image.repository -}}
{{- $tag := default .Chart.AppVersion .Values.global.image.tag -}}
{{- printf "%s:%s" $repository $tag -}}
{{- end }}


{{/*
=====================================================================
Configuration
=====================================================================
*/}}


{{/*
---------------------------------------------------------------------
ConfigMap name
---------------------------------------------------------------------
*/}}

{{- define "osha.configMapName" -}}
{{- printf "%s-config" (include "osha.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end }}


{{/*
=====================================================================
Namespace
=====================================================================
*/}}

{{- define "osha.namespace" -}}
{{- .Release.Namespace -}}
{{- end }}


{{/*
=====================================================================
Storage names
=====================================================================
*/}}


{{/*
---------------------------------------------------------------------
PersistentVolumeClaim name
---------------------------------------------------------------------
*/}}

{{- define "osha.persistence.claimName" -}}
{{- printf "%s-data" (include "osha.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end }}


{{/*
=====================================================================
Validation helpers
=====================================================================
*/}}


{{/*
---------------------------------------------------------------------
Validate worker type
---------------------------------------------------------------------
*/}}

{{- define "osha.worker.validate" -}}
{{- if not .Values.worker.type -}}
{{- fail "worker.type must not be empty" -}}
{{- end -}}
{{- end }}


{{/*
---------------------------------------------------------------------
Validate API port
---------------------------------------------------------------------
*/}}

{{- define "osha.api.validate" -}}
{{- if or (lt (int .Values.api.port) 1) (gt (int .Values.api.port) 65535) -}}
{{- fail "api.port must be between 1 and 65535" -}}
{{- end -}}
{{- end }}


{{/*
---------------------------------------------------------------------
Render optional pod annotations
---------------------------------------------------------------------
*/}}

{{- define "osha.podAnnotations" -}}
{{- with .Values.podAnnotations }}
{{ toYaml . }}
{{- end }}
{{- end }}
