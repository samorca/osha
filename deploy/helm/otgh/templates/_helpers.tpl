{{/*
==============================================================================
OTGH Helm template helpers
==============================================================================

This file contains reusable naming and labelling helpers for the OTGH chart.

Template names are prefixed with "otgh." to avoid collisions with helpers
defined by parent charts or subcharts.
*/}}


{{/*
------------------------------------------------------------------------------
Chart name
------------------------------------------------------------------------------

Returns the base application name.

Priority:

  .Values.nameOverride
      ↓
  .Chart.Name
*/}}
{{- define "otgh.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}


{{/*
------------------------------------------------------------------------------
Fully qualified application name
------------------------------------------------------------------------------

Priority:

  .Values.fullnameOverride
      ↓
  Release.Name already contains chart name
      ↓
  Release.Name + chart name

The final Kubernetes resource name is limited to 63 characters.
*/}}
{{- define "otgh.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := include "otgh.name" . -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}


{{/*
------------------------------------------------------------------------------
Chart identifier
------------------------------------------------------------------------------

Example:

  otgh-0.1.0

Used by:

  helm.sh/chart
*/}}
{{- define "otgh.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}


{{/*
------------------------------------------------------------------------------
Namespace
------------------------------------------------------------------------------

When namespace.name is explicitly configured, use it.

Otherwise use the Helm release namespace.
*/}}
{{- define "otgh.namespace" -}}
{{- if .Values.namespace.name -}}
{{- .Values.namespace.name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- .Release.Namespace | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}


{{/*
------------------------------------------------------------------------------
Selector labels
------------------------------------------------------------------------------

These labels must remain stable because Deployment selectors depend on them.

Do not include chart version, application version, or arbitrary pod labels in
this helper.
*/}}
{{- define "otgh.selectorLabels" -}}
app.kubernetes.io/name: {{ include "otgh.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: transporter
{{- end -}}


{{/*
------------------------------------------------------------------------------
Common labels
------------------------------------------------------------------------------

Used across OTGH Kubernetes resources.
*/}}
{{- define "otgh.labels" -}}
helm.sh/chart: {{ include "otgh.chart" . }}
{{ include "otgh.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/part-of: osha
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}


{{/*
------------------------------------------------------------------------------
Pod labels
------------------------------------------------------------------------------

Starts with common application labels and allows additional pod labels from
values.yaml.

User labels are rendered after standard labels so environment-specific labels
can be added without duplicating the core set in each deployment template.
*/}}
{{- define "otgh.podLabels" -}}
{{ include "otgh.labels" . }}
{{- with .Values.podLabels }}
{{ toYaml . }}
{{- end }}
{{- end -}}


{{/*
------------------------------------------------------------------------------
ServiceAccount name
------------------------------------------------------------------------------

When Helm manages the ServiceAccount:

  explicit serviceAccount.name
      ↓
  generated fullname

When Helm does not manage it:

  explicit serviceAccount.name
      ↓
  default
*/}}
{{- define "otgh.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "otgh.fullname" .) .Values.serviceAccount.name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}


{{/*
------------------------------------------------------------------------------
ConfigMap name
------------------------------------------------------------------------------ */}}
{{- define "otgh.configMapName" -}}
{{- printf "%s-config" (include "otgh.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}


{{/*
------------------------------------------------------------------------------
Secret name
------------------------------------------------------------------------------

Priority:

  secrets.existingSecret
      ↓
  generated <fullname>-secrets
*/}}
{{- define "otgh.secretName" -}}
{{- if .Values.secrets.existingSecret -}}
{{- .Values.secrets.existingSecret | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-secrets" (include "otgh.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}


{{/*
------------------------------------------------------------------------------
PersistentVolumeClaim name
------------------------------------------------------------------------------

Priority:

  persistence.existingClaim
      ↓
  generated <fullname>-spool
*/}}
{{- define "otgh.pvcName" -}}
{{- if .Values.persistence.existingClaim -}}
{{- .Values.persistence.existingClaim | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-spool" (include "otgh.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}


{{/*
------------------------------------------------------------------------------
Container image
------------------------------------------------------------------------------

Produces:

  repository:tag

Tag priority:

  image.tag
      ↓
  Chart.appVersion
*/}}
{{- define "otgh.image" -}}
{{- $tag := default .Chart.AppVersion .Values.image.tag -}}
{{- printf "%s:%s" .Values.image.repository $tag -}}
{{- end -}}


{{/*
------------------------------------------------------------------------------
Secret resource creation

The chart should create its own Kubernetes Secret only when:

  secrets.create == true

AND

  secrets.existingSecret is empty.

If an existing secret is configured, that external resource remains the source
of credentials.
*/}}
{{- define "otgh.createSecret" -}}
{{- if and .Values.secrets.create (not .Values.secrets.existingSecret) -}}
true
{{- else -}}
false
{{- end -}}
{{- end -}}


{{/*
------------------------------------------------------------------------------
PersistentVolumeClaim resource creation

Create a claim only when persistence is enabled and no existing claim has been
configured.
*/}}
{{- define "otgh.createPVC" -}}
{{- if and .Values.persistence.enabled (not .Values.persistence.existingClaim) -}}
true
{{- else -}}
false
{{- end -}}
{{- end -}}


{{/*
------------------------------------------------------------------------------
OTGH spool mount path
------------------------------------------------------------------------------ */}}
{{- define "otgh.spoolPath" -}}
{{- .Values.config.spool.directory -}}
{{- end -}}


{{/*
------------------------------------------------------------------------------
OTGH configuration file path
------------------------------------------------------------------------------ */}}
{{- define "otgh.configPath" -}}
/etc/otgh/otgh.json
{{- end -}}


{{/*
------------------------------------------------------------------------------
OTGH temporary filesystem path
------------------------------------------------------------------------------ */}}
{{- define "otgh.tmpPath" -}}
/tmp
{{- end -}}
