#!/bin/bash

set -euo pipefail

source_directory="${OSHA_MODEL_SEED_DIRECTORY:-/opt/osha/model-seed}"
target_directory="${OSHA_AI_MODELS_DIRECTORY:-/app/models}"

if [[ ! -d "$source_directory" ]]; then
    printf 'Model seed directory does not exist: %s\n' "$source_directory" >&2
    exit 1
fi

mapfile -d '' manifests < <(
    find "$source_directory" -type f -name artifact.json -print0
)

if (( ${#manifests[@]} == 0 )); then
    printf 'Model seed directory contains no files: %s\n' "$source_directory" >&2
    exit 1
fi

mkdir -p "$target_directory"

for manifest in "${manifests[@]}"; do
    source_model_directory="$(dirname "$manifest")"
    relative_directory="${source_model_directory#"$source_directory"/}"
    target_model_directory="$target_directory/$relative_directory"
    model_file="$source_model_directory/model.pt"

    if [[ ! -s "$model_file" ]]; then
        printf 'Seeded model is missing its model.pt file: %s\n' "$source_model_directory" >&2
        exit 1
    fi

    if [[ -e "$target_model_directory" ]]; then
        if [[ ! -s "$target_model_directory/model.pt" || ! -s "$target_model_directory/artifact.json" ]]; then
            printf 'Existing model directory is incomplete: %s\n' "$target_model_directory" >&2
            exit 1
        fi
        continue
    fi

    target_parent="$(dirname "$target_model_directory")"
    mkdir -p "$target_parent"
    staging_directory="$(mktemp -d "$target_parent/.seed-$(basename "$target_model_directory").XXXXXX")"
    cp -a -- "$source_model_directory/." "$staging_directory/"
    mv -T -n -- "$staging_directory" "$target_model_directory"

    if [[ -e "$staging_directory" ]]; then
        if [[ ! -s "$target_model_directory/model.pt" || ! -s "$target_model_directory/artifact.json" ]]; then
            printf 'Concurrent model seed produced an incomplete directory: %s\n' "$target_model_directory" >&2
            exit 1
        fi
        rm -rf -- "$staging_directory"
    fi
done
