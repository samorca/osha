from __future__ import annotations

import os

from kubernetes import client, config

from src.collectors.kubernetes.config import (
    KubernetesCollectorConfig,
)


def load_kubernetes_client(
    settings: KubernetesCollectorConfig,
) -> client.CoreV1Api:
    if settings.connection_mode not in {
        "auto",
        "in_cluster",
        "kubeconfig",
    }:
        raise ValueError(
            "connection_mode must be 'auto', "
            "'in_cluster', or 'kubeconfig'"
        )

    use_in_cluster_config = (
        settings.connection_mode == "in_cluster"
        or (
            settings.connection_mode == "auto"
            and settings.kubeconfig is None
            and bool(
                os.getenv(
                    "KUBERNETES_SERVICE_HOST"
                )
            )
        )
    )

    if use_in_cluster_config:
        config.load_incluster_config()
    else:
        kwargs = {}

        if settings.kubeconfig:
            kwargs["config_file"] = (
                settings.kubeconfig
            )

        if settings.context:
            kwargs["context"] = (
                settings.context
            )

        config.load_kube_config(
            **kwargs
        )

    return client.CoreV1Api()
