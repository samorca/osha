from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class KubernetesCollectorConfig:
    """
    Configuration for Kubernetes telemetry collection.

    Design principle:
      1. Automatically discover normal Kubernetes configuration.
      2. Allow explicit overrides when required.
      3. Do not require shell environment setup for ordinary use.
    """

    cluster_id: Optional[str] = None
    environment_id: str = "default"
    namespace: Optional[str] = None

    kubeconfig: Optional[str] = None
    context: Optional[str] = None

    connection_mode: str = "auto"

    tail_lines: int = 200

    output_path: str = (
        "orchestration-ai-data/live/"
        "kubernetes/telemetry.jsonl"
    )

    @classmethod
    def auto(
        cls,
        *,
        namespace: Optional[str] = None,
        cluster_id: Optional[str] = None,
        environment_id: str = "default",
        context: Optional[str] = None,
        kubeconfig: Optional[str] = None,
        tail_lines: int = 200,
        output_path: Optional[str] = None,
    ) -> "KubernetesCollectorConfig":

        discovered_kubeconfig = (
            kubeconfig
            or cls._discover_kubeconfig()
        )

        return cls(
            cluster_id=cluster_id,
            environment_id=environment_id,
            namespace=namespace,
            kubeconfig=discovered_kubeconfig,
            context=context,
            connection_mode="auto",
            tail_lines=tail_lines,
            output_path=(
                output_path
                or (
                    "orchestration-ai-data/live/"
                    "kubernetes/telemetry.jsonl"
                )
            ),
        )

    @staticmethod
    def _discover_kubeconfig() -> Optional[str]:
        """
        Resolve a kubeconfig location without requiring
        the user to export KUBECONFIG manually.
        """

        explicit = os.getenv("KUBECONFIG")

        if explicit:
            first_path = explicit.split(
                os.pathsep
            )[0]

            candidate = Path(
                first_path
            ).expanduser()

            if candidate.is_file():
                return str(
                    candidate
                )

        default_path = (
            Path.home()
            / ".kube"
            / "config"
        )

        if default_path.is_file():
            return str(
                default_path
            )

        return None
