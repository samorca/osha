from __future__ import annotations

"""
Runtime model registry for OSHA AI serving.

The registry tracks:

    discovered/registered model descriptors
    instantiated model adapters
    currently loaded models
    model access ordering
    loaded-model cache limits
    descriptor replacement policy

The registry does not discover model files.
Discovery belongs to loader.py.
"""

import time

from collections import (
    OrderedDict,
)

from threading import (
    RLock,
)

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)

from src.ai.base import (
    AIModelAdapter,
)

from src.ai.config import (
    AIConfig,
)

from src.ai.errors import (
    AIModelAlreadyRegisteredError,
    AIModelCacheLimitError,
    AIModelNotRegisteredError,
    AIModelReplacementError,
    AIModelStateError,
)

from src.ai.models import (
    create_model_adapter,
)

from src.ai.schema import (
    AIModelDescriptor,
    AIModelState,
)


# ---------------------------------------------------------------
# Registry entry
# ---------------------------------------------------------------


class AIModelRegistryEntry:
    """
    One descriptor plus optional runtime adapter.
    """

    def __init__(
        self,
        descriptor: AIModelDescriptor,
        *,
        adapter: Optional[AIModelAdapter] = None,
    ) -> None:
        descriptor.finalize()

        self.descriptor = descriptor

        self.adapter = adapter

        now = time.monotonic()

        self.registered_monotonic = now

        self.last_access_monotonic = now

        self.access_count = 0

    # ---------------------------------------------------------
    # Identity
    # ---------------------------------------------------------

    @property
    def model_id(
        self,
    ) -> str:
        return self.descriptor.identity.model_id

    # ---------------------------------------------------------
    # Runtime status
    # ---------------------------------------------------------

    @property
    def loaded(
        self,
    ) -> bool:
        return bool(self.adapter is not None and self.adapter.loaded)

    @property
    def ready(
        self,
    ) -> bool:
        return bool(self.adapter is not None and self.adapter.ready)

    # ---------------------------------------------------------
    # Access tracking
    # ---------------------------------------------------------

    def touch(
        self,
    ) -> None:
        self.last_access_monotonic = time.monotonic()

        self.access_count += 1

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "model_id": (self.model_id),
            "descriptor": (self.descriptor.to_dict()),
            "adapter_created": (self.adapter is not None),
            "loaded": (self.loaded),
            "ready": (self.ready),
            "access_count": (self.access_count),
        }


# ---------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------


class AIModelRegistry:
    """
    Thread-safe runtime registry.

    Descriptors are cheap and may remain
    registered without being loaded.

    Runtime adapters are created lazily or eagerly
    depending on the caller/configuration.
    """

    def __init__(
        self,
        *,
        config: Optional[AIConfig] = None,
    ) -> None:
        self.config = config or AIConfig()

        self.config.validate()

        self._entries: Dict[
            str,
            AIModelRegistryEntry,
        ] = {}

        # Ordered from least recently used to most
        # recently used.
        self._loaded_order: OrderedDict[
            str,
            None,
        ] = OrderedDict()

        self._lock = RLock()

    # ---------------------------------------------------------
    # Model ID
    # ---------------------------------------------------------

    @staticmethod
    def _model_id(
        descriptor: AIModelDescriptor,
    ) -> str:
        descriptor.finalize()

        return descriptor.identity.model_id

    # ---------------------------------------------------------
    # Register descriptor
    # ---------------------------------------------------------

    def register(
        self,
        descriptor: AIModelDescriptor,
        *,
        replace: bool = False,
        create_adapter: bool = False,
        load: bool = False,
    ) -> AIModelRegistryEntry:
        """
        Register one model descriptor.

        Replacement requires BOTH:

            replace=True
            config.loading.allow_runtime_replacement=True

        unless the existing and incoming objects
        are literally the same descriptor object.
        """

        descriptor.finalize()

        if not self.config.family_allowed(descriptor.identity.family):
            raise AIModelReplacementError(
                "Model family is not enabled by AI configuration.",
                details={
                    "family": (descriptor.identity.family),
                    "model_id": (descriptor.identity.model_id),
                },
            )

        model_id = self._model_id(descriptor)

        with self._lock:
            existing = self._entries.get(model_id)

            if existing is not None:
                if not replace:
                    raise (
                        AIModelAlreadyRegisteredError(
                            "AI model is already registered.",
                            details={
                                "model_id": (model_id),
                            },
                        )
                    )

                if not (self.config.loading.allow_runtime_replacement):
                    raise AIModelReplacementError(
                        "Runtime model replacement is disabled.",
                        details={
                            "model_id": (model_id),
                        },
                    )

                if existing.adapter is not None:
                    self._unload_entry_locked(existing)

                self._entries.pop(
                    model_id,
                    None,
                )

            entry = AIModelRegistryEntry(descriptor)

            self._entries[model_id] = entry

        if create_adapter or load or not self.config.loading.lazy_loading:
            self.ensure_adapter(model_id)

        if load or not self.config.loading.lazy_loading:
            self.load(model_id)

        return self.require(model_id)

    # ---------------------------------------------------------
    # Register many
    # ---------------------------------------------------------

    def register_many(
        self,
        descriptors: Iterable[AIModelDescriptor],
        *,
        replace: bool = False,
        create_adapters: bool = False,
        load: bool = False,
    ) -> List[AIModelRegistryEntry]:
        output: List[AIModelRegistryEntry] = []

        for descriptor in descriptors:
            output.append(
                self.register(
                    descriptor,
                    replace=replace,
                    create_adapter=(create_adapters),
                    load=load,
                )
            )

        return output

    # ---------------------------------------------------------
    # Get
    # ---------------------------------------------------------

    def get(
        self,
        model_id: str,
    ) -> Optional[AIModelRegistryEntry]:
        with self._lock:
            entry = self._entries.get(str(model_id))

            if entry is not None:
                entry.touch()

                if entry.loaded:
                    self._touch_loaded_locked(entry.model_id)

            return entry

    # ---------------------------------------------------------
    # Peek
    # ---------------------------------------------------------

    def peek(
        self,
        model_id: str,
    ) -> Optional[AIModelRegistryEntry]:
        """
        Lookup without affecting LRU/access
        accounting.
        """

        with self._lock:
            return self._entries.get(str(model_id))

    # ---------------------------------------------------------
    # Require
    # ---------------------------------------------------------

    def require(
        self,
        model_id: str,
    ) -> AIModelRegistryEntry:
        entry = self.get(model_id)

        if entry is None:
            raise AIModelNotRegisteredError(
                "AI model is not registered.",
                details={
                    "model_id": (model_id),
                },
            )

        return entry

    # ---------------------------------------------------------
    # Find matching descriptors
    # ---------------------------------------------------------

    def find(
        self,
        *,
        family: Optional[str] = None,
        model_name: Optional[str] = None,
        version: Optional[str] = None,
        state: Optional[str] = None,
        loaded: Optional[bool] = None,
    ) -> List[AIModelRegistryEntry]:
        with self._lock:
            entries = list(self._entries.values())

        output: List[AIModelRegistryEntry] = []

        for entry in entries:
            identity = entry.descriptor.identity

            if family is not None and identity.family != family:
                continue

            if model_name is not None and identity.model_name != model_name:
                continue

            if version is not None and identity.version != version:
                continue

            if state is not None and entry.descriptor.state != state:
                continue

            if loaded is not None and entry.loaded is not loaded:
                continue

            output.append(entry)

        return sorted(
            output,
            key=lambda entry: (
                entry.descriptor.identity.family,
                entry.descriptor.identity.model_name,
                entry.descriptor.identity.version,
            ),
        )

    # ---------------------------------------------------------
    # Descriptors
    # ---------------------------------------------------------

    def descriptors(
        self,
        *,
        family: Optional[str] = None,
    ) -> List[AIModelDescriptor]:
        return [entry.descriptor for entry in self.find(family=family)]

    # ---------------------------------------------------------
    # Adapter construction
    # ---------------------------------------------------------

    def ensure_adapter(
        self,
        model_id: str,
    ) -> AIModelAdapter:
        with self._lock:
            entry = self._entries.get(str(model_id))

            if entry is None:
                raise (
                    AIModelNotRegisteredError(
                        "AI model is not registered.",
                        details={
                            "model_id": (model_id),
                        },
                    )
                )

            if entry.adapter is not None:
                entry.touch()

                return entry.adapter

            adapter = create_model_adapter(
                entry.descriptor,
                config=self.config,
            )

            if adapter.model_id != entry.model_id:
                raise AIModelStateError(
                    "Created model adapter identity does not match the registry entry."
                )

            entry.adapter = adapter

            entry.touch()

            return adapter

    # ---------------------------------------------------------
    # Loaded count
    # ---------------------------------------------------------

    def loaded_count(
        self,
    ) -> int:
        with self._lock:
            return sum(1 for entry in self._entries.values() if entry.loaded)

    # ---------------------------------------------------------
    # LRU accounting
    # ---------------------------------------------------------

    def _touch_loaded_locked(
        self,
        model_id: str,
    ) -> None:
        self._loaded_order.pop(
            model_id,
            None,
        )

        self._loaded_order[model_id] = None

    # ---------------------------------------------------------
    # Select eviction candidate
    # ---------------------------------------------------------

    def _eviction_candidate_locked(
        self,
        *,
        exclude: Optional[str] = None,
    ) -> Optional[AIModelRegistryEntry]:
        for model_id in list(self._loaded_order.keys()):
            if exclude is not None and model_id == exclude:
                continue

            entry = self._entries.get(model_id)

            if entry is not None and entry.loaded:
                return entry

        # Recover gracefully if loaded-order
        # tracking was empty/stale.
        candidates = [
            entry
            for entry in self._entries.values()
            if (entry.loaded and entry.model_id != exclude)
        ]

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda entry: entry.last_access_monotonic,
        )

    # ---------------------------------------------------------
    # Ensure capacity
    # ---------------------------------------------------------

    def _ensure_capacity_locked(
        self,
        *,
        loading_model_id: str,
    ) -> None:
        limit = self.config.loading.maximum_loaded_models

        while sum(1 for entry in self._entries.values() if entry.loaded) >= limit:
            if not (self.config.loading.unload_unused_models):
                raise AIModelCacheLimitError(
                    "Maximum loaded-model limit has been reached.",
                    details={
                        "limit": limit,
                        "loading_model_id": (loading_model_id),
                    },
                )

            candidate = self._eviction_candidate_locked(exclude=(loading_model_id))

            if candidate is None:
                raise AIModelCacheLimitError(
                    "No loaded model can "
                    "be evicted to make "
                    "room for the requested "
                    "model.",
                    details={
                        "limit": limit,
                    },
                )

            self._unload_entry_locked(candidate)

    # ---------------------------------------------------------
    # Load
    # ---------------------------------------------------------

    def load(
        self,
        model_id: str,
    ) -> AIModelAdapter:
        """
        Ensure the requested model has a ready
        runtime adapter.

        If the loaded-model cache is full and
        automatic unloading is enabled, the least
        recently used model is unloaded first.
        """

        model_id = str(model_id)

        with self._lock:
            entry = self._entries.get(model_id)

            if entry is None:
                raise (
                    AIModelNotRegisteredError(
                        "AI model is not registered.",
                        details={
                            "model_id": (model_id),
                        },
                    )
                )

            if entry.adapter is not None and entry.adapter.ready:
                entry.touch()

                self._touch_loaded_locked(model_id)

                return entry.adapter

            self._ensure_capacity_locked(loading_model_id=(model_id))

            if entry.adapter is None:
                entry.adapter = create_model_adapter(
                    entry.descriptor,
                    config=self.config,
                )

            adapter = entry.adapter

        # Actual model loading may involve disk
        # and GPU operations, so avoid holding the
        # registry lock during adapter.load().
        try:
            adapter.load()

        except Exception:
            with self._lock:
                self._loaded_order.pop(
                    model_id,
                    None,
                )

            raise

        with self._lock:
            current = self._entries.get(model_id)

            if current is None:
                # Defensive guard against removal
                # during an unusual external
                # registry mutation.
                try:
                    adapter.unload()
                finally:
                    pass

                raise AIModelNotRegisteredError(
                    "AI model was removed while loading.",
                    details={
                        "model_id": (model_id),
                    },
                )

            current.adapter = adapter
            current.touch()

            self._touch_loaded_locked(model_id)

            return adapter

    # ---------------------------------------------------------
    # Unload entry
    # ---------------------------------------------------------

    def _unload_entry_locked(
        self,
        entry: AIModelRegistryEntry,
    ) -> None:
        adapter = entry.adapter

        self._loaded_order.pop(
            entry.model_id,
            None,
        )

        if adapter is None:
            entry.descriptor.state = AIModelState.UNLOADED.value

            return

        adapter.unload()

    # ---------------------------------------------------------
    # Unload
    # ---------------------------------------------------------

    def unload(
        self,
        model_id: str,
        *,
        drop_adapter: bool = False,
    ) -> None:
        with self._lock:
            entry = self._entries.get(str(model_id))

            if entry is None:
                raise (
                    AIModelNotRegisteredError(
                        "AI model is not registered.",
                        details={
                            "model_id": (model_id),
                        },
                    )
                )

            self._unload_entry_locked(entry)

            if drop_adapter:
                entry.adapter = None

    # ---------------------------------------------------------
    # Unload all
    # ---------------------------------------------------------

    def unload_all(
        self,
        *,
        drop_adapters: bool = False,
    ) -> List[str]:
        with self._lock:
            entries = list(self._entries.values())

        unloaded: List[str] = []

        for entry in entries:
            if entry.adapter is None and not entry.loaded:
                continue

            self.unload(
                entry.model_id,
                drop_adapter=(drop_adapters),
            )

            unloaded.append(entry.model_id)

        return unloaded

    # ---------------------------------------------------------
    # Remove
    # ---------------------------------------------------------

    def remove(
        self,
        model_id: str,
        *,
        unload: bool = True,
    ) -> AIModelDescriptor:
        model_id = str(model_id)

        with self._lock:
            entry = self._entries.get(model_id)

            if entry is None:
                raise (
                    AIModelNotRegisteredError(
                        "AI model is not registered.",
                        details={
                            "model_id": (model_id),
                        },
                    )
                )

            if entry.loaded and not unload:
                raise AIModelStateError(
                    "Loaded model cannot be removed without unloading it first.",
                    details={
                        "model_id": (model_id),
                    },
                )

            if unload:
                self._unload_entry_locked(entry)

            self._entries.pop(
                model_id,
                None,
            )

            self._loaded_order.pop(
                model_id,
                None,
            )

            return entry.descriptor

    # ---------------------------------------------------------
    # Access adapter
    # ---------------------------------------------------------

    def adapter(
        self,
        model_id: str,
        *,
        load: bool = True,
    ) -> AIModelAdapter:
        """
        Return a model adapter, optionally loading
        the runtime model if necessary.
        """

        if load:
            return self.load(model_id)

        return self.ensure_adapter(model_id)

    # ---------------------------------------------------------
    # Ready models
    # ---------------------------------------------------------

    def ready_models(
        self,
        *,
        family: Optional[str] = None,
    ) -> List[AIModelRegistryEntry]:
        entries = self.find(family=family)

        return [entry for entry in entries if entry.ready]

    # ---------------------------------------------------------
    # Latest version helper
    # ---------------------------------------------------------

    def latest(
        self,
        *,
        family: str,
        model_name: Optional[str] = None,
    ) -> Optional[AIModelRegistryEntry]:
        """
        Return the lexicographically latest
        registered version.

        This intentionally does not try to infer
        semantic-version meaning. routing.py may
        later provide a stricter version policy.
        """

        entries = self.find(
            family=family,
            model_name=(model_name),
        )

        if not entries:
            return None

        return max(
            entries,
            key=lambda entry: entry.descriptor.identity.version,
        )

    # ---------------------------------------------------------
    # Registry size
    # ---------------------------------------------------------

    def __len__(
        self,
    ) -> int:
        with self._lock:
            return len(self._entries)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        with self._lock:
            entries = list(self._entries.values())

            loaded_order = list(self._loaded_order.keys())

        by_family: Dict[
            str,
            int,
        ] = {}

        by_state: Dict[
            str,
            int,
        ] = {}

        loaded_count = 0
        ready_count = 0

        for entry in entries:
            family = entry.descriptor.identity.family

            state = entry.descriptor.state

            by_family[family] = (
                by_family.get(
                    family,
                    0,
                )
                + 1
            )

            by_state[state] = (
                by_state.get(
                    state,
                    0,
                )
                + 1
            )

            if entry.loaded:
                loaded_count += 1

            if entry.ready:
                ready_count += 1

        return {
            "registered_models": len(entries),
            "loaded_models": (loaded_count),
            "ready_models": (ready_count),
            "maximum_loaded_models": (self.config.loading.maximum_loaded_models),
            "lazy_loading": (self.config.loading.lazy_loading),
            "automatic_unload": (self.config.loading.unload_unused_models),
            "runtime_replacement": (self.config.loading.allow_runtime_replacement),
            "by_family": (by_family),
            "by_state": (by_state),
            "loaded_lru_order": (loaded_order),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "AIModelRegistryEntry",
    "AIModelRegistry",
]
