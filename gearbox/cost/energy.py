"""GPU energy measurement via NVML's cumulative energy counter (Volta and newer,
e.g. RTX A5000). Measures the whole device, so use it around whole episodes:
when calls overlap on one GPU, per-call attribution is not well defined.

Requires the optional `gpu` extra (nvidia-ml-py). Without it, or without an
NVIDIA GPU, spans report joules=None.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class EnergySpan:
    joules: float | None = None
    wall_s: float = 0.0


class EnergyMeter:
    def __init__(self, device_index: int = 0) -> None:
        self._nvml = None
        self._handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)  # probe support
            self._nvml, self._handle = pynvml, handle
        except Exception:
            pass

    @property
    def available(self) -> bool:
        return self._handle is not None

    def read_mj(self) -> int | None:
        if self._handle is None:
            return None
        return self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)

    @contextmanager
    def span(self) -> Iterator[EnergySpan]:
        span = EnergySpan()
        start_mj = self.read_mj()
        start = time.perf_counter()
        try:
            yield span
        finally:
            span.wall_s = time.perf_counter() - start
            end_mj = self.read_mj()
            if start_mj is not None and end_mj is not None:
                span.joules = (end_mj - start_mj) / 1000.0
