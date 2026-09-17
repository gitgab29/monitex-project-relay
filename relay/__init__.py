"""Project Relay -- camera feed to structured security events to routed actions.

Thread caps are set here, before anything imports numpy or torch, and that ordering is the
whole point of doing it in the package __init__.

OpenBLAS reserves per-thread scratch buffers at load time, sized off the core count. On a
20-core machine with a few GB free that allocation simply fails:

    OpenBLAS error: Memory allocation still failed after 10 retries, giving up.

The workload here is one 640px YOLO inference per second, which is latency-trivial and gains
nothing from 20 threads, so capping costs no measurable throughput and makes the program run
on a normal, busy laptop. Anything already set in the environment is left alone, so a user
who wants the threads back just exports the variable.
"""

from __future__ import annotations

import os as _os

__version__ = "0.1.0"

#: Override with RELAY_NUM_THREADS, or by setting any of the underlying variables yourself.
_DEFAULT_THREADS = _os.environ.get("RELAY_NUM_THREADS", "2")

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_var, _DEFAULT_THREADS)


def torch_threads() -> int:
    """Apply the same cap to torch's own intra-op pool. Called by the CLI at startup."""
    n = max(1, int(_os.environ.get("RELAY_NUM_THREADS", _DEFAULT_THREADS)))
    try:
        import torch

        torch.set_num_threads(n)
    except Exception:  # torch absent or already configured -- not worth failing a run over
        pass
    return n
