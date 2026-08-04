"""Environment-agnostic process actors for independent rollout jobs.

The pool deliberately knows nothing about an environment, observation, action,
or learning algorithm.  A caller supplies a picklable worker function whose
job owns any mutable environment state it creates.  Results are returned in
submission order so seeded experiments remain straightforward to audit.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from typing import Any, Callable, Generic, Sequence, TypeVar


JobT = TypeVar("JobT")
ResultT = TypeVar("ResultT")


class ProcessActorPool(Generic[JobT, ResultT]):
    """Persistent spawn-based workers for independent actor jobs.

    ``spawn`` is the default because it avoids inheriting CUDA and environment
    runtime state from the learner.  The supplied initializer is run exactly
    once in each worker process; policy snapshots or other read-only bootstrap
    data can therefore be installed without putting domain semantics here.
    """

    def __init__(
        self,
        worker: Callable[[JobT], ResultT],
        *,
        workers: int,
        initializer: Callable[..., Any] | None = None,
        initargs: Sequence[Any] = (),
        start_method: str = "spawn",
    ) -> None:
        if int(workers) <= 0:
            raise ValueError("workers must be positive")
        if not str(start_method).strip():
            raise ValueError("start_method must not be empty")
        self.worker = worker
        self.workers = int(workers)
        self.start_method = str(start_method)
        self._closed = False
        context = multiprocessing.get_context(self.start_method)
        self._executor = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=context,
            initializer=initializer,
            initargs=tuple(initargs),
        )

    def map(self, jobs: Sequence[JobT]) -> tuple[ResultT, ...]:
        """Run jobs in workers and retain the caller's submission order."""

        if self._closed:
            raise RuntimeError("actor pool is closed")
        if not jobs:
            return ()
        return tuple(self._executor.map(self.worker, jobs, chunksize=1))

    def close(self) -> None:
        if not self._closed:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._closed = True

    def __enter__(self) -> "ProcessActorPool[JobT, ResultT]":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.close()
