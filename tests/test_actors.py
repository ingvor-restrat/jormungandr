from __future__ import annotations

import os

import pytest

from jormungandr.actors import ProcessActorPool


_ACTOR_OFFSET = 0


def _initialize_actor(offset: int) -> None:
    global _ACTOR_OFFSET
    _ACTOR_OFFSET = int(offset)


def _actor_job(value: int) -> tuple[int, int, int]:
    return value, _ACTOR_OFFSET + value, os.getpid()


def test_process_actor_pool_runs_initialized_jobs_in_submission_order() -> None:
    parent_pid = os.getpid()
    with ProcessActorPool(
        _actor_job,
        workers=2,
        initializer=_initialize_actor,
        initargs=(100,),
    ) as pool:
        results = pool.map(tuple(range(8)))

    assert [item[0] for item in results] == list(range(8))
    assert [item[1] for item in results] == list(range(100, 108))
    assert all(item[2] != parent_pid for item in results)
    assert 1 <= len({item[2] for item in results}) <= 2


def test_process_actor_pool_rejects_invalid_lifecycle() -> None:
    with pytest.raises(ValueError, match="workers"):
        ProcessActorPool(_actor_job, workers=0)

    pool = ProcessActorPool(_actor_job, workers=1)
    pool.close()
    with pytest.raises(RuntimeError, match="closed"):
        pool.map((1,))
