"""Runtime identity for single-process experiments outside Flower's app runner."""

from contextlib import contextmanager


@contextmanager
def local_task_identity():
    """Restore caller state after a synchronous local experiment.

    Flower normally sets this process-wide identity in its app runner. This
    helper is only for local experiments and cannot isolate concurrent tasks.
    """
    from flwr.supercore.task_identity import TaskIdentity

    previous = (TaskIdentity._run_id, TaskIdentity._node_id, TaskIdentity._task_id)
    TaskIdentity.run_id = 1
    TaskIdentity.node_id = 0
    TaskIdentity.task_id = 1
    try:
        yield
    finally:
        TaskIdentity._run_id, TaskIdentity._node_id, TaskIdentity._task_id = previous
