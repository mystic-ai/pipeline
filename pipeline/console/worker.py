from pipeline.util.worker import Worker


def worker() -> int:
    worker = Worker()
    worker.begin()
    return 0
