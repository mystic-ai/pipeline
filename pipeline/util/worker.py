import contextlib
import sys

import dill

from pipeline.objects import Graph


class DummyFile(object):
    def write(self, x):
        pass


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout


class Worker:

    pipelines: dict = {}

    def __init__(self) -> None:
        pass

    def begin(self) -> None:
        print("worker-started", flush=True)
        while True:
            command = sys.stdin.readline()
            data = sys.stdin.readline()
            if command == "add-pipeline\n":
                graph: Graph = dill.loads(bytearray.fromhex(data.strip()))
                self.pipelines[graph.local_id] = graph
                print("done\n", flush=True)
            elif command == "run-pipeline\n":
                with nostdout():
                    run_info: dict = dill.loads(bytearray.fromhex(data.strip()))
                    pipeline_id = run_info.get("pipeline_id")
                    data = run_info.get("data")
                    pipeline: Graph = self.pipelines[pipeline_id]
                    result = pipeline.run(*data)
                print(f"{str(result).strip()}\n", flush=True)
            else:
                print("bad-command\n", flush=True)
