import io
import sys

import dill

from pipeline.objects import Graph


class stdsurpress:
    def __init__(self) -> None:
        pass

    def __enter__(self):
        self.save_stdout = sys.stdout
        self.save_stderr = sys.stderr
        sys.stdout = io.BytesIO()
        sys.stderr = io.BytesIO()

    def __exit__(self, type, value, traceback):
        sys.stdout = self.save_stdout
        sys.stderr = self.save_stderr


class Worker:

    pipelines: dict = {}

    def __init__(self) -> None:
        pass

    def begin(self) -> None:
        print("worker-started", flush=True)
        while True:
            command = sys.stdin.readline()
            data = sys.stdin.readline()
            try:
                if command == "add-pipeline\n":
                    graph: Graph = dill.loads(bytearray.fromhex(data.strip()))
                    self.pipelines[graph.local_id] = graph
                    print("done\n", flush=True)
                elif command == "run-pipeline\n":
                    with stdsurpress():
                        run_info: dict = dill.loads(bytearray.fromhex(data.strip()))
                        pipeline_id = run_info.get("pipeline_id")
                        data = run_info.get("data")
                        pipeline: Graph = self.pipelines[pipeline_id]
                        result = pipeline.run(*data)
                    print(f"{str(result).strip()}\n", flush=True)
                else:
                    print("bad-command\n", flush=True)
            except Exception:
                print("failed\n", flush=True)
