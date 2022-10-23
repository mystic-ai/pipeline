import argparse
import sys

from pipeline.util.worker import Worker


def main() -> int:
    base_parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Create or run pipelines locally or in the cloud!",
        add_help=True,
    )
    command_parser = base_parser.add_subparsers(dest="command")
    command_parser.add_parser("worker")

    args: argparse.Namespace = base_parser.parse_args()

    if args.command == "worker":
        worker = Worker()
        worker.begin()
    else:
        base_parser.print_help()


if __name__ == "__main__":
    sys.exit(main())
