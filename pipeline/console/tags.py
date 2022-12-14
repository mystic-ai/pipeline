import argparse

from pipeline import PipelineCloud


def tags(args: argparse.Namespace) -> int:
    # sub_command = getattr(args, "sub-command", None)

    remote_service = PipelineCloud(verbose=False)
    remote_service.authenticate()
