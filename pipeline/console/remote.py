import argparse

from pipeline import PipelineCloud, configuration
from pipeline.exceptions.MissingActiveToken import MissingActiveToken
from pipeline.util.logging import _print


def remote(args: argparse.Namespace) -> int:
    sub_command = getattr(args, "sub-command", None)

    if sub_command == "set":
        default_url = args.url
        configuration.config["DEFAULT_REMOTE"] = default_url
        configuration._save_config()
        _print(f"Set new default remote to '{configuration.config['DEFAULT_REMOTE']}'")
        return 0
    elif sub_command in ["list", "ls"]:
        remotes = [
            f"{_remote} (active)"
            if _remote == configuration.DEFAULT_REMOTE
            else f"{_remote}"
            for _remote in configuration.remote_auth.keys()
        ]
        _print("Authenticated remotes:")
        [print(_remote) for _remote in remotes]
        return 0
    elif sub_command == "login":
        try:
            PipelineCloud(token=args.token, url=args.url, verbose=False)
        except MissingActiveToken:
            _print(f"Couldn't authenticate with {args.url}", level="ERROR")
            return 1

        configuration.remote_auth[args.url] = args.token
        configuration._save_auth()
        _print(f"Successfully authenticated with {args.url}")
        return 0
