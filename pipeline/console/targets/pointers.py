import re
from argparse import ArgumentParser, _SubParsersAction

VALID_TAG_NAME = re.compile(
    r"^[a-z0-9][a-z0-9-._/]*[a-z0-9]:[0-9A-Za-z_][0-9A-Za-z-_.]{0,127}$"
)


def create_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    ...


def edit_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    ...


def get_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    ...


def delete_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    ...
