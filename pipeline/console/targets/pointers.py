from argparse import ArgumentParser, _SubParsersAction


def create_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    ...


def edit_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    ...


def get_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    ...


def delete_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    ...
