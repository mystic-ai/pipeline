from argparse import ArgumentParser, _SubParsersAction

from pipeline.console import cluster
from pipeline.console.targets import environments, pipelines, pointers, resources


def create_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    create_parser = command_parser.add_parser(
        "create",
        description="Create a new object.",
        help="Create a new object.",
    )
    create_parser.set_defaults(func=lambda _: create_parser.print_help())

    create_sub_parser = create_parser.add_subparsers(
        dest="target",
    )

    environments.create_parser(create_sub_parser)
    pointers.create_parser(create_sub_parser)
    resources.create_parser(create_sub_parser)


def edit_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    edit_parser = command_parser.add_parser(
        "edit",
        description="Edit an object.",
        help="Edit and object.",
    )
    edit_parser.set_defaults(func=lambda _: edit_parser.print_help())

    edit_sub_parser = edit_parser.add_subparsers(
        dest="target",
    )

    environments.edit_parser(edit_sub_parser)
    pipelines.edit_parser(edit_sub_parser)
    pointers.edit_parser(edit_sub_parser)


def get_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    get_parser = command_parser.add_parser(
        "get",
        description="Get an object(s).",
        help="Get an object(s).",
    )
    get_parser.set_defaults(func=lambda _: get_parser.print_help())

    get_sub_parser = get_parser.add_subparsers(
        dest="target",
    )

    environments.get_parser(get_sub_parser)
    pipelines.get_parser(get_sub_parser)
    pointers.get_parser(get_sub_parser)
    resources.get_parser(get_sub_parser)


def delete_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    delete_parser = command_parser.add_parser(
        "delete",
        aliases=["del", "rm"],
        description="Delete an object.",
        help="Delete an object.",
    )
    delete_parser.set_defaults(func=lambda _: delete_parser.print_help())

    delete_sub_parser = delete_parser.add_subparsers(
        dest="target",
    )

    environments.delete_parser(delete_sub_parser)
    pipelines.delete_parser(delete_sub_parser)
    pointers.delete_parser(delete_sub_parser)
    resources.delete_parser(delete_sub_parser)


def cluster_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    cluster_parser = command_parser.add_parser(
        "cluster",
        aliases=["cl"],
        description="Manage compute clusters.",
        help="Manage compute clusters.",
    )
    cluster_parser.set_defaults(func=lambda _: cluster_parser.print_help())

    cluster_sub_parser = cluster_parser.add_subparsers(
        dest="target",
    )

    cluster.login_parser(cluster_sub_parser)
    cluster.use_parser(cluster_sub_parser)
    cluster.remove_parser(cluster_sub_parser)
    cluster.get_parser(cluster_sub_parser)
