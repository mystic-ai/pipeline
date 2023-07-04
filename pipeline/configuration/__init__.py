import json
import os
import sys
import typing as t
from pathlib import Path

import yaml
from pydantic import BaseModel

PIPELINE_DIR = Path(
    os.getenv(
        "PIPELINE_DIR",
        Path(os.getenv("LOCALAPPDATA")) / ".pipeline/"
        if (sys.platform == "win32" or sys.platform == "cygwin")
        else Path.home() / ".pipeline/",
    )
)


class _RemoteModel(BaseModel):
    alias: t.Optional[str]
    url: str
    token: t.Optional[str]
    active: bool = False


class _ConfigurationModel(BaseModel):
    remotes: t.Optional[t.List[_RemoteModel]]


class Configuration:
    def __init__(
        self,
    ) -> None:
        self._config: _ConfigurationModel = None

        PIPELINE_DIR.mkdir(exist_ok=True)
        (PIPELINE_DIR / "files").mkdir(exist_ok=True)

    @property
    def active_remote(self) -> _RemoteModel:
        if self._config is not None and self._config.remotes is not None:
            for remote in self._config.remotes:
                if remote.active:
                    return remote
        return None

    @property
    def files_cache(self) -> str:
        return PIPELINE_DIR / "files"

    @property
    def remotes(self) -> t.List[_RemoteModel]:
        return self._config.remotes

    def load(self) -> None:
        path = PIPELINE_DIR / "config.yaml"
        if not path.exists():
            self._config = _ConfigurationModel()
            return

        try:
            with open(path, "r") as configuration_file:
                self._config = _ConfigurationModel.parse_obj(
                    yaml.load(
                        configuration_file,
                        Loader=yaml.FullLoader,
                    ),
                )
        except Exception:
            self._config = _ConfigurationModel()

    def save(self) -> None:
        path = PIPELINE_DIR / "config.yaml"

        PIPELINE_DIR.mkdir(exist_ok=True)

        with open(path, "w") as configuration_file:
            yaml.dump(
                json.loads(
                    self._config.json(
                        exclude_none=True,
                    )
                ),
                configuration_file,
            )

    def set_active_remote(self, alias: str) -> None:
        for remote in self._config.remotes:
            remote.active = remote.alias == alias
        self.save()

    def add_remote(
        self,
        alias: str,
        url: str,
        token: str,
    ) -> None:
        if self._config.remotes is None:
            self._config.remotes = []

        if any([remote.alias == alias for remote in self._config.remotes]):
            raise ValueError(f"Remote with alias '{alias}' already exists")

        self._config.remotes.append(
            _RemoteModel(
                alias=alias,
                url=url,
                token=token,
            )
        )
        self.save()

    def remove_remote(self, alias: str) -> None:
        if self._config.remotes is None:
            return

        alias_index_array = [
            idx for idx, obj in enumerate(self._config.remotes) if obj.alias == alias
        ]
        if len(alias_index_array) == 0:
            raise ValueError(f"Remote with alias '{alias}' does not exist")

        self._config.remotes.pop(alias_index_array[0])


current_configuration = Configuration()
current_configuration.load()
