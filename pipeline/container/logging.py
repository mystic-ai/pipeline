import contextlib
import inspect
import json
import logging
import os
import sys
import traceback

from loguru import logger


class StreamToLogger:
    def __init__(self, level="INFO"):
        self._level = level

    def write(self, buffer):
        for line in buffer.rstrip().splitlines():
            logger.opt(depth=1).log(self._level, line.rstrip())

    def flush(self):
        pass


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def default_log_handler(message, file=sys.stderr):
    # No `end` as `message` already has a newline
    print(message, file=file, end="")


def json_log_handler(message, file=sys.stderr):
    record = message.record
    payload = {
        "log.level": record["level"].name,
        "log.time.timestamp": record["time"].timestamp(),
        "log.time.iso8601": record["time"].isoformat(),
        "log.source.file": f"{record['file'].name}:{record['line']}",
        "log.source.module": record["name"],
        "log.message": record["message"],
        **record["extra"],
    }
    if (ex := record["exception"]) is not None:
        payload["exception.type"] = repr(ex.type)
        payload["exception.value"] = repr(ex.value)
        payload["exception.traceback"] = traceback.format_tb(ex.traceback)
    print(json.dumps(payload), file=file)


def setup_logging():
    logging.getLogger("uvicorn").setLevel(logging.CRITICAL)
    logging.getLogger("uvicorn.error").setLevel(logging.CRITICAL)
    logging.getLogger("uvicorn.access").handlers.clear()
    logger.remove()
    use_json_logging = os.environ.get("USE_JSON_LOGGING", False)
    if use_json_logging:
        handler = dict(
            sink=json_log_handler,
            colorize=False,
        )
    else:
        handler = dict(
            sink=default_log_handler,
            colorize=True,
        )
    logger.configure(handlers=[handler])
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)


@contextlib.contextmanager
def redirect_stdout():
    stream = StreamToLogger()
    with contextlib.redirect_stdout(stream):
        yield
