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
        self.encoding = "utf-8"

    def write(self, buffer):
        for line in buffer.rstrip().splitlines():
            logger.opt(depth=1).log(self._level, line.rstrip())

    def flush(self):
        pass


@contextlib.contextmanager
def redirect_stdout():
    """Contextmanager for redirecting stdout to core logger.

    For more info see:
    https://loguru.readthedocs.io/en/stable/resources/recipes.html#capturing-standard-stdout-stderr-and-warnings
    """
    stream = StreamToLogger()
    with contextlib.redirect_stdout(stream):
        yield


class InterceptHandler(logging.Handler):
    """
    Ensure standard logging calls get forwarded to core Loguru logger
    https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging
    """

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
        payload["exception.traceback"] = "".join(
            traceback.format_exception(ex.type, ex.value, ex.traceback, limit=200)
        )
    print(json.dumps(payload, default=str, ensure_ascii=False), file=file)


def setup_logging():
    """Setup logging:
    - Suppress uvicorn logs
    - Use loguru as default logging library
    - If USE_JSON_LOGGING is set, write JSON logs (these can then be queried and
        returned by main API), otherwise use default log format
    - Use InterceptHandler to forward all logs to loguru
    """
    # suppress uvicorn logs
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
