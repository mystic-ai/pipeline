import os
import sys
from datetime import datetime

VERBOSE = bool(int(os.environ.get("VERBOSE", "1")))

LOG_FILE = None


class bcolors:
    PURPLE = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    ORANGE = "\33[38;5;208m"


levels = {
    "WARNING": bcolors.WARNING,
    "INFO": bcolors.PURPLE,
    "ERROR": bcolors.FAIL,
    "REMOTE_LOG": bcolors.OKCYAN,
    "SUCCESS": bcolors.OKGREEN,
}
PIPELINE_STR = f"{bcolors.OKBLUE}Pipeline{bcolors.ENDC}"
PIPELINE_FILE_STR = f"{bcolors.OKBLUE}File{bcolors.ENDC}"


def _print(val, level="INFO"):
    if not VERBOSE:
        return
    time_stamp = datetime.now().strftime("%H:%M:%S")

    log_str = (
        f"{PIPELINE_STR} {time_stamp} - [{levels[level]}{level}{bcolors.ENDC}]: {val}"
    )
    print(f"{log_str}")


def _print_remote_log(val: tuple):
    time_stamp = datetime.fromtimestamp(float(val[0]) / 1e9).strftime("%H:%M:%S.%f")[
        :-3
    ]
    text = val[1]
    normal_string = text

    log_str = f"{PIPELINE_STR} - {bcolors.ORANGE}logs{bcolors.ENDC} {time_stamp}: {normal_string}"  # noqa
    print(f"{log_str}")


def set_print_to_file(path: str):
    global LOG_FILE
    if LOG_FILE is None:
        LOG_FILE = open(path, "w")
        sys.stdout = LOG_FILE
    else:
        raise Exception("Already printing to a log file.")


def stop_print_to_file():
    global LOG_FILE
    if LOG_FILE is not None:
        LOG_FILE.close()
    else:
        raise Exception("Not printing to log file")
