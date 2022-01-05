import sys
from datetime import datetime

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


levels = {"WARNING": bcolors.ORANGE, "INFO": bcolors.PURPLE, "ERROR": bcolors.FAIL}
PIPELINE_STR = f"{bcolors.OKBLUE}Pipeline{bcolors.ENDC}"


def _print(val, level="INFO"):
    time_stamp = datetime.utcnow().strftime("%H:%M:%S")

    log_str = (
        f"{PIPELINE_STR} {time_stamp} - [{levels[level]}{level}{bcolors.ENDC}]: {val}"
    )
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
