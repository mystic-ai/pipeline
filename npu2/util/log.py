from datetime import datetime
class bcolors:
    PURPLE = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ORANGE = '\33[38;5;208m'


levels = {"WARNING": bcolors.ORANGE, "INFO": bcolors.PURPLE, "ERROR": bcolors.FAIL}
NEURO_STR = f"[{bcolors.OKBLUE}Neuro{bcolors.ENDC}]"

def npu_print(val, level="INFO"):
    time_stamp = datetime.utcnow().strftime("%H:%M:%S")
    
    log_str = f"{NEURO_STR} {time_stamp} - [{levels[level]}{level}{bcolors.ENDC}]: {val}"
    print(f"{log_str}")