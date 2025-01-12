from logging import getLogger, StreamHandler, Formatter, FileHandler, LogRecord
from traceback import format_stack, format_exc
from colorama import Fore, Style, Back
from datetime import datetime
from os import makedirs

LOG_DIR = ".logs"
LOGGER_NAME = "poisoning"

makedirs(LOG_DIR, exist_ok=True)


def asctime():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class ShellHandler(StreamHandler):
    class ShellFormatter(Formatter):
        COLORS = {
            "DEBUG": Fore.CYAN,
            "INFO": Fore.GREEN,
            "WARNING": Fore.YELLOW,
            "ERROR": Fore.RED,
            "CRITICAL": Fore.RED,
            "FATAL": Fore.RED,
        }

        def format(self, record: LogRecord):
            color = self.COLORS.get(record.levelname, Fore.RESET)
            return f"{color}{record.levelname.lower():8} {Style.RESET_ALL}{record.msg}"

    def __init__(self):
        super().__init__()
        self.setLevel("INFO")
        self.setFormatter(ShellHandler.ShellFormatter())

    def emit(self, record: LogRecord):
        super().emit(record)


class LogHandler(FileHandler):
    class LogFormatter(Formatter):
        def format(self, record: LogRecord):
            if (record.levelname == "CRITICAL") or (record.levelname == "FATAL") or (record.levelname == "ERROR"):
                record.msg += f"\n{''.join(format_exc())}"

            # Remove every Fore, Back or Style character from the message
            fore = Fore.__dict__.values()
            back = Back.__dict__.values()
            style = Style.__dict__.values()

            full = [*fore, *back, *style]

            for f in full:
                record.msg = record.msg.replace(f, "")

            return f"[{asctime()}] {record.levelname}: {record.msg}"

    def __init__(self):
        super().__init__(f"{LOG_DIR}/log-{timestamp()}.log")
        self.setLevel("DEBUG")
        self.setFormatter(LogHandler.LogFormatter())

    def emit(self, record: LogRecord):
        super().emit(record)
        if record.levelname == "CRITICAL":
            raise SystemExit(1)


logger = getLogger(LOGGER_NAME)
logger.setLevel("DEBUG")
logger.addHandler(ShellHandler())
logger.addHandler(LogHandler())
