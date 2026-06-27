import logging
import sys
from pathlib import Path


def setup_logging(
        out_file="temp.log",
        format="%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s: %(message)s",
        level=logging.DEBUG
    ):
    logging.basicConfig(
        filename=out_file,
        format=format,
        filemode="w",
        level=level,
    )
    logging.getLogger().setLevel(level)

    # Create a console handler manually
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(format))
    logging.getLogger().addHandler(console_handler)

    # Hook uncaught exceptions
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logging.error(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    sys.excepthook = handle_exception


def setup_logging_v2(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)

    format = "%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s: %(message)s"

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(format))
    root_logger.addHandler(file_handler)

    # Create a console handler manually
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(format))
    logging.getLogger().addHandler(console_handler)

    # Hook uncaught exceptions
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logging.error(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    sys.excepthook = handle_exception
