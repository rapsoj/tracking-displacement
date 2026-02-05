import logging


def setup_logging(log_file_name):
    # Create a logger
    logger = logging.getLogger(log_file_name)
    logger.setLevel(logging.DEBUG)

    # File handler (DEBUG level)
    file_handler = logging.FileHandler(log_file_name + ".log")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # Stream handler (INFO level)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler.setFormatter(stream_formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
