import logging
import logging.handlers
import socket
from dotenv import load_dotenv
import os
import inspect



# Define custom logging levels globally before anything else
DEBUG1_LEVEL = 11
DEBUG2_LEVEL = 12
DEBUG3_LEVEL = 13
DEBUG4_LEVEL = 14
DEBUG5_LEVEL = 15

logging.addLevelName(DEBUG1_LEVEL, "DEBUG-1")
logging.addLevelName(DEBUG2_LEVEL, "DEBUG-2")
logging.addLevelName(DEBUG3_LEVEL, "DEBUG-3")
logging.addLevelName(DEBUG4_LEVEL, "DEBUG-4")
logging.addLevelName(DEBUG5_LEVEL, "DEBUG-5")


from contextlib import contextmanager

@contextmanager
def suppress_logging(level=logging.WARNING):
    logger = logging.getLogger()  # You could also target a module/class logger
    previous_level = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(previous_level)


# Extend Logger class to support custom debug levels
def debug1(self, message, *args, **kwargs):
    if self.isEnabledFor(DEBUG1_LEVEL):
        # Set stacklevel=3 to skip the lambda and the custom function itself.
        kwargs.setdefault("stacklevel", 3)
        self._log(DEBUG1_LEVEL, message, args, **kwargs)

def debug2(self, message, *args, **kwargs):
    if self.isEnabledFor(DEBUG2_LEVEL):
        kwargs.setdefault("stacklevel", 3)
        self._log(DEBUG2_LEVEL, message, args, **kwargs)

def debug3(self, message, *args, **kwargs):
    if self.isEnabledFor(DEBUG3_LEVEL):
        kwargs.setdefault("stacklevel", 3)
        self._log(DEBUG3_LEVEL, message, args, **kwargs)

def debug4(self, message, *args, **kwargs):
    if self.isEnabledFor(DEBUG4_LEVEL):
        kwargs.setdefault("stacklevel", 4)
        self._log(DEBUG3_LEVEL, message, args, **kwargs)


def debug5(self, message, *args, **kwargs):
    if self.isEnabledFor(DEBUG5_LEVEL):
        kwargs.setdefault("stacklevel", 3)
        self._log(DEBUG5_LEVEL, message, args, **kwargs)

# Add the new methods to the Logger class BEFORE creating LoggingHandler
logging.Logger.debug1 = debug1
logging.Logger.debug2 = debug2
logging.Logger.debug3 = debug3
logging.Logger.debug4 = debug4
logging.Logger.debug5 = debug5

class LoggingHandler:
    def __init__(self, logging_level = os.getenv("LOGGING_LEVEL")):
        self.logging_level = logging_level
        self.setup_logging(logging_level)  # Ensure logging is set up

    @staticmethod
    def setup_logging(logging_level = os.getenv("LOGGING_LEVEL")):
        """
        Configures the logging settings for the application.

        Parameters:
            logging_level (str): The logging level to set.
                                Acceptable values are "DEBUG", "DEBUG-1", "DEBUG-3", "DEBUG-5",
                                "INFO", "WARNING", "ERROR", "CRITICAL".

        Returns:
            None
        """

        HOSTNAME = os.getenv("HOSTNAME")



        # Map logging level string to actual logging level
        if logging_level == "DEBUG":
            logging_level_value = logging.DEBUG
        elif logging_level == "DEBUG-1":
            logging_level_value = DEBUG1_LEVEL
        elif logging_level == "DEBUG-2":
            logging_level_value = DEBUG2_LEVEL
        elif logging_level == "DEBUG-3":
            logging_level_value = DEBUG3_LEVEL
        elif logging_level == "DEBUG-4":
            logging_level_value = DEBUG5_LEVEL
        elif logging_level == "DEBUG-4":
            logging_level_value = DEBUG5_LEVEL
        elif logging_level == "INFO":
            logging_level_value = logging.INFO
        elif logging_level == "WARNING":
            logging_level_value = logging.WARNING
        elif logging_level == "ERROR":
            logging_level_value = logging.ERROR
        elif logging_level == "CRITICAL":
            logging_level_value = logging.CRITICAL
        else:
            logging_level_value = logging.INFO  # Default to INFO if an unknown level is passed

        # Configure the root logger
        logger = logging.getLogger()
        logger.setLevel(logging_level_value)

        # Remove existing handlers to prevent duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)


        # Custom formatter that ensures a default for 'classname'
        class CustomFormatter(logging.Formatter):
            def format(self, record):
                try:
                    if not hasattr(record, 'classname'):
                        record.classname = 'N/A'
                    formatted = super().format(record)
                    # Only append line number info for error-level messages or above.
                    if record.levelno >= logging.ERROR:
                        # Check if exc_info is provided and valid.
                        if record.exc_info and record.exc_info[2]:
                            tb = record.exc_info[2]
                            # Traverse to the last traceback frame.
                            while tb and tb.tb_next:
                                tb = tb.tb_next
                            error_line = tb.tb_lineno if tb else record.lineno
                            formatted += f" (Error occurred at line: {error_line})"
                        else:
                            formatted += f" (Log call at line: {record.lineno})"
                    return formatted
                except Exception as e:
                    # In case something goes wrong in formatting, fall back to a safe representation.
                    return f"Formatting error: {e}"

        # Get the current stack (skip the current frame)
        class ClassNameFilter(logging.Filter):
            def filter(self, record):
                record.classname = self._get_caller_class_name()
                return True

            def _get_caller_class_name(self):
                stack = inspect.stack()
                for frame_info in stack[2:]:
                    module_name = frame_info.frame.f_globals.get('__name__', '')
                    # Skip logging modules and our own filtering module.
                    if module_name.startswith('logging') or module_name == __name__:
                        continue
                    # Skip known wrapper functions (adjust these names as needed)
                    if frame_info.function in ("_run_script_thread", "scriptThread"):
                        continue
                    # If there's a 'self', return its class name.
                    if 'self' in frame_info.frame.f_locals:
                        return frame_info.frame.f_locals['self'].__class__.__name__
                return "N/A"


        formatter = CustomFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(module)s - %(classname)s - %(funcName)s - %(message)s'
        )
        formatter_paperTrail = CustomFormatter(
            f'%(asctime)s DA_2025@{HOSTNAME} %(name)s - %(levelname)s - %(threadName)s - %(module)s - %(classname)s - %(funcName)s - %(message)s',
            datefmt='%b %d %H:%M:%S'
        )




        # Add a file handler
        file_handler = logging.FileHandler('streamlit.log')
        file_handler.setLevel(logging.DEBUG)  # Ensure all logs are written to file
        file_handler.setFormatter(formatter)
        file_handler.addFilter(ClassNameFilter())
        logger.addHandler(file_handler)

        # Add a stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging_level_value)  # Show logs in console based on configured level
        stream_handler.setFormatter(formatter)
        stream_handler.addFilter(ClassNameFilter())
        logger.addHandler(stream_handler)
        
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        load_dotenv(os.path.join(BASE_DIR, "..", ".env"))
        

        SYS_LOG_ACTIVE = os.getenv("SYS_LOG_ACTIVE")
        
        if SYS_LOG_ACTIVE == "True":
            syslog_handler = logging.handlers.SysLogHandler(
                address=(PAPERTRAIL_HOST, PAPERTRAIL_PORT),
            )
            PAPERTRAIL_HOST = os.getenv("PAPERTRAIL_HOST")
            PAPERTRAIL_PORT = int(os.getenv("PAPERTRAIL_PORT"))
            syslog_handler.setLevel(logging.DEBUG)  # Ensure all logs are written to file
            syslog_handler.setFormatter(formatter_paperTrail)
            syslog_handler.addFilter(ClassNameFilter())
            logger.addHandler(syslog_handler)


# Forward custom methods from the root logger to the logging module.
# This allows you to call logging.debug1(), logging.debug3(), and logging.debug5() directly.
logging.debug1 = lambda message, *args, **kwargs: logging.getLogger().debug1(message, *args, **kwargs)
logging.debug2 = lambda message, *args, **kwargs: logging.getLogger().debug2(message, *args, **kwargs)
logging.debug3 = lambda message, *args, **kwargs: logging.getLogger().debug3(message, *args, **kwargs)
logging.debug4 = lambda message, *args, **kwargs: logging.getLogger().debug(message, *args, **kwargs)
logging.debug5 = lambda message, *args, **kwargs: logging.getLogger().debug5(message, *args, **kwargs)

