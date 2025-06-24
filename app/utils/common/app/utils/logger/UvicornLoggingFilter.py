import logging

class SuppressStatusPollLogs(logging.Filter):
    def filter(self, record):
        return "/tasks/status" not in record.getMessage()

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": {
        "suppress_status": {
            "()": SuppressStatusPollLogs
        }
    },
    "handlers": {
        "uvicorn_access": {
            "class": "logging.StreamHandler",
            "filters": ["suppress_status"],
            "formatter": "default"
        }
    },
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(levelname)s - %(message)s"
        }
    },
    "loggers": {
        "uvicorn.access": {
            "handlers": ["uvicorn_access"],
            "level": "INFO",
            "propagate": False
        }
    }
}
