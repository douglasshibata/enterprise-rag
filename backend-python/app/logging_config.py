import logging
import sys
from contextvars import ContextVar
from pythonjsonlogger import jsonlogger

# Context variable to store the request ID for the current request context
request_id_var: ContextVar[str] = ContextVar("request_id", default="")

class RequestIdFilter(logging.Filter):
    """
    A logging filter that injects the current request_id from the contextvars
    into the log record.
    """
    def filter(self, record):
        record.request_id = request_id_var.get()
        return True

def setup_logging():
    """
    Configures the root logger to output structured JSON logs with
    correlated request_id context.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clean up existing handlers to avoid duplicate output
    logger.handlers = []

    handler = logging.StreamHandler(sys.stdout)

    # Configure JSON Formatter with desired field mappings
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s %(request_id)s",
        rename_fields={
            "levelname": "severity",
            "asctime": "timestamp"
        }
    )
    handler.setFormatter(formatter)

    # Register the RequestIdFilter
    handler.addFilter(RequestIdFilter())

    logger.addHandler(handler)
