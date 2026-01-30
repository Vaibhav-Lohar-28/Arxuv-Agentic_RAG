import logging

logger = logging.getLogger(__name__)



def log_request(method: str, path: str) -> None:
    """Simple request logging for Week 1."""
    logger.info(f"{method} {path}")


def log_error(error: str, method: str, path: str) -> None:
    """Simple error logging for Week 1."""
    logger.error(f"Error in {method} {path}: {error}")
