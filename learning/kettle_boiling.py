"""Simple function to log if the kettle is boiling or not."""

from __future__ import annotations

import logging
import random

logger = logging.getLogger(__name__)


def kettle_boiling_notification(*, is_kettle_boiled: bool):
    """A simple function that logs if the kettle is boiled or not.
    Args:
        is_kettle_boiled (boolean): status of kettle boiling
    Returns:
        None
    Raises:
        None
    """
    if is_kettle_boiled:
        logger.info("Kettle Done! time to make Chai")
    else:
        logger.info("Kettle Not Done! wait for some more time.")


def main():
    # Configure logging so INFO messages are printed to the console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d - %(levelname)s - %(module)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    kettle_boiled = bool(random.getrandbits(1))
    kettle_boiling_notification(is_kettle_boiled=kettle_boiled)


if __name__ == "__main__":
    main()
