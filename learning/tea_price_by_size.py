"""A program to calculate tea price by cup size"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

# Error message constants
ERROR_MESSAGES = {
    "no_input": "No input provided",
    "interrupted": "User interrupted input",
    "no_data": "No input received",
    "invalid_choice": "Invalid snack choice",
}

PRICE_CHART = {
    "small": 10,
    "medium": 15,
    "large": 20,
}


class CupSizeSelectorError(Exception):
    """Base exception for cup size selector errors."""


class InvalidInputError(CupSizeSelectorError):
    """Raised when the user provides invalid input."""


def get_user_input() -> str:
    """Get user input from the console."""
    try:
        user_input = input("\nPlease provide your choice of cup size: ")
        cleaned_input = user_input.strip().lower()

        if not cleaned_input:
            raise InvalidInputError(ERROR_MESSAGES["no_input"])
        return cleaned_input  # noqa: TRY300
    except KeyboardInterrupt as e:
        raise InvalidInputError(ERROR_MESSAGES["interrupted"]) from e
    except EOFError as e:
        raise InvalidInputError(ERROR_MESSAGES["no_data"]) from e


def process_user_request(user_input: str) -> None:
    """Process the user request and ask for the appropriate price"""
    if user_input in PRICE_CHART:
        logger.info("Price of %s cup of team would be %d", user_input, PRICE_CHART[user_input])
    else:
        logger.info("Sorry, unknown cup size. Please try again later.")


def main() -> None:
    """Main function to run the snack selector."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d - %(levelname)s - %(module)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        user_input = get_user_input()
        process_user_request(user_input)
    except InvalidInputError:
        logger.exception("Invalid input")
        sys.exit(1)
    except Exception:
        logger.exception("Unexpected error")
        sys.exit(1)


if __name__ == "__main__":
    main()
