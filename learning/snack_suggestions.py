"""Check and provide snack if available else ask user to pick something else from menu"""

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

AVAILABLE_SNACKS = {"pizza", "pasta"}


class SnackSelectorError(Exception):
    """Base exception for snack selector errors."""


class InvalidInputError(SnackSelectorError):
    """Raised when user provides invalid input."""

    def __init__(self, error_code: str):
        self.error_code = error_code
        messages = {
            "no_input": "No input provided",
            "interrupted": "User interrupted input",
            "no_data": "No input received",
        }
        super().__init__(messages.get(error_code, "Unknown input error"))


def get_user_input() -> str:
    """Get user input from console."""
    try:
        user_input = input("\nPlease provide your choice of snack: ")
        cleaned_input = user_input.strip()

        if not cleaned_input:
            raise InvalidInputError(ERROR_MESSAGES["no_input"])

        return cleaned_input.lower()

    except KeyboardInterrupt as e:
        raise InvalidInputError(ERROR_MESSAGES["interrupted"]) from e
    except EOFError as e:
        raise InvalidInputError(ERROR_MESSAGES["no_data"]) from e


def is_snack_available(snack: str) -> bool:
    """Check if the requested snack is available."""
    return snack in AVAILABLE_SNACKS


def process_snack_request(snack: str) -> None:
    """Process the snack request and log the appropriate response."""
    if is_snack_available(snack):
        logger.info("Here you go %s", snack)
    else:
        available_list = ", ".join(sorted(AVAILABLE_SNACKS))
        logger.info("Sorry, we don't have %s in our menu. Available options: %s", snack, available_list)


def main() -> None:
    """Main function to run the snack selector."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d - %(levelname)s - %(module)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        user_input = get_user_input()
        process_snack_request(user_input)

    except InvalidInputError:
        logger.exception("Invalid input")
        sys.exit(1)
    except Exception:
        logger.exception("Unexpected error")
        sys.exit(1)


if __name__ == "__main__":
    main()
