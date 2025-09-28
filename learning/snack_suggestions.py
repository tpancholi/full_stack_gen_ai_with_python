"""Check and provide snack if available else ask user to pick something else from menu"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

# Define available snacks as a constant
AVAILABLE_SNACKS = {"pizza", "pasta"}


def get_user_input() -> str:
    """Get user input from console.

    Returns:
        str: User input in lowercase, or 'error' if invalid
    """
    try:
        user_input = input("\nPlease provide your choice of snack: ")
        # Better validation - check if input is not empty after stripping
        cleaned_input = user_input.strip()
        return cleaned_input.lower() if cleaned_input else "error"
    except KeyboardInterrupt:
        logger.info("\nGoodbye!")
        sys.exit(0)
    except EOFError:
        logger.info("\nNo input received. Exiting...")
        sys.exit(0)


def is_snack_available(snack: str) -> bool:
    """Check if the requested snack is available.

    Args:
        snack: The snack to check

    Returns:
        bool: True if snack is available, False otherwise
    """
    return snack in AVAILABLE_SNACKS


def main():
    """Main function to run the snack selector."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d - %(levelname)s - %(module)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    user_input = get_user_input()

    if user_input == "error":
        logger.error("Invalid input provided. Exiting...")
        sys.exit(1)  # Use exit code 1 for error
    elif is_snack_available(user_input):
        logger.info("Here you go %s", user_input)
    else:
        logger.info("Sorry, we don't have %s in our menu. Please try something else.", user_input)


if __name__ == "__main__":
    main()
