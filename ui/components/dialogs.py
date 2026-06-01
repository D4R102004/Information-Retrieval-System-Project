"""Dialog helpers for confirmation and feedback flows."""


def confirm_action(message: str) -> dict[str, str]:
    """Return a confirmation dialog payload."""
    return {"message": message}
