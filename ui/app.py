"""Gradio application entry point for the frontend scaffold."""

from __future__ import annotations

from .config import APP_TITLE


def create_app() -> str:
    """Return a placeholder app descriptor for the frontend scaffold."""
    return APP_TITLE


def main() -> None:
    """Launch the frontend scaffold entry point."""
    print(create_app())


if __name__ == "__main__":
    main()
