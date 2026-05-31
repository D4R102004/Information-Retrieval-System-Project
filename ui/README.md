# UI Frontend

This directory contains the Gradio-based frontend scaffolding for the information retrieval system.

## Layout
- `app.py`: application entry point.
- `config.py`: UI-specific constants and defaults.
- `state.py`: session state helpers.
- `services/`: backend-facing service adapters.
- `tabs/`: Gradio tab builders.
- `components/`: reusable UI primitives.
- `styles/`: theme and custom CSS.
- `assets/`: static assets for the frontend.

## Goal
Provide a clear separation between presentation, orchestration, and system operations so the UI can evolve without coupling to backend implementation details.
