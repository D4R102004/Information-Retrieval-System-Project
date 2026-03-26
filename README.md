# Information Retrieval System — Tecnología y Software

## Description

This system returns a ranked list of technology and software articles
relevant to the user's query. It uses Latent Semantic Indexing (LSI)
to understand hidden topics within documents, returning relevant results
even when they don't contain the exact query words. Documents are
collected automatically from the web using a crawler that searches for
technology and software content and saves it to disk.

## Architecture

| Module | Responsibility |
|---|---|
| `crawler` | Collects documents from the web |
| `indexer` | Processes text and builds search index |
| `retrieval` | LSI model — finds relevant documents |
| `vectordb` | Stores and searches document embeddings |
| `rag` | Generates natural language answers |
| `ranking` | Orders results by relevance |
| `web_search` | Fallback when local database is insufficient |
| `interface` | Visual UI for user queries |

## Requirements

- Python 3.10+
- pip

## Setup
```bash
# Clone the repository
git clone https://github.com/D4R102004/Information-Retrieval-System-Project.git
cd Information-Retrieval-System-Project

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
pre-commit install --hook-type commit-msg
```

## Usage
```bash
# Step 1 — crawl documents
python -m sri.crawler

# Step 2 — build index
python -m sri.indexer

# Step 3 — launch interface
python -m sri.interface
```

## Team

- Darío Francisco Alfonso (@D4R102004)
- Juan Carlos Carmenate (@Juank404)
- Sebastian González Alfonso (@sebagonz106)

## Bibliography

- Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K., &
  Harshman, R. (1990). Indexing by latent semantic analysis.
  *Journal of the American Society for Information Science*, 41(6),
  391–407.

## License

Academic project — Universidad de La Habana, 2025-2026.
