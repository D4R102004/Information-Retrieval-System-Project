# Quick Start Development Guide

**Fast-Track Guide for Developers and Researchers**

---

## 5-Minute Setup

### Prerequisites Check

```bash
# Check Python version (need 3.10+)
python --version

# Check pip is available
pip --version
```

### Installation

```bash
# Clone and enter directory
git clone https://github.com/D4R102004/Information-Retrieval-System-Project.git
cd Information-Retrieval-System-Project

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install project
pip install -e ".[dev]"

# Start Ollama (in another terminal)
ollama pull llama3.2:latest && ollama serve
```

### First Query

```bash
# In the project directory
python src/main.py --query "What is machine learning?" --verbose
```

**Expected Output:**
```
[OK] System ready

=======================================================================
QUERY: What is machine learning?
=======================================================================

ANSWER
------
Machine learning is a subset of artificial intelligence...

METADATA
--------
Local documents used:  5
Web documents used:    0
Total documents:       5
Insufficiency:         No
Generation time:       2.34s

CITATIONS
---------
[1] Deep Learning Fundamentals (https://example.com/article)
    Snippet: Neural networks are computational systems...

[2] ML in Practice (https://example.com/article2)
    Snippet: Supervised learning requires labeled data...

```

---

## 10-Minute Full Workflow

### Step 1: Load Data

```bash
python src/main.py --load-data --verbose
```

Expected: 1000+ documents indexed from crawlers

### Step 2: Check Status

```bash
python src/main.py --status
```

Expected: `Status: healthy | Indexed Documents: 1000+`

### Step 3: Ask Multiple Questions

```bash
python src/main.py --interactive
```

Then at the prompt:
```
ask How does LSI work?
ask What is ChromaDB?
ask Explain vector embeddings
status
exit
```

### Step 4: Run Evaluation

```bash
python src/main.py --evaluate
```

Expected: IR metrics (MAP, MRR, P@5, etc.)

---

## Common Tasks

### Load Fresh Data

```bash
# Standard load (uses cached documents if available)
python src/main.py --load-data

# Force fresh crawl
python src/main.py --load-data --force

# With custom limits
python src/main.py --load-data --max-articles 500
```

### Query with Custom Parameters

```bash
# More local results
python src/main.py --query "Your question" --max-local 10

# Disable web search
python src/main.py --query "Your question" --no-web-search

# Both
python src/main.py --query "Your question" --max-local 8 --no-web-search
```

### Debugging

```bash
# Enable verbose logging
python src/main.py --query "test" --verbose

# Save logs to file
python src/main.py --query "test" --log-file debug.log --verbose

# Check what's in the logs
tail -f debug.log
```

### Database Management

```bash
# View database health
python src/main.py --status

# Clear database
python src/main.py --clear-db --force

# Re-index from consolidated file
python src/main.py --load-data
```

---

## Python API Usage (Quick Reference)

### Basic Query

```python
from src.main_orchestator import MainOrchestator

orchestrator = MainOrchestator()

# Complete pipeline
response = orchestrator.query("Your question")
print(response.answer)
print([c.title for c in response.citations])
```

### Retrieve Only (No RAG)

```python
result = orchestrator.retrieve_documents(
    question="Your question",
    max_local_results=10,
    use_web_search=True
)

documents = result['documents']
metadata = result['metadata']
```

### Custom Document RAG

```python
my_docs = [
    {
        "id": "1",
        "title": "Article Title",
        "content": "Article content here...",
        "url": "https://example.com"
    }
]

response = orchestrator.augment_response(
    question="Your question",
    documents=my_docs
)
```

### System Status

```python
status = orchestrator.get_status()
print(f"Indexed: {status['database']['indexed_documents']}")
print(f"Status: {status['database']['status']}")
```

### Load Data

```python
result = orchestrator.load_documents_from_crawlers(max_articles=1000)
print(f"Indexed {result['indexed_documents']} documents")
```

### Run Evaluation

```python
eval_result = orchestrator.evaluate_test()
print(f"MAP: {eval_result['aggregate']['MAP']:.3f}")
print(f"MRR: {eval_result['aggregate']['MRR']:.3f}")
```

---

## Testing

### Run All Tests

```bash
# Full test suite
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific module
pytest tests/sri/retrieval/ -v
```

### Run Specific Test

```bash
# Test LSI model
pytest tests/sri/retrieval/test_lsi_model.py -v

# Test RAG module
pytest tests/sri/rag/test_rag_module.py -v

# Test evaluation
pytest tests/sri/evaluation/ -v
```

### Test With Coverage Report

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Code Quality

### Format Code

```bash
# Auto-format all Python files
ruff format src/

# Check formatting
ruff check src/
```

### Lint Code

```bash
# Check for errors and warnings
ruff check src/ --extend-select=E,W,F

# Auto-fix simple issues
ruff check src/ --fix
```

### Full Quality Check

```bash
make check  # Runs all linters and formatters
```

---

## Configuration Changes

### Change LLM Model

```python
# In src/rag/config.py or at runtime
from rag.config import config

config.ollama_model = "mistral"
config.temperature = 0.5
config.max_tokens = 2048
```

### Change RAG Template

```python
from rag.rag_module import RAGModule
from rag.llm_provider import OllamaProvider

llm = OllamaProvider()
rag = RAGModule(llm, template_type="chain_of_thought")
```

### Change Search Parameters

```python
from sri.crawler.settings import CrawlerSettings

settings = CrawlerSettings()
settings.MIN_RESULTS_FOR_QUERY = 3
settings.MIN_AVG_SCORE_THRESHOLD = 0.4
```

---

## Common Errors & Fixes

### Error: "Ollama connection failed"

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Pull model
ollama pull llama3.2:latest

# Terminal 3: Test system
python src/main.py --query "test"
```

### Error: "Database is empty"

```bash
python src/main.py --load-data
```

### Error: "ChromaDB not installed"

```bash
# System will auto-use custom backend
# Or install ChromaDB explicitly:
pip install chromadb
```

### Error: "Import failed"

```bash
# Make sure you're in project root
cd Information-Retrieval-System-Project

# And virtual environment is activated
source .venv/bin/activate
```

---

## Development Workflow

### Making Changes

```bash
# 1. Create feature branch
git checkout -b feature/my-feature

# 2. Make changes
# 3. Test locally
pytest tests/ -v

# 4. Format and lint
make format
make lint

# 5. Commit with conventional commits
git add .
git commit -m "feat: description of changes"

# 6. Push
git push origin feature/my-feature
```

### Adding New Components

**Adding a new prompt template:**

```python
# src/rag/prompt_templates.py
class MyTemplate(BasePromptTemplate):
    def apply(self, query, documents):
        # Your logic
        return prompt, metadata

# Register it
PromptTemplateFactory.register("my_template", MyTemplate)

# Use it
rag = RAGModule(llm, template_type="my_template")
```

**Adding a new crawler:**

```python
# src/sri/crawler/spiders/myspider.py
class MySpider(scrapy.Spider):
    name = "myspider"
    allowed_domains = ["example.com"]
    start_urls = ["https://example.com"]

    def parse(self, response):
        # Your scraping logic
        yield {...}

# Update settings to enable
# src/sri/crawler/settings.py: add "myspider" to SPIDERS list
```

---

## Performance Tips

### Speed Up Queries

```python
# 1. Fewer local results
orchestrator.query(question, max_local_results=3)

# 2. Disable web search if not needed
orchestrator.query(question, use_web_search=False)

# 3. Use smaller LLM model
config.ollama_model = "mistral"

# 4. Lower max_tokens
config.max_tokens = 512
```

### Speed Up Indexing

```python
# Use fewer components
pipeline = SRIPipeline(lsi_components=50)

# Index in batches
for batch in chunks(documents, 100):
    pipeline.index(batch)
```

### Reduce Memory Usage

```bash
# Use smaller embedding model
# (modify VectorStore initialization)

# Or disable ChromaDB (use custom backend)
vstore = VectorStore(use_chromadb=False)
```

---

## Monitoring & Logging

### View Logs

```bash
# Real-time logs
tail -f system.log

# Last 50 lines
tail -50 system.log

# Search for errors
grep ERROR system.log

# Count by level
grep -c INFO system.log
grep -c ERROR system.log
```

### Enable Debug Logging

```bash
# CLI
python src/main.py --query "test" --verbose --log-file debug.log

# Python API
import logging
logging.basicConfig(level=logging.DEBUG)

orchestrator = MainOrchestator()
```

---

## Docker (Optional)

```bash
# Build
docker-compose build

# Run
docker-compose up

# Access
# Frontend: http://localhost:7860
# API: http://localhost:8000

# Stop
docker-compose down
```

---

---

## Visual Interface (Gradio UI)

The system includes a full graphical interface accessible from the browser.

### Launching the UI

Launch with:
  source .venv/bin/activate
  PYTHONPATH=src python -m ui.app

Then open: http://127.0.0.1:7860

---

### Tab 1: Search

Type your question in the Search query box and press Search.

Examples:
- What is Docker and how does it work?
- How do I use async/await in Python?
- Best practices for REST API design

Results shown: ranked documents with source, score, URL, snippet.
RAG answer: synthesized answer with source citations.
Press Clear to reset.

---

### Tab 2: Configuration

Adjust settings before searching. Press Save Query Settings to apply.

Query Parameters:
- Max Local Results (default 5): documents from local index
- Max Web Results (default 5): documents from web fallback
- Enable Web Search: DuckDuckGo when local is insufficient
- Auto-reload Empty Database: run crawlers if index is empty

RAG Configuration:
- Ollama Model: llama3.2:latest
- Temperature: 0.7
- Max Tokens: 1024
- Max Citations in Response: 10
- Max Document Context Length: 1000 chars

Database Management:
- Clear All Data, Reindex Database, Reload from Crawlers, Refresh Status

---

### Tab 3: Evaluation

1. Test Configuration - Load Default Test
2. Run Evaluation - Run Evaluation
3. Results - Display Results

Shows: MAP, MRR, NDCG@5, P@5, R@10

Custom tests: use Test Designer tab to add queries with relevant document IDs.

---

### Tab 4: System Status

Click Refresh All Status to see:
- Health Summary (Healthy/Degraded/Critical)
- Database document counts
- Crawler status per source
- LLM Service connectivity

---

## Next Steps

1. **Read the full README.md** for complete documentation
2. **Explore BACKEND_ARCHITECTURE.md** for system design details
3. **Check FRONTEND_IMPLEMENTATION_PLAN.md** for UI specifications
4. **Run tests** to understand component interactions
5. **Try modifications** to experiment with the system

---

**Happy hacking!** 🚀

For issues, open a GitHub issue with:
- Python version
- Error message
- Steps to reproduce
- System logs (with --verbose)
