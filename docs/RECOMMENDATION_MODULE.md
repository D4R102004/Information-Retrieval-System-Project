# Recommendation Module

This document describes the optional recommendation module added to the SRI/RAG project.

## Purpose

The recommendation module suggests documents from the local corpus using two complementary modes:

1. **Manual/content-based recommendations**: recommendations are generated from a query, explicit interests, or selected/liked document IDs.
2. **Automatic recommendations from search history**: recommendations are generated from the user's most recent searches in the Search tab.

The module is designed to work independently from the RAG answer generator, so it can run even when Ollama is not active.

## Files Added or Updated

| File | Purpose |
|---|---|
| `src/recommendation/recommender.py` | Content-based recommender using TF-IDF similarity, document metadata, freshness, and source signals. |
| `src/recommendation/user_history.py` | Stores and manages recent user searches in `data/user_history.json`. |
| `src/recommendation/__init__.py` | Exposes recommendation module classes. |
| `src/main_orchestator.py` | Integrates recommendation methods into the main backend API. |
| `ui/tabs/recommendation.py` | Adds the Gradio Recommendation tab. |
| `ui/services/orchestrator_service.py` | Adds service methods used by the UI to call the backend recommender. |
| `ui/app.py` | Connects the Search tab with the Recommendation tab so automatic recommendations update after searches. |
| `tests/test_recommender.py` | Tests the content-based recommender. |
| `tests/test_user_history.py` | Tests search history persistence. |

## Backend API

The main orchestrator exposes the following recommendation methods:

```python
orchestrator.refresh_recommender()
orchestrator.recommend_documents(...)
orchestrator.recommend_similar_documents(...)
orchestrator.recommend_from_retrieval(...)
orchestrator.record_search_history(...)
orchestrator.get_search_history(...)
orchestrator.clear_search_history(...)
orchestrator.recommend_from_history(...)
```

## Manual Recommendation Flow

Use `recommend_documents()` when the user provides a query, interests, or liked documents:

```python
result = orchestrator.recommend_documents(
    query="serverless APIs",
    interests="cloud computing, backend development",
    liked_doc_ids=["010a7286-edfa-4143-9e46-462829787546"],
    top_k=10,
)
```

Use `recommend_similar_documents()` to recommend documents similar to a specific document:

```python
result = orchestrator.recommend_similar_documents(
    document_id="010a7286-edfa-4143-9e46-462829787546",
    top_k=10,
)
```

## Automatic Recommendation Flow from Search History

The automatic recommender uses the user's latest searches to infer interests. Each successful Search tab query records:

```json
{
  "user_id": "default",
  "query": "cloudflare workers websocket",
  "timestamp": "2026-06-02T11:29:00+00:00",
  "retrieved_doc_ids": ["doc-id-1", "doc-id-2"]
}
```

The history is stored in:

```text
data/user_history.json
```

The backend then uses the latest 5 searches by default:

```python
result = orchestrator.recommend_from_history(
    user_id="default",
    top_k=10,
    history_limit=5,
)
```

The recommendation profile is created from the recent query texts and retrieved document IDs. Documents already retrieved in the recent searches can be excluded to avoid recommending the same items again.

## UI Behavior

The Gradio interface includes a **Recommendation** tab with:

- manual recommendations from interests/query text;
- similar-document recommendations;
- automatic recommendations based on recent searches;
- search history preview;
- a button to regenerate automatic recommendations;
- a button to clear the stored search history.

After each successful query in the **Search** tab, the UI records the search and refreshes the automatic recommendations shown in the **Recommendation** tab.

## Data Privacy and Reset

Search history is local to the project and stored only in `data/user_history.json`. To reset personalization, use the **Clear search history** button in the UI or call:

```python
orchestrator.clear_search_history(user_id="default")
```

## Testing

Run the recommendation-related tests with:

```bash
pytest tests/test_recommender.py tests/test_user_history.py
```

Or run the complete test suite:

```bash
pytest
```

## Notes and Limitations

- The recommender is content-based, not collaborative filtering.
- The automatic profile is based on recent searches, not on clicks or reading time.
- By default, at least the latest 5 searches are considered when available.
- If fewer than 5 searches exist, the module uses the available searches.
- The module requires `documents.json` to contain documents before meaningful recommendations can be generated.
