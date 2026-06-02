from recommendation import ContentBasedRecommender


def _sample_docs():
    return [
        {
            "id": "doc-python-api",
            "title": "Building Python APIs",
            "content": "FastAPI Python backend REST APIs testing deployment",
            "tags": "python, api, backend",
            "source": "sample",
            "date": "2026-01-01T00:00:00Z",
        },
        {
            "id": "doc-cloudflare-chat",
            "title": "Realtime chat on Cloudflare Workers",
            "content": "Cloudflare Durable Objects websocket realtime serverless JavaScript",
            "tags": "cloudflare, serverless, websockets",
            "source": "sample",
            "date": "2026-01-02T00:00:00Z",
        },
        {
            "id": "doc-design",
            "title": "Design systems for product teams",
            "content": "UI components accessibility typography colors tokens",
            "tags": "design, ui",
            "source": "sample",
            "date": "2026-01-03T00:00:00Z",
        },
    ]


def test_recommend_by_query_returns_relevant_document(tmp_path):
    recommender = ContentBasedRecommender(tmp_path / "missing.json")
    recommender.load_documents(_sample_docs())

    result = recommender.recommend(query="serverless websocket cloudflare", top_k=1)

    assert result["status"] == "success"
    assert result["recommendations"][0]["id"] == "doc-cloudflare-chat"


def test_similar_to_document_excludes_seed_document(tmp_path):
    recommender = ContentBasedRecommender(tmp_path / "missing.json")
    recommender.load_documents(_sample_docs())

    result = recommender.similar_to_document("doc-python-api", top_k=2)

    ids = [doc["id"] for doc in result["recommendations"]]
    assert result["status"] == "success"
    assert "doc-python-api" not in ids
