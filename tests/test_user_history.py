from recommendation import UserSearchHistory


def test_user_history_keeps_latest_five_searches(tmp_path):
    history = UserSearchHistory(tmp_path / "history.json", max_entries=10)

    for index in range(7):
        history.add_search(
            query=f"query {index}",
            retrieved_documents=[{"id": f"doc-{index}"}],
        )

    profile = history.build_profile(limit=5)

    assert profile["queries"] == ["query 2", "query 3", "query 4", "query 5", "query 6"]
    assert profile["seed_doc_ids"] == ["doc-2", "doc-3", "doc-4", "doc-5", "doc-6"]
    assert len(profile["searches_used"]) == 5
