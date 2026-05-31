"""Import compatibility tests for WebSearcher."""

# Standard library
import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_searcher_imports_duckduckgo_search_when_ddgs_is_unavailable(tmp_path):
    """Searcher should support the declared duckduckgo-search dependency."""
    package_dir = tmp_path / "duckduckgo_search"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("class DDGS:\n    pass\n")
    (tmp_path / "httpx.py").write_text("class HTTPError(Exception):\n    pass\n")
    bs4_dir = tmp_path / "bs4"
    bs4_dir.mkdir()
    (bs4_dir / "__init__.py").write_text("class BeautifulSoup:\n    pass\n")

    script = textwrap.dedent(
        """
        import importlib.util
        import sys

        sys.path = [path for path in sys.path if "site-packages" not in path]

        spec = importlib.util.spec_from_file_location(
            "searcher_under_test",
            "src/sri/web_search/searcher.py",
        )
        searcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(searcher)

        assert searcher.DDGS.__module__ == "duckduckgo_search"
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)

    subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        env=env,
        check=True,
        text=True,
    )
