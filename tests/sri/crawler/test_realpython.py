"""Tests for the RealPythonSpider class.

Verifies that the spider correctly parses HTML pages,
extracts article fields using BeautifulSoup, and builds
ArticleItems properly. No real HTTP requests are made.
"""

from bs4 import BeautifulSoup

from sri.crawler.spiders.realpython import RealPythonSpider


def test_build_item_returns_article_item():
    """_build_item should return an ArticleItem when HTML is valid."""
    # Arrange
    spider = RealPythonSpider()
    html = """
    <html><body>
        <h1>Python Variables Guide</h1>
        <span class="text-muted">Publication date Apr 15, 2026</span>
        <p>This is the article content.</p>
        <a href="/tag/python">python</a>
    </body></html>
    """
    soup = BeautifulSoup(html, "html.parser")
    url = "https://realpython.com/python-variables/"

    # Act
    result = spider._build_item(url, soup)

    # Assert
    assert result is not None
    assert result["title"] == "Python Variables Guide"
    assert result["url"] == url
    assert result["source"] == "realpython"
    assert "python" in result["tags"]


def test_build_item_returns_none_when_missing_title():
    """_build_item should return None when the HTML has no <h1> title."""
    # Arrange
    spider = RealPythonSpider()
    html = """
    <html><body>
        <span class="text-muted">Publication date Apr 15, 2026</span>
        <p>This is the article content.</p>
        <a href="/tag/python">python</a>
    </body></html>
    """
    soup = BeautifulSoup(html, "html.parser")
    url = "https://realpython.com/python-variables/"

    # Act
    result = spider._build_item(url, soup)

    # Assert
    assert result is None
