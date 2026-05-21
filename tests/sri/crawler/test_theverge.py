"""Tests for the TheVergeSpider class.

Verifies that The Verge spider correctly parses Atom XML feeds,
builds ArticleItems from <entry> nodes, and handles missing or invalid
XML fields gracefully.

All HTTP calls are mocked to avoid real network requests.
"""

from unittest.mock import MagicMock

from bs4 import BeautifulSoup

from sri.crawler.spiders.theverge import TheVergeSpider


def test_build_item_returns_article_item_on_valid_xml():
    """_build_item should return an ArticleItem when Atom entry is valid."""

    # Arrange
    spider = TheVergeSpider()

    xml = """
    <entry>
        <title>Test Article</title>
        <link rel="alternate" href="https://example.com/article" />
        <published>2026-05-20T21:11:28Z</published>
        <category term="AI" />
        <category term="Web" />
        <content><![CDATA[
            <p>First paragraph.</p>
            <p>Second paragraph.</p>
        ]]></content>
    </entry>
    """

    entry = BeautifulSoup(xml, "xml").find("entry")

    # Act
    result = spider._build_item(entry)

    # Assert
    assert result is not None
    assert result["title"] == "Test Article"
    assert result["url"] == "https://example.com/article"
    assert result["source"] == "theverge"
    assert "First paragraph." in result["content"]
    assert "Second paragraph." in result["content"]
    assert "AI" in result["tags"]
    assert "Web" in result["tags"]


def test_build_item_returns_none_when_fields_missing():
    """_build_item should return None when required XML fields are missing."""

    # Arrange
    spider = TheVergeSpider()

    xml = """
    <entry>
        <title>Test Article</title>
        <published>2026-05-20T21:11:28Z</published>
        <category term="AI" />
    </entry>
    """

    entry = BeautifulSoup(xml, "xml").find("entry")

    # Act
    result = spider._build_item(entry)

    # Assert
    assert result is None


def test_fetch_articles_returns_articles_on_success():
    """fetch_articles should return ArticleItems when Atom feed is valid."""

    # Arrange
    spider = TheVergeSpider(max_articles=2)

    atom_xml = """
    <feed>
        <entry>
            <title>Article 1</title>
            <link rel="alternate" href="https://example.com/1" />
            <published>2026-05-20T21:11:28Z</published>
            <category term="AI" />
            <content><![CDATA[
                <p>Content 1</p>
            ]]></content>
        </entry>
        <entry>
            <title>Article 2</title>
            <link rel="alternate" href="https://example.com/2" />
            <published>2026-05-20T21:11:28Z</published>
            <category term="Web" />
            <content><![CDATA[
                <p>Content 2</p>
            ]]></content>
        </entry>
    </feed>
    """

    spider._client.get = MagicMock(return_value=MagicMock(text=atom_xml))

    # Act
    result = spider.fetch_articles()

    # Assert
    assert len(result) == 2
    assert result[0]["title"] == "Article 1"
    assert result[1]["title"] == "Article 2"
    assert result[0]["source"] == "theverge"
