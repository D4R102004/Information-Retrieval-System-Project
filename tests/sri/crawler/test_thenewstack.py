"""Tests for the TheNewStackSpider class.

Verifies that the The New Stack spider correctly parses RSS XML feeds,
builds ArticleItems from <item> entries, and handles missing or invalid
XML fields gracefully.

All HTTP calls are mocked to avoid real network requests.
"""

from unittest.mock import MagicMock

from bs4 import BeautifulSoup

from sri.crawler.spiders.thenewstack import TheNewStackSpider


def test_build_item_returns_article_item_on_valid_xml():
    """_build_item should return an ArticleItem when XML item is valid."""

    # Arrange
    spider = TheNewStackSpider()

    xml = """
    <item>
        <title>Test Article</title>
        <link>https://example.com/article</link>
        <pubDate>Wed, 20 May 2026 21:11:28 +0000</pubDate>
        <category>AI</category>
        <category>Web</category>
        <content:encoded><![CDATA[
            <p>First paragraph.</p>
            <p>Second paragraph.</p>
        ]]></content:encoded>
    </item>
    """

    raw_item = BeautifulSoup(xml, "xml").find("item")

    # Act
    result = spider._build_item(raw_item)

    # Assert
    assert result is not None
    assert result["title"] == "Test Article"
    assert result["url"] == "https://example.com/article"
    assert result["source"] == "thenewstack"
    assert "First paragraph." in result["content"]
    assert "Second paragraph." in result["content"]
    assert "AI" in result["tags"]
    assert "Web" in result["tags"]


def test_build_item_returns_none_when_fields_missing():
    """_build_item should return None when required XML fields are missing."""

    # Arrange
    spider = TheNewStackSpider()

    xml = """
    <item>
        <title>Test Article</title>
        <link>https://example.com/article</link>
        <pubDate>Wed, 20 May 2026 21:11:28 +0000</pubDate>
        <category>AI</category>
    </item>
    """

    raw_item = BeautifulSoup(xml, "xml").find("item")

    # Act
    result = spider._build_item(raw_item)

    # Assert
    assert result is None


def test_fetch_articles_returns_articles_on_success():
    """fetch_articles should return ArticleItems when RSS feed is valid."""

    # Arrange
    spider = TheNewStackSpider(max_articles=2)

    rss_xml = """
    <rss>
        <channel>
            <item>
                <title>Article 1</title>
                <link>https://example.com/1</link>
                <pubDate>Wed, 20 May 2026 21:11:28 +0000</pubDate>
                <category>AI</category>
                <content:encoded><![CDATA[
                    <p>Content 1</p>
                ]]></content:encoded>
            </item>
            <item>
                <title>Article 2</title>
                <link>https://example.com/2</link>
                <pubDate>Wed, 20 May 2026 21:11:28 +0000</pubDate>
                <category>Web</category>
                <content:encoded><![CDATA[
                    <p>Content 2</p>
                ]]></content:encoded>
            </item>
        </channel>
    </rss>
    """

    spider._client.get = MagicMock(return_value=MagicMock(text=rss_xml))

    # Act
    result = spider.fetch_articles()

    # Assert
    assert len(result) == 2
    assert result[0]["title"] == "Article 1"
    assert result[1]["title"] == "Article 2"
    assert result[0]["source"] == "thenewstack"
