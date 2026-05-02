"""Items module — defines the data schema for crawled articles.

This module contains the Scrapy Item class that acts as a typed
container for data extracted by the spider before it is passed
to the pipeline for storage.
"""

import scrapy  # type: ignore


class ArticleItem(scrapy.Item):
    """A single technology article extracted from the web.

    Attributes:
        id: Unique UUID for the document.
        url: The canonical URL of the article.
        title: The headline of the article.
        date: Publication date as ISO 8601 string (YYYY-MM-DDTHH:MM:SSZ).
        content: The full body text of the article, stripped of HTML.
        source: The domain the article was crawled from.
        tags: List of topic tags or categories (may be empty).
    """

    id = scrapy.Field()
    url = scrapy.Field()
    title = scrapy.Field()
    date = scrapy.Field()
    content = scrapy.Field()
    source = scrapy.Field()
    tags = scrapy.Field()
