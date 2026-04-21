"""Entry point for the crawler module.

Run with: python -m sri.crawler

Fetches technology articles from all configured spiders and saves
them as JSON files to data/raw/ for downstream indexing.
"""

import argparse
import sys

from sri.crawler.pipeline import JsonPipeline
from sri.crawler.spiders.devto import DevToSpider


def main() -> int:
    """Run the crawler and save articles to disk.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(description="SRI crawler")
    parser.add_argument(
        "--max-articles",
        type=int,
        default=500,
        help="Maximum number of articles to fetch (default: 500)",
    )
    args = parser.parse_args()

    # 1. Print that we're starting
    print(f"Starting SRI crawler with max_articles={args.max_articles}...")
    # 2. Create the spider and pipeline
    spider = DevToSpider(max_articles=args.max_articles)
    pipeline = JsonPipeline(output_directory="data/raw")
    try:
        # 3. Fetch articles
        collected_articles = spider.fetch_articles()

        # 4. Save each article
        for article in collected_articles:
            pipeline.save_item(article)
        # 5. Print summary
        print(f"Fetched and saved {len(collected_articles)} articles.")
        # 6. Return 0
        return 0
    except Exception as error:
        print(f"[ERROR] Crawler failed: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
