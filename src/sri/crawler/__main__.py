"""Entry point for the crawler module.

Run with: python -m sri.crawler

Fetches technology articles from all configured spiders and saves
them as JSON files to data/raw/ for downstream indexing.
"""

import argparse
import sys

from sri.crawler.pipeline import JsonPipeline
from sri.crawler.spiders.devto import DevToSpider
from sri.crawler.spiders.hackernews import HackerNewsSpider
from sri.crawler.spiders.realpython import RealPythonSpider


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
    spiders = [
        DevToSpider(max_articles=args.max_articles),
        HackerNewsSpider(max_articles=args.max_articles),
        RealPythonSpider(max_articles=args.max_articles),
    ]
    pipeline = JsonPipeline(output_directory="data/raw")
    try:
        # 3. Fetch articles
        total_saved = 0
        for spider in spiders:
            print(f"Running {spider.__class__.__name__}...")
            articles = spider.fetch_articles()
            # 4. Save each article
            for article in articles:
                pipeline.save_item(article)
            # 5. Print summary
            print(f"  Saved {len(articles)} articles.")
            total_saved += len(articles)

        print(f"Done. Total articles saved: {total_saved}")
        return 0

    except Exception as error:
        print(f"[ERROR] Crawler failed: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
