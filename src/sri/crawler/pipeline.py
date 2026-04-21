"""Pipeline module — saves crawled articles to disk as JSON files.

Each ArticleItem received from a spider is persisted to
data/raw/{source}/{id}.json following the team's interface contract.
"""

import json
from pathlib import Path

from sri.crawler.items import ArticleItem


class JsonPipeline:
    """Pipeline that saves ArticleItems as JSON files."""

    def __init__(self, output_directory: Path | str = "data/raw"):
        """Initialise the pipeline with the output directory.

        Args:
            output_directory: Root directory where JSON files will be saved.
                Files are organized as {output_dir}/{source}/{id}.json.
        """

        self._output_dir = Path(output_directory)

    def save_item(self, item: ArticleItem) -> None:
        """Save an ArticleItem as a JSON file.

        Args:
            item: An ArticleItem instance to be saved.
                Must contain 'source' and 'id' attributes for file organization.
        """

        # Build the path
        file_path = self._output_dir / item["source"] / f"{item['id']}.json"

        # Create the directory
        file_path.parent.mkdir(parents=True, exist_ok=True)

        file_path.write_text(
            json.dumps(dict(item), ensure_ascii=False, indent=2), encoding="utf-8"
        )
