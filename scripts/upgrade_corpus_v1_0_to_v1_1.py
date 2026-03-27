"""Upgrade corpus from v1.0 to v1.1 by filtering source_pdf entries.

Filters each record's source_pdf list to keep only entries containing 'abstract' field.
Removes entries without 'abstract' from source_pdf. Records with empty source_pdf after
filtering are kept as-is.
"""

import json
import os
from pathlib import Path


def upgrade_corpus():
    """Upgrade corpus v1.0 to v1.1."""
    v1_0_dir = Path("corpus/v1.0/data")
    v1_1_dir = Path("corpus/v1.1/data")

    # Get all JSON files except example.json
    json_files = [f for f in v1_0_dir.glob("*.json") if f.name != "example.json"]

    for json_file in json_files:
        print(f"Processing {json_file.name}...")

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Filter each record and remove records with empty source_pdf
        filtered_data = []
        for record in data:
            if "source_pdf" in record and isinstance(record["source_pdf"], list):
                # Keep only entries with 'abstract' field
                record["source_pdf"] = [
                    item for item in record["source_pdf"]
                    if "abstract" in item
                ]
                # Keep record only if source_pdf is not empty
                if record["source_pdf"]:
                    filtered_data.append(record)
            else:
                filtered_data.append(record)

        # Write to v1.1
        output_file = v1_1_dir / json_file.name
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)

        print(f"  Saved to {output_file.name}")


if __name__ == "__main__":
    upgrade_corpus()
    print("Upgrade complete!")
