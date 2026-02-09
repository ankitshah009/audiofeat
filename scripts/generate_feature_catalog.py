#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from audiofeat.catalog import build_feature_catalog, catalog_to_markdown, summarize_catalog


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate code-aligned feature catalog artifacts.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("docs/FEATURE_CATALOG.md"),
        help="Where to write the markdown catalog.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("docs/FEATURE_CATALOG.json"),
        help="Where to write the JSON catalog.",
    )
    args = parser.parse_args()

    catalog = build_feature_catalog()
    summary = summarize_catalog(catalog)

    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(catalog_to_markdown(catalog))

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps({"summary": summary, "components": catalog}, indent=2),
    )

    print(f"Wrote markdown catalog to {args.markdown_output}")
    print(f"Wrote JSON catalog to {args.json_output}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
