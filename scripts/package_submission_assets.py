#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import lira_pipeline  # noqa: E402 - repository path is added above


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the V44 Kaggle asset manifest.")
    parser.add_argument("--require-bundle", action="store_true", help="Fail if the V44 model bundle does not exist.")
    args = parser.parse_args()

    if args.require_bundle and not lira_pipeline.V44_MODEL_BUNDLE_PATH.exists():
        raise RuntimeError(
            f"Missing model bundle: {lira_pipeline.V44_MODEL_BUNDLE_PATH}. "
            "Run `python lira_pipeline.py cv` before packaging Kaggle assets."
        )

    manifest = lira_pipeline.write_kaggle_manifest(include_bundle=args.require_bundle, include_hashes=True)
    print(f"Saved Kaggle asset manifest: {lira_pipeline.V44_MANIFEST_PATH}")
    print(f"Pipeline version: {manifest['pipeline_version']}")
    print(f"Entries: {len(manifest['entries'])}")


if __name__ == "__main__":
    main()


