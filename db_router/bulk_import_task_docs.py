#!/usr/bin/env python3
import os
import glob
import argparse
import requests

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000",
                    help="Router API base URL")
    ap.add_argument("--folder", required=True,
                    help="Folder containing per-task CSVs")
    ap.add_argument("--pattern", default="*.csv",
                    help="Glob pattern inside folder (default: *.csv)")
    ap.add_argument("--examples", type=int, default=4,
                    help="examples_per_task for /import/task_docs_csv")
    ap.add_argument("--overwrite", action="store_true",
                    help="If set, overwrite existing tasks with same name")
    ap.add_argument("--no-skip-existing", action="store_true",
                    help="If set, skip_existing=false (otherwise true)")
    args = ap.parse_args()

    api = args.api.rstrip("/")
    glob_pattern = os.path.join(args.folder, args.pattern)
    paths = sorted(glob.glob(glob_pattern))
    if not paths:
        print(f"No CSVs matching: {glob_pattern}")
        return

    skip_existing = not args.no_skip_existing
    print(f"API: {api}")
    print(f"Found {len(paths)} CSVs in {args.folder}")
    print(f"examples_per_task={args.examples}, skip_existing={skip_existing}, overwrite={args.overwrite}")
    print()

    for p in paths:
        print(f"Uploading {p} ...")
        with open(p, "rb") as f:
            files = {"file": (os.path.basename(p), f, "text/csv")}
            params = {
                "examples_per_task": str(args.examples),
                "overwrite": str(args.overwrite).lower(),
                "skip_existing": str(skip_existing).lower(),
            }
            r = requests.post(f"{api}/import/task_docs_csv", files=files, params=params)
        if not r.ok:
            print(f"  ❌ Error {r.status_code}: {r.text}")
        else:
            js = r.json()
            print(f"  ✅ Inserted {len(js.get('items', []))} task rows.")

if __name__ == "__main__":
    main()
