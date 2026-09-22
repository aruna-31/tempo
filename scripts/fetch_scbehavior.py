"""Downloads SCBehavior YOLO images+labels from GitHub LFS media URLs.

The repo's git objects are LFS pointers (git lfs pull silently no-ops in this
environment), but the https://media.githubusercontent.com/media/... endpoint
serves the real binaries. This script reads each pointer file, extracts the
OID, downloads the binary and writes it in place of the pointer.
"""
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

REPO = "CCNUZFW/SCBehavior"
BRANCH = "master"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "training_data", "scbehavior", "SCBehavior_YOLO"))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "training_data", "scbehavior_dl"))
MEDIA = f"https://media.githubusercontent.com/media/{REPO}/{BRANCH}/SCBehavior_YOLO/"

PTR_RE = re.compile(r"oid sha256:([0-9a-f]{64})")


def fetch(rel: str, dest: str, session: requests.Session, retries: int = 3) -> str:
    dest_path = os.path.join(OUT, dest)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 100:
        return "skip"
    url = MEDIA + rel.replace(" ", "%20")
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 200 and len(r.content) > 100:
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                with open(dest_path, "wb") as f:
                    f.write(r.content)
                return "ok"
            return f"http{r.status_code}"
        except Exception as e:
            if attempt == retries - 1:
                return f"err:{e}"
            time.sleep(1.5)
    return "err"


def convert_pointers(root_rel: str) -> int:
    """For already-cloned pointer files, read the OID and download via media URL."""
    base = os.path.join(os.path.dirname(OUT), "scbehavior", "SCBehavior_YOLO")
    n_ok = 0
    session = requests.Session()
    tasks = []
    for dirpath, _dirs, files in os.walk(base):
        for fn in files:
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, base).replace("\\", "/")
            if os.path.getsize(p) > 500:
                continue  # real file
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    head = f.read(200)
            except OSError:
                continue
            if "git-lfs" not in head:
                continue
            tasks.append(rel)
    print(f"pointers to fetch: {len(tasks)}")
    done = 0
    from requests.adapters import HTTPAdapter, Retry
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1)))
    with ThreadPoolExecutor(max_workers=12) as ex:
        for res in ex.map(lambda rel: fetch_lfs(rel, session), tasks):
            done += 1
            if res == "ok":
                n_ok += 1
            if done % 150 == 0:
                print(f"  {done}/{len(tasks)} ({n_ok} ok)")
    return n_ok


def fetch_lfs(rel: str, session: requests.Session) -> str:
    base = os.path.join(os.path.dirname(OUT), "scbehavior", "SCBehavior_YOLO")
    p = os.path.join(base, rel.replace("/", os.sep))
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        m = PTR_RE.search(f.read())
    if not m:
        return "noptr"
    dest_path = os.path.join(OUT, rel.replace("/", os.sep))
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 100:
        return "skip"
    url = MEDIA + rel
    try:
        r = session.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0 TEMPO-dataset-fetch"})
        if r.status_code == 200 and len(r.content) > 100:
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(r.content)
            return "ok"
        return f"http{r.status_code}"
    except Exception as e:
        return f"err:{e}"


def main() -> int:
    n = convert_pointers("")
    print(f"downloaded {n} files into {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
