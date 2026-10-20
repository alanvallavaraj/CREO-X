#!/usr/bin/env python3
"""Create a DRAFT Zenodo deposit for the CREO-X results bundle.

Creates a draft and stops. It does NOT publish: publishing mints a permanent DOI
and is a decision for the author. Review the draft in the browser, then press
Publish there (or set PUBLISH=1 deliberately).

Needs a Zenodo personal access token with scopes `deposit:write` and
`deposit:actions`:
    Zenodo -> your account -> Applications -> Personal access tokens -> New token

Usage:
    export ZENODO_TOKEN=...            # never commit this
    python zenodo_deposit.py creo_x_results_v1.zip .zenodo.json
"""
import json, os, sys, urllib.request, urllib.error

API = "https://zenodo.org/api"


def call(method, url, token, data=None, headers=None, raw=False):
    h = {"Authorization": f"Bearer {token}"}
    if not raw:
        h["Content-Type"] = "application/json"
    h.update(headers or {})
    body = data if raw else (json.dumps(data).encode() if data is not None else None)
    req = urllib.request.Request(url, data=body, method=method, headers=h)
    try:
        with urllib.request.urlopen(req) as r:
            txt = r.read().decode()
            return json.loads(txt) if txt else {}
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code} on {method} {url}\n{e.read().decode()[:600]}")


def main():
    zip_path = sys.argv[1] if len(sys.argv) > 1 else "creo_x_results_v1.zip"
    meta_path = sys.argv[2] if len(sys.argv) > 2 else ".zenodo.json"
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("ZENODO_TOKEN is not set.")
    for p in (zip_path, meta_path):
        if not os.path.exists(p):
            sys.exit(f"missing input: {p}")

    meta = json.load(open(meta_path))
    # .zenodo.json uses the repository-integration schema; the REST API wants it
    # nested under "metadata", and it rejects unknown keys, so map explicitly.
    md = {
        "title": meta["title"],
        "upload_type": meta.get("upload_type", "dataset"),
        "description": meta["description"],
        "creators": meta["creators"],
        "license": meta.get("license", "cc-by-4.0"),
        "keywords": meta.get("keywords", []),
        "access_right": "open",
    }
    if meta.get("related_identifiers"):
        md["related_identifiers"] = meta["related_identifiers"]

    print("creating draft deposit ...")
    dep = call("POST", f"{API}/deposit/depositions", token, {})
    dep_id = dep["id"]
    print(f"  draft id {dep_id}")

    print(f"uploading {zip_path} ({os.path.getsize(zip_path)/1024:.0f} KB) ...")
    bucket = dep["links"]["bucket"]
    with open(zip_path, "rb") as fh:
        call("PUT", f"{bucket}/{os.path.basename(zip_path)}", token,
             data=fh.read(), headers={"Content-Type": "application/octet-stream"},
             raw=True)

    print("applying metadata ...")
    dep = call("PUT", f"{API}/deposit/depositions/{dep_id}", token, {"metadata": md})

    print("\nDRAFT CREATED, NOT PUBLISHED")
    print("  review :", dep["links"].get("html"))
    print("  reserved DOI:", (dep.get("metadata", {}).get("prereserve_doi") or {}).get("doi", "assigned on publish"))
    print("\nPublish from the browser once you are satisfied. Publishing is permanent.")

    if os.environ.get("PUBLISH") == "1":
        print("\nPUBLISH=1 set; publishing ...")
        out = call("POST", f"{API}/deposit/depositions/{dep_id}/actions/publish", token)
        print("  published DOI:", out.get("doi"))


if __name__ == "__main__":
    main()
