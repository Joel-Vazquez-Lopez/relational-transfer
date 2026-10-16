#!/usr/bin/env python3
"""Source-integrity checker for unlabelled, open-development Wikipedia revisions.

This reads actual MediaWiki action=query revision payloads, not rendered current pages.
It does not annotate event roles, clear rights, or run any LLM. Source is saved locally.
"""
import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://en.wikipedia.org/w/api.php"

def validate_revision(row, payload):
    if not isinstance(payload, dict) or "query" not in payload:
        raise ValueError("API_MALFORMED")
    pages = payload["query"].get("pages", [])
    if not isinstance(pages, list) or len(pages) != 1:
        raise ValueError("PAGE_COUNT_MISMATCH")
    page = pages[0]
    revisions = page.get("revisions", [])
    if len(revisions) != 1:
        raise ValueError("REVISION_MISSING_OR_AMBIGUOUS")
    rev = revisions[0]
    if rev.get("revid") != row["oldid"]:
        raise ValueError("OLDID_MISMATCH")
    if page.get("title") != row["title"]:
        raise ValueError("TITLE_MISMATCH")
    if row.get("pageid") is not None and page.get("pageid") != row["pageid"]:
        raise ValueError("PAGEID_MISMATCH")
    content = rev.get("slots", {}).get("main", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("SOURCE_TEXT_MISSING")
    if not rev.get("sha1") or not rev.get("timestamp"):
        raise ValueError("REVISION_META_MISSING")
    raw = content.encode("utf-8")
    return {"candidate_id": row["candidate_id"], "title": page["title"],
            "oldid": rev["revid"], "pageid": page.get("pageid"),
            "timestamp": rev["timestamp"], "mediawiki_sha1": rev["sha1"],
            "wikitext_sha256": hashlib.sha256(raw).hexdigest(),
            "bytes_utf8": len(raw), "source_text": content}

def fetch_revision(oldid, user_agent):
    query = urllib.parse.urlencode({"action":"query","prop":"revisions",
        "revids":str(oldid),"rvprop":"ids|timestamp|sha1|content",
        "rvslots":"main","formatversion":"2","format":"json"})
    request = urllib.request.Request(API+"?"+query,
        headers={"User-Agent":user_agent,"Accept":"application/json"})
    with urllib.request.urlopen(request,timeout=30) as response:
        return json.loads(response.read())

def self_test():
    row = {"candidate_id":"FIXTURE","title":"Example","oldid":111,"pageid":100}
    payload = {"query":{"pages":[{"title":"Example","pageid":100,
        "revisions":[{"revid":111,"timestamp":"2026-10-16T00:00:00Z",
                      "sha1":"abc","slots":{"main":{"content":"Example sentence."}}}]}]}}
    assert validate_revision(row,payload)["bytes_utf8"] > 0
    import copy
    for path,bad,expected in [
        (("revid",),999,"OLDID_MISMATCH"),
        (("title",),"Other","TITLE_MISMATCH"),
        (("pageid",),101,"PAGEID_MISMATCH"),
        (("content",),"","SOURCE_TEXT_MISSING"),
    ]:
        p=copy.deepcopy(payload)
        page=p["query"]["pages"][0]
        if path[0]=="title": page["title"]=bad
        elif path[0]=="pageid": page["pageid"]=bad
        elif path[0]=="revid": page["revisions"][0]["revid"]=bad
        else: page["revisions"][0]["slots"]["main"]["content"]=bad
        try:
            validate_revision(row,p)
        except ValueError as e:
            assert str(e)==expected, (str(e),expected)
        else:
            raise AssertionError("Expected "+expected)
    print("SELF_TEST_PASS: correct fixture and 4 integrity rejections")

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--manifest",type=Path,default=Path(__file__).with_name("candidate_articles.jsonl"))
    a.add_argument("--out",type=Path,default=Path("local_wiki_revision_cache"))
    a.add_argument("--user-agent",help="Real contactable research User-Agent (mandatory for live fetch)")
    a.add_argument("--self-test",action="store_true")
    args=a.parse_args()
    if args.self_test:
        self_test();return
    if not args.user_agent:
        a.error("Provide --user-agent with truthful research identity/contact")
    rows=[json.loads(s) for s in args.manifest.read_text(encoding="utf-8").splitlines() if s.strip()]
    assert len(set(r["oldid"] for r in rows)) == len(rows)
    assert len(set(r["candidate_id"] for r in rows)) == len(rows)
    args.out.mkdir(parents=True,exist_ok=True)
    for i,row in enumerate(rows):
        item=validate_revision(row,fetch_revision(row["oldid"],args.user_agent))
        text=item.pop("source_text")
        (args.out/(row["candidate_id"]+".wikitext")).write_text(text,encoding="utf-8")
        (args.out/(row["candidate_id"]+".source.json")).write_text(json.dumps(item,indent=2)+"\n",encoding="utf-8")
        print(json.dumps({"id":row["candidate_id"],"status":"EXACT_REVISION_VERIFIED","sha256":item["wikitext_sha256"]}))
        if i<len(rows)-1:time.sleep(1)
    print("Source reference check completed; passage licensing and human semantic gold STILL PENDING")

if __name__=="__main__":
    main()
