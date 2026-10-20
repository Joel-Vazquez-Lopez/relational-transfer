# Exact raw-output reconstruction

All1,733 original NPZ files are retained locally under payloads/. To reduce duplicate storage,the public archive stores each unique compressed-array member once while retaining exact original local ZIP headers,footers and other byte ranges in RAW_ARCHIVE_MANIFEST.json. It does not reserialize floats or change a single original payload byte.

The manifest describes every original path,byte count,SHA256 and ordered reconstruction pieces. Large pieces are content-addressed members in RAW_MEMBERS_*.tar;small pieces are base64 literals. Each bundle and every content chunk has a SHA256. All original files were reconstructed and byte-compared during archival,and every bundled chunk was independently read and hash-verified.

Using only Python's standard library,from this directory:

```sh
python3 restore_payloads.py --destination /path/to/new/output
```

This refuses to overwrite existing files and verifies bundle,chunk and complete reconstructed-file hashes. The regenerated payloads/CALL_*.npz paths match FORWARDS.jsonl exactly. The five-file smoke restoration result is saved in RESTORE_SMOKE.json. Full scientific replay also needs the separately archived native Llama screen,available at the parent study path/commit.

Original bytes:1,523,132,886. Deduplicated tar bundle bytes:954,060,800,plus the small reconstruction manifest. No model weights are included. These are complete raw outputs for the bounded study,not a selected collection of successful cases.
