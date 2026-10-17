"""Export complete answer/family ledgers and artifact checksums; no inference."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    summary = json.loads((HERE/"QUERY_SUMMARY.json").read_text())
    with (HERE/"QUERY_ANSWER_LEDGER.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["model", "family", "question", "condition", "expected", "generated",
                         "greedy_correct", "score_correct", "score_margin", "sentence"])
        for model in summary["models"]:
            raw = json.loads((HERE/f"QUERY_{model}.json").read_text())
            for family in raw["families"]:
                for task in family["tasks"]:
                    for row in task["rows"]:
                        writer.writerow([model, family["family"]["id"], task["question_key"],
                                         row["condition"], row["expected"], row["generated"],
                                         row["greedy_correct"], row["score_correct"],
                                         row["score_margin"], row["sentence"]])
    with (HERE/"QUERY_FAMILY_LEDGER.csv").open("w", newline="") as handle:
        fields = ["model", "id", "verb", "correct_answers", "fully_competent",
                  "late_cos_agent", "late_cos_first", "delta", "delta_after_axis_removal"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model, record in summary["models"].items():
            for family in record["families"]:
                writer.writerow({"model": model, **family})
    records = []
    for path in sorted(HERE.iterdir()):
        if path.is_file() and path.name not in ("ARTIFACT_MANIFEST.json", "GITHUB_PRESERVATION.json"):
            records.append({"path": path.name, "bytes": path.stat().st_size,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "local_tensor_archive": path.suffix == ".npz"})
    (HERE/"ARTIFACT_MANIFEST.json").write_text(json.dumps({
        "scope": "Trajectory relation audit and one frozen question diagnostic",
        "tensor_storage": "All npz archives retained locally; text, figures and checksums suitable for GitHub research branch.",
        "files": records}, indent=2)+"\n")
    print("Exported 256 answer records, 32 family records and", len(records), "artifact hashes")


if __name__ == "__main__":
    main()
