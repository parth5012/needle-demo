"""Score the finetuned MAIN model against eval_cases.json (exact-match per field).

Local (venv backend):  .venv/Scripts/python.exe finetune/eval.py [--weights <path>] [--limit N]
Colab (after cells 1-4): !python /content/needle-demo/backend/finetune/eval.py --weights /content/needle-demo/backend/checkpoints/needle3-artisan.cact

Needs: cactus-needle (+ pydantic). Uses the full app engine (validators
included) so the score reflects production behavior. Default weights:
$NEEDLE_MAIN_WEIGHTS, backend/checkpoints/needle3-artisan.cact, else base.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from app.needle_engine import Needle3InferenceEngine  # noqa: E402
from app.schemas import ExtractionRequest  # noqa: E402

FIELDS = ("name", "phone", "pin", "pehchan_id", "trifed_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None,
                    help=".cact path (sets NEEDLE_MAIN_WEIGHTS for this run)")
    ap.add_argument("--limit", type=int, default=0, help="run first N cases only")
    ap.add_argument("--cases", default=os.path.join(HERE, "eval_cases.json"))
    args = ap.parse_args()
    if args.weights:
        os.environ["NEEDLE_MAIN_WEIGHTS"] = args.weights

    with open(args.cases, encoding="utf-8") as f:
        cases = json.load(f)
    if args.limit:
        cases = cases[:args.limit]

    engine = Needle3InferenceEngine()
    per_field = {k: [0, 0] for k in FIELDS}
    perfect, total = 0, 0
    failures = []
    for case in cases:
        total += 1
        res = engine.extract(ExtractionRequest(text=case["text"]))
        got = {k: getattr(res.form_data, k) for k in FIELDS}
        want = {k: case["expected"].get(k) for k in FIELDS}
        ok_all = True
        for k in FIELDS:
            per_field[k][1] += 1
            if got[k] == want[k]:
                per_field[k][0] += 1
            else:
                ok_all = False
        if ok_all:
            perfect += 1
        else:
            bad = {k: (want[k], got[k]) for k in FIELDS if want[k] != got[k]}
            failures.append((case["id"], bad))

    print(f"\ncases 5/5 perfect: {perfect}/{total}")
    for k in FIELDS:
        c, n = per_field[k]
        print(f"  {k:<11} {c}/{n} = {c / n:.0%}")
    if failures:
        print("\nfailed cases (expected -> got):")
        for cid, bad in failures:
            print(f"  - {cid}")
            for k, (w, g) in bad.items():
                print(f"      {k}: {w!r} -> {g!r}")
    else:
        print("\nALL CASES 5/5 PERFECT")


if __name__ == "__main__":
    main()
