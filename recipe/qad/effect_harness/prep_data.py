# Effect-lane: real-text QAD corpus from a ModelScope dataset slice.
#
# Source: AI-ModelScope/alpaca-gpt4-data-en (Alpaca-GPT4 English instruct corpus, 52k samples).
# Deterministic slice: native file order, fixed index windows — train and holdout index ranges
# are disjoint (holdout comes from a far tail window of the same file, never overlapping train).
import argparse
import json
from pathlib import Path

from modelscope.msdatasets import MsDataset


SOURCE = "AI-ModelScope/alpaca-gpt4-data-en"
TRAIN_LO, TRAIN_HI = 0, 6000  # deterministic train window (native order)
HOLD_LO, HOLD_HI = 50000, 50060  # disjoint tail window: holdout ids never seen in training


def to_messages(row: dict) -> dict | None:
    instruction = (row.get("instruction") or "").strip()
    inp = (row.get("input") or "").strip()
    inp = "" if inp.lower() == "none" else inp
    output = (row.get("output") or "").strip()
    if not instruction or not output:
        return None
    user = f"{instruction}\n\n{inp}" if inp else instruction
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": output}]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="local Qwen3-0.6B dir (tokenizer)")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--max-tokens", type=int, default=380, help="drop samples longer than this")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_dir)

    ds = MsDataset.load(SOURCE, split="train")
    rows = list(ds)
    print(f"[data] loaded {len(rows)} rows from {SOURCE}")

    def slice_convert(lo: int, hi: int) -> list[dict]:
        out = []
        for i in range(lo, min(hi, len(rows))):
            m = to_messages(rows[i])
            if m is None:
                continue
            tokenized = tok.apply_chat_template(m["messages"], tokenize=True)
            ids = tokenized["input_ids"] if "input_ids" in tokenized else tokenized
            if ids and isinstance(ids[0], list):  # batched encoding: take the first row
                ids = ids[0]
            if len(ids) > args.max_tokens or len(ids) < 12:
                continue
            m["num_tokens"] = len(ids)
            out.append(m)
        return out

    train = slice_convert(TRAIN_LO, TRAIN_HI)
    hold = slice_convert(HOLD_LO, HOLD_HI)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "effect_train.jsonl"
    hold_path = args.out_dir / "effect_holdout.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        for m in train:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    with hold_path.open("w", encoding="utf-8") as f:
        for m in hold:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    info = {
        "source": SOURCE,
        "source_url": f"https://modelscope.cn/datasets/{SOURCE}",
        "train_window": [TRAIN_LO, TRAIN_HI],
        "holdout_window": [HOLD_LO, HOLD_HI],
        "index_ranges_disjoint": HOLD_LO >= TRAIN_HI,
        "max_tokens_filter": args.max_tokens,
        "num_train": len(train),
        "num_holdout": len(hold),
        "train_tokens": sum(m["num_tokens"] for m in train),
        "holdout_tokens": sum(m["num_tokens"] for m in hold),
        "deterministic": "native dataset file order; fixed index windows; no RNG",
    }
    (args.out_dir / "effect_corpus_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
