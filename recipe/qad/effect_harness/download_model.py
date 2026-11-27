# Effect-lane: download the REAL Qwen3-0.6B checkpoint from ModelScope and verify
# config + tokenizer load in the venv. Artifacts under work_dir/qad_sim/effect/.
import json
import sys
import time
from pathlib import Path

from modelscope import snapshot_download


OUT_DIR = Path(__file__).parent
TARGET = OUT_DIR / "model" / "Qwen3-0.6B"
CANDIDATE_IDS = ("Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-0.5B-Instruct")


def main() -> None:
    info: dict = {"target": str(TARGET)}
    if (TARGET / "config.json").exists() and any(TARGET.glob("*.safetensors")):
        info["status"] = "already-downloaded"
    else:
        last_err = None
        for repo_id in CANDIDATE_IDS:
            t0 = time.time()
            try:
                path = snapshot_download(repo_id, local_dir=str(TARGET))
                info.update({"repo_id": repo_id, "snapshot_path": path, "download_s": round(time.time() - t0, 1)})
                info["status"] = "downloaded"
                break
            except Exception as exc:  # try the next candidate id
                last_err = f"{repo_id}: {type(exc).__name__}: {exc}"
                print(f"[download] FAILED {last_err}", flush=True)
        if info.get("status") != "downloaded":
            info.update({"status": "failed", "error": last_err})
            (OUT_DIR / "download_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
            sys.exit(2)

    # Verify config + tokenizer load through the venv stack.
    from transformers import AutoConfig, AutoTokenizer

    cfg = AutoConfig.from_pretrained(str(TARGET))
    tok = AutoTokenizer.from_pretrained(str(TARGET))
    info.update(
        {
            "model_type": cfg.model_type,
            "vocab_size": cfg.vocab_size,
            "num_hidden_layers": cfg.num_hidden_layers,
            "hidden_size": cfg.hidden_size,
            "tie_word_embeddings": bool(getattr(cfg, "tie_word_embeddings", False)),
            "qk_norm_present": "q_norm"
            in json.loads((TARGET / "model.safetensors.index.json").read_text())["weight_map"]
            if (TARGET / "model.safetensors.index.json").exists()
            else any(TARGET.glob("*.safetensors")),
            "tokenizer_vocab": len(tok),
            "pad_token_id": tok.pad_token_id,
            "eos_token_id": tok.eos_token_id,
            "chat_template_builtin": tok.chat_template is not None,
            "total_bytes": sum(f.stat().st_size for f in TARGET.rglob("*") if f.is_file()),
        }
    )
    (OUT_DIR / "download_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
