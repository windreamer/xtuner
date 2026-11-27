# Effect-lane: prepare a working copy of Qwen3-0.6B with tie_word_embeddings=false.
#
# The ModelScope checkpoint ships BOTH lm_head.weight and model.embed_tokens.weight, bitwise
# equal (verified below — the shipped weights are exactly the tied pair). The trainer's strict
# checkpoint load rejects the paired key set against a tied config ("Unexpected keys:
# {'lm_head.weight'}"), and the T9 sim measured an FSDP2 shared-parameter rejection on the tied
# wrap path, so the effect lane trains UNTIED copies of the two identical tensors. Init is
# numerically identical to the tied model; the deviation is that master lm_head/embed may drift
# apart during training (both are outside the QAD quantization face, §4.1). Recorded in
# report.md; teacher is frozen at the identical untied init, so the KL chain is unaffected.
import json
import shutil
from pathlib import Path


SRC = Path(__file__).parent / "model" / "Qwen3-0.6B"
DST = Path(__file__).parent / "model" / "Qwen3-0.6B-untied"


def main() -> None:
    import torch  # noqa: PLC0415
    from safetensors import safe_open  # noqa: PLC0415

    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)
    for f in SRC.iterdir():
        if f.is_file():
            shutil.copy2(f, DST / f.name)
    cfg = json.loads((DST / "config.json").read_text(encoding="utf-8"))
    assert cfg.get("tie_word_embeddings") is True, "upstream config expected tied"
    with safe_open(str(SRC / "model.safetensors"), framework="pt") as sf:
        lm = sf.get_tensor("lm_head.weight")
        em = sf.get_tensor("model.embed_tokens.weight")
    bitwise_equal = bool(torch.equal(lm, em))
    assert bitwise_equal, "shipped lm_head.weight != embed_tokens.weight; untie would change init"
    cfg["tie_word_embeddings"] = False
    (DST / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    info = {
        "source": str(SRC),
        "target": str(DST),
        "shipped_tied_config": True,
        "lm_head_embed_bitwise_equal": bitwise_equal,
        "untie_rationale": "strict-load rejects paired keys under tied config; FSDP2 tied-wrap "
        "rejection measured in T9 sim; untied init numerically identical (bitwise-equal pair)",
    }
    (DST / "untie_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
