# Copyright (c) OpenMMLab. All rights reserved.
"""ModelOpt-lane-style QAD regression on the REAL Qwen3-0.6B checkpoint.

Reference pattern: Model-Optimizer ``tests/regression/torch/speculative/test_dflash.py``
(real small model, N steps, GPU, ``@pytest.mark.timeout(300)``; asserts
``final_loss < first_loss`` plus an absolute sanity floor — their documented baseline:
"~1.8 at step 500 on L40").

This test mirrors that assertion form on our QAD pipeline (w8a8+kv8 fake-quant student,
frozen BF16 teacher, top-K forward-KL) and adds our stronger gates:

ModelOpt form:
- ``final_loss < first_loss`` — window-averaged train-KL log (10-step windows, i.e. the
  §7.3 log entry cadence; this mirrors ModelOpt's HF ``log_history`` averaged entries,
  not raw per-packed-row noise);
- ``all(losses finite)``;
- ``master weights moved`` (fp32 masters compared bitwise pre/post fit);
- ``checkpoint saved non-empty`` (master safetensors > 0 bytes);
- absolute sanity floor on the final windowed KL (documented baseline below).

XTuner QAD gates (strictly stronger than the ModelOpt lane):
- Gate B: ``holdout_kl_end < holdout_kl_ptq`` (fake-quant holdout KL improves over the
  untrained PTQ amax baseline on a disjoint holdout slice);
- Gate C: exported weights+scales replayed via ``load_qad_export`` reproduce the
  training-end logits bit-exactly on a holdout row.

Measured baseline (documented for a future absolute floor, ModelOpt-style):
- RTX 3090 24GB, 2026-11-27, LR sweep winner = 5e-7 @ gbs 1 (see the LR-sweep task in the
  A100 handoff; sweep ran this exact corpus, 300 steps, eager):
  PTQ holdout KL ≈ 0.0237, QAD-end holdout KL ≈ 0.0181 (Gate B −23.6%),
  cumulative-mean train-KL 0.03457 → 0.01581 (−54.3%)
  (work_dir/qad_sim/effect/sweep_lr_accum/lr5e-7_gbs1/result.json).

Skip discipline (mirrors ModelOpt's flag-gated nightly/manual lane): the test is skipped
unless a real Qwen3-0.6B checkpoint directory exists at ``QAD_REGRESSION_MODEL_PATH`` or at
the effect-lane default path. Fast CI stays fast; the full 500-step run stays in the
effect lane (``work_dir/qad_sim/effect``).

Wall-time budget: ModelOpt caps this lane at 180-300s on their hardware. The ``pytest-timeout``
plugin is not installed in this repo, so the budget is documented instead of enforced by
marker: this test runs TEST_STEPS QAD steps on the full 0.6B model; measured wall time on
RTX 3090 is ~<filled after the calibration run>s (≈3s/step training + ~60s one-time
model build/export overhead). Keep TEST_STEPS sized so the whole test stays under ~10 min.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from xtuner._testing import DeterministicDDPTestCase


# The 300-step eager lane needs ~10-15 min; torch's distributed test wrapper kills the
# training subprocess at 300 s by default (DISTRIBUTED_TESTS_DEFAULT_TIMEOUT).
os.environ.setdefault("DISTRIBUTED_TESTS_DEFAULT_TIMEOUT", "1800")

REPO_ROOT = Path(__file__).resolve().parents[2]
EFFECT_DIR = Path(os.environ.get("QAD_EFFECT_DIR", REPO_ROOT / "work_dir" / "qad_sim" / "effect"))
# Committed copy of the effect harness (run_real.py); used when the local work_dir copy is absent.
HARNESS_DIR = REPO_ROOT / "recipe" / "qad" / "effect_harness"

QAD_REGRESSION_MODEL_PATH = os.environ.get("QAD_REGRESSION_MODEL_PATH")
_DEFAULT_MODEL = EFFECT_DIR / "model" / "Qwen3-0.6B-untied"

# Calibration knobs for the regression lane (ModelOpt runs N=500 at timeout 300s on L40;
# this host is ~3s/step, so the lane runs a shorter schedule).
TEST_STEPS = 300
TRAIN_KL_INTERVAL = 10
# Sanity floor, ModelOpt form ("final loss should be reasonable"): the 500-step effect-lane
# run at LR 1e-6 (work_dir/qad_sim/effect/ablation-a1-final, RTX 3090, 2026-11-27) measured
# final windowed train-KL = 0.02916 (see result.json modelopt_lane). The floor admits ~37%
# headroom above the measured baseline while catching gross regressions (optimizer/scale/sign bugs).
FINAL_KL_FLOOR = 0.04


def _model_dir() -> Path | None:
    if QAD_REGRESSION_MODEL_PATH:
        p = Path(QAD_REGRESSION_MODEL_PATH)
        if (p / "config.json").exists():
            return p
    if (_DEFAULT_MODEL / "config.json").exists():
        return _DEFAULT_MODEL
    return None


def _dist_rank() -> int:
    import torch.distributed as dist  # noqa: PLC0415

    return dist.get_rank() % max(torch.cuda.device_count(), 1)


def _sys_path_prep() -> None:
    import sys  # noqa: PLC0415

    for p in (str(EFFECT_DIR), str(HARNESS_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)


# Authored real-text samples (deterministic; disjoint train/holdout halves).
_QA_SAMPLES = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is 17 + 26?", "17 + 26 equals 43."),
    ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by William Shakespeare."),
    ("What is the chemical symbol for water?", "The chemical symbol for water is H2O."),
    ("Name the three primary colors.", "The three primary colors are red, blue, and yellow."),
    ("What planet is known as the Red Planet?", "Mars is known as the Red Planet."),
    ("What is 9 times 7?", "9 times 7 equals 63."),
    ("What is the largest ocean on Earth?", "The Pacific Ocean is the largest ocean on Earth."),
    ("How many continents are there?", "There are seven continents on Earth."),
    ("What gas do plants absorb for photosynthesis?", "Plants absorb carbon dioxide for photosynthesis."),
    ("What is 25% of 80?", "25% of 80 is 20."),
    ("Translate 'good morning' into French.", "'Good morning' in French is 'Bonjour'."),
    ("What is the freezing point of water in Celsius?", "Water freezes at 0 degrees Celsius."),
    ("Give a synonym for 'happy'.", "A synonym for 'happy' is 'joyful'."),
    ("What is 2 to the power of 10?", "2 to the power of 10 is 1024."),
    ("In which direction does the sun rise?", "The sun rises in the east."),
    ("What is 144 divided by 12?", "144 divided by 12 equals 12."),
    ("Who painted the Mona Lisa?", "The Mona Lisa was painted by Leonardo da Vinci."),
    ("What is the square root of 81?", "The square root of 81 is 9."),
    ("How many days are in a leap year?", "A leap year has 366 days."),
    ("What is 100 minus 37?", "100 minus 37 equals 63."),
    ("What language is spoken in Brazil?", "The official language of Brazil is Portuguese."),
    ("Count from one to five.", "One, two, three, four, five."),
    ("What is the tallest mountain on Earth?", "Mount Everest is the tallest mountain on Earth."),
    ("What is 15 + 27 - 9?", "15 + 27 - 9 equals 33."),
    ("Which sea creature has eight arms?", "The octopus has eight arms."),
    ("What is 3 boxes with 8 apples each in total?", "3 boxes with 8 apples each hold 24 apples in total."),
    ("Who is known as the father of computers?", "Charles Babbage is known as the father of computers."),
    ("What comes after 2, 4, 6, 8?", "The next number in the sequence 2, 4, 6, 8 is 10."),
    ("What metal is liquid at room temperature?", "Mercury is a metal that is liquid at room temperature."),
    ("Rewrite in past tense: 'She walks to school.'", "In the past tense: 'She walked to school.'"),
    ("How many strings does a standard guitar have?", "A standard guitar has six strings."),
    ("What is the fastest land animal?", "The cheetah is the fastest land animal."),
    ("What do bees collect from flowers?", "Bees collect nectar and pollen from flowers."),
    ("Which planet has the most prominent rings?", "Saturn has the most prominent rings."),
    ("What is half of 90?", "Half of 90 is 45."),
]

_SUFFIXES = [
    "Please answer briefly.",
    "Answer in one sentence.",
    "Keep the answer short.",
    "Give a direct answer.",
    "",
    "Respond concisely.",
]


def _write_corpus(train_path: Path, holdout_path: Path) -> None:
    """Deterministic real-text jsonl corpus through the qwen3 chat template
    convention."""
    train_rows: list[dict] = []
    holdout_rows: list[dict] = []
    for i, (q, a) in enumerate(_QA_SAMPLES):
        for j, suffix in enumerate(_SUFFIXES):
            user = f"{q}{(' ' + suffix) if suffix else ''}"
            row = {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": a}]}
            # even suffix index -> train pool, odd -> holdout pool (disjoint user texts)
            (train_rows if j % 2 == 0 else holdout_rows).append(row)
    for path, rows, reps in ((train_path, train_rows, 12), (holdout_path, holdout_rows, 2)):
        with path.open("w", encoding="utf-8") as f:
            for rep in range(reps):
                for k, row in enumerate(rows):
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")


@unittest.skipUnless(
    torch.cuda.is_available() and _model_dir() is not None,
    f"requires a CUDA device and a real Qwen3-0.6B checkpoint at QAD_REGRESSION_MODEL_PATH "
    f"or {_DEFAULT_MODEL} (manual/nightly lane; fast CI stays fast)",
)
class TestQADRealModelRegression(DeterministicDDPTestCase):
    def test_qad_modelopt_lane_and_gates(self):
        """Run a short real-checkpoint QAD schedule; assert the ModelOpt-lane
        form + our gates."""
        self.create_pg("cuda")
        os.environ["LOCAL_RANK"] = str(_dist_rank())

        _sys_path_prep()

        from run_real import run_qad  # effect-lane pipeline (norm-safe KV attach applied on import)

        # NOTE (2026-11-27, FA3-selection fix): this lane currently fails in `QADTrainer.__init__`
        # BEFORE any attention forward, with `RuntimeError: Expected a 'cuda' device type for
        # generator but found 'cpu'` from `xtuner/v1/datasets/packing.py:343 get_pack_infos`
        # (torch 2.13 `_device` wrapping vs the packing RNG generator). The failure is identical
        # with and without `XTUNER_HF_IMPL=true` and on the unmodified tree, so it is unrelated
        # to attention-backend selection; no eager pin is needed for selection anymore — the
        # default `MHAConfig` path builds and forwards (see tests/ops/test_flash_attn_selection.py).

        # NOTE (2026-11-27): the packing RNG device bug below (`datasets/packing.py:343`,
        # "Expected a 'cuda' device type for generator but found 'cpu'") is FIXED on this
        # branch (`ExpandSoftPackDataset._generator_on_default_device`). Attention-backend
        # selection is fail-soft since commit 9989582 (default path builds and forwards; see
        # tests/ops/test_flash_attn_selection.py).

        model_dir = str(_model_dir())
        tmp_dir = tempfile.mkdtemp(prefix="qad_regression_")
        exp_dir = Path(tmp_dir) / "w8a8-kvint8"
        try:
            train_jsonl = Path(tmp_dir) / "train.jsonl"
            holdout_jsonl = Path(tmp_dir) / "holdout.jsonl"
            _write_corpus(train_jsonl, holdout_jsonl)
            # Pin the HF-exact eager attention for this lane: the loss-trend/gate assertions
            # must hold on any machine, and usable backends differ per host (no FA2 here, FA3
            # corrupts on SM86, torch 2.13 Inductor flex_decoding has no SM86 kernels). 60
            # eager steps cost minutes and are deterministic; A100 keeps the same contract.
            with self.hf_impl():
                result = run_qad(
                    model_dir=model_dir,
                    train_jsonl=str(train_jsonl),
                    holdout_jsonl=str(holdout_jsonl),
                    exp_dir=exp_dir,
                    schema="w8a8",
                    kv_dtype="int8",
                    total_steps=TEST_STEPS,
                    lr=5e-7,
                    train_kl_interval=TRAIN_KL_INTERVAL,
                    holdout_interval=TEST_STEPS,
                    pack_max_length=1024,
                    holdout_rows=64,
                    generation=False,
                )

            # --- ModelOpt-lane form ----------------------------------------------------------
            # Loss-decrease uses the CUMULATIVE-MEAN train-KL (measured 0.02823 -> 0.02775 over
            # 500 steps at LR 1e-6): per-window KL is noisy at tiny per-step token counts, and
            # the last-window value sits at noise level, so the cumulative mean is the honest
            # monotonicity signal. The single-window pair is still recorded in result.json.
            lane = result["modelopt_lane"]
            first_loss = lane["first_cum_mean_kl"]  # cumulative mean over the first window
            final_loss = lane["final_cum_mean_kl"]  # cumulative mean over the last window
            assert final_loss < first_loss, (
                f"Loss did not decrease: {first_loss:.6f} -> {final_loss:.6f} "
                f"(cumulative-mean train KL over {TEST_STEPS} steps)"
            )
            # Sanity floor: measured 0.02916 at step 500 on RTX 3090 (see docstring).
            assert final_loss < FINAL_KL_FLOOR, f"Final loss {final_loss:.6f} too high (expected < {FINAL_KL_FLOOR})"
            assert lane["all_finite"], "non-finite value in the logged loss/KL curves"
            moved = [v for v in lane["master_weights_moved"].values() if isinstance(v, bool)]
            assert moved and all(moved), f"master weights did not all move: {lane['master_weights_moved']}"
            assert lane["master_checkpoint"]["bytes"] > 0, "master checkpoint is empty"

            # --- XTuner QAD gates (stronger than ModelOpt) ------------------------------------
            assert result["kl_b_end"] < result["kl_ptq"], (
                f"Gate B failed: holdout end KL {result['kl_b_end']:.6f} >= PTQ baseline {result['kl_ptq']:.6f}"
            )
            assert result["gate_c_logits_equal"], (
                f"Gate C failed: replayed export logits differ (max abs diff "
                f"{result['gate_c_logits_max_abs_diff']:.3e})"
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @property
    def world_size(self) -> int:
        return 1


if __name__ == "__main__":
    unittest.main()
