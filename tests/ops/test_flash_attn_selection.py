"""Flash-attention selection must probe real callables and fail soft, never
resolve to None.

Regression coverage for the crash chain where an installed `flash-attn-3` dist made ``MHAConfig.model_post_init`` keep
``attn_impl="flash_attention"`` while flash-attn 2 was absent, so the resolved op callable was ``None`` and every
default-config forward crashed with ``TypeError: 'NoneType' object is not callable``.
"""

import pytest
import torch

import xtuner.v1.module.attention.mha as mha_module
import xtuner.v1.ops.flash_attn as flash_attn_ops
from xtuner.v1.module.attention.mha import MHAConfig, _flex_attention_available
from xtuner.v1.ops import get_attn_impl_fn
from xtuner.v1.ops.flash_attn import flash_attn_available, get_flash_attn_varlen


def _mha_cfg() -> MHAConfig:
    return MHAConfig(num_attention_heads=4, num_key_value_heads=2, head_dim=16)


class TestFlashAttnSelection:
    def test_default_config_fails_soft_without_fa2(self, monkeypatch):
        # The reported bug: FA2 callables absent + flash-attn-3 installed must not keep the
        # flash selection. The device is forced to cuda so the probes are exercised even on
        # CPU-only CI; no CUDA tensors are touched.
        monkeypatch.setattr(flash_attn_ops, "get_device", lambda: "cuda")
        monkeypatch.setattr(flash_attn_ops, "_fa2_available", lambda: False)
        monkeypatch.delenv("XTUNER_USE_FA3", raising=False)
        monkeypatch.setattr(mha_module, "get_device", lambda: "cuda")
        monkeypatch.setattr(mha_module, "flash_attn_available", lambda: False)

        assert _mha_cfg().attn_impl == "flex_attention"

    def test_available_callables_keep_flash_selection(self, monkeypatch):
        monkeypatch.setattr(mha_module, "get_device", lambda: "cuda")
        monkeypatch.setattr(mha_module, "flash_attn_available", lambda: True)

        assert _mha_cfg().attn_impl == "flash_attention"

    def test_unavailable_flex_downgrades_to_eager(self, monkeypatch):
        monkeypatch.setattr(mha_module, "get_device", lambda: "cuda")
        monkeypatch.setattr(mha_module, "flash_attn_available", lambda: False)
        monkeypatch.setattr(mha_module, "_flex_attention_available", lambda: False)

        assert _mha_cfg().attn_impl == "eager_attention"

    def test_default_path_returns_stub_when_fa2_absent(self, monkeypatch):
        monkeypatch.setattr(flash_attn_ops, "get_device", lambda: "cuda")
        monkeypatch.setattr(flash_attn_ops, "_fa2_available", lambda: False)
        monkeypatch.delenv("XTUNER_USE_FA3", raising=False)

        backend = get_flash_attn_varlen()
        assert backend is flash_attn_ops._unavailable_flash_attn
        assert not flash_attn_available()
        with pytest.raises(RuntimeError, match="flex_attention"):
            backend()

    def test_fa3_flag_without_entry_fails_fast_with_guidance(self, monkeypatch):
        monkeypatch.setattr(flash_attn_ops, "get_device", lambda: "cuda")
        monkeypatch.setattr(flash_attn_ops, "_fa3_entry_available", lambda: False)
        monkeypatch.setenv("XTUNER_USE_FA3", "1")

        with pytest.raises(RuntimeError, match="XTUNER_USE_FA3"):
            get_flash_attn_varlen()
        # Explicit FA3 requests surface as unavailable to model construction, not as a crash.
        assert not flash_attn_available()

    def test_fa3_flag_on_pre_hopper_degrades_to_fa2(self, monkeypatch):
        monkeypatch.setattr(flash_attn_ops, "get_device", lambda: "cuda")
        monkeypatch.setattr(flash_attn_ops, "_fa3_entry_available", lambda: True)
        monkeypatch.setattr(flash_attn_ops, "_fa3_arch_supported", lambda: False)
        monkeypatch.setattr(flash_attn_ops, "_fa2_available", lambda: True)
        sentinel = lambda *args, **kwargs: None  # noqa: E731
        monkeypatch.setattr("xtuner.v1.ops.flash_attn.gpu.flash_attn_varlen_func_v2", sentinel)
        monkeypatch.setenv("XTUNER_USE_FA3", "1")

        # FA3 dist present but kernels unusable on this arch: warn and take the FA2 backend.
        assert get_flash_attn_varlen() is sentinel

    def test_fa3_flag_on_pre_hopper_without_fa2_returns_stub(self, monkeypatch):
        monkeypatch.setattr(flash_attn_ops, "get_device", lambda: "cuda")
        monkeypatch.setattr(flash_attn_ops, "_fa3_entry_available", lambda: True)
        monkeypatch.setattr(flash_attn_ops, "_fa3_arch_supported", lambda: False)
        monkeypatch.setattr(flash_attn_ops, "_fa2_available", lambda: False)
        monkeypatch.setenv("XTUNER_USE_FA3", "1")

        assert get_flash_attn_varlen() is flash_attn_ops._unavailable_flash_attn
        assert not flash_attn_available()

    def test_forced_flash_op_reports_missing_backend_clearly(self):
        # Without FA2 the mapping-selected flash op must raise the descriptive stub error,
        # not the historical TypeError from a None callable. Skip on machines with FA2.
        if flash_attn_available():
            pytest.skip("FA2 callables available; flash op is functional on this machine")

        flash_op = get_attn_impl_fn("flash_attention")
        q = torch.randn(1, 2, 8, 8, dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="FlashAttention callables are unavailable"):
            flash_op(q, q, q, cu_seqlens_q=torch.tensor([0, 8]))

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_default_config_forward_runs_on_cuda(self):
        # The exact reported scenario, unmocked: this machine has flash-attn-3 installed and
        # no flash-attn 2, so the default config must downgrade and the selected op must run.
        cfg = MHAConfig(num_attention_heads=4, num_key_value_heads=2, head_dim=16)
        if flash_attn_available():
            assert cfg.attn_impl == "flash_attention"
        else:
            assert cfg.attn_impl == "flex_attention"

        module = cfg.build(hidden_size=32).cuda().to(torch.bfloat16)
        hidden = torch.randn(1, 12, 32, dtype=torch.bfloat16, device="cuda")
        position_embeddings = (
            torch.randn(1, 12, 16, dtype=torch.bfloat16, device="cuda"),
            torch.randn(1, 12, 16, dtype=torch.bfloat16, device="cuda"),
        )
        from xtuner.v1.data_proto import SequenceContext

        seq_ctx = SequenceContext(
            input_ids=hidden.long(),
            cu_seq_lens_q=torch.tensor([0, 12], device="cuda", dtype=torch.int32),
            cu_seq_lens_k=torch.tensor([0, 12], device="cuda", dtype=torch.int32),
            max_length_q=12,
            max_length_k=12,
        )
        outputs = module(hidden, position_embeddings, seq_ctx)
        assert outputs["projected_output"].shape == (1, 12, 32)
        assert torch.isfinite(outputs["projected_output"].float()).all()

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_downgraded_flex_op_matches_eager_reference(self):
        # Numerics guard for the fallback path: flex vs the HF-bitwise eager op. Tolerances
        # are reported, not asserted as bit-parity (fused kernels are close, not bitwise).
        if not _flex_attention_available():
            pytest.skip("flex_attention unavailable")

        total, heads, dim = 12, 4, 16
        q = torch.randn(1, heads, total, dim, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, heads, total, dim, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(1, heads, total, dim, dtype=torch.bfloat16, device="cuda")
        kwargs = {
            "cu_seqlens_q": torch.tensor([0, total], device="cuda", dtype=torch.int32),
            "cu_seqlens_k": torch.tensor([0, total], device="cuda", dtype=torch.int32),
            "max_seqlen_q": total,
            "max_seqlen_k": total,
            "window_size": (-1, -1),
            "dropout_p": 0.0,
            "softmax_scale": dim**-0.5,
            "causal": True,
            "deterministic": False,
        }
        flex_out = get_attn_impl_fn("flex_attention")(q, k, v, **kwargs)["raw_output"]
        eager_out = get_attn_impl_fn("eager_attention")(q, k, v, **kwargs)["raw_output"]
        max_abs_diff = (flex_out.float() - eager_out.float()).abs().max().item()
        print(f"flex-vs-eager max abs diff: {max_abs_diff:.6f}")
        assert torch.allclose(flex_out.float(), eager_out.float(), atol=5e-2, rtol=5e-2)
