# Copyright (c) OpenMMLab. All rights reserved.
"""Regression tests for ``ExpandSoftPackDataset`` pack-generator device alignment.

``torch.randperm`` runs on the default device; under FSDP2 the default device is CUDA
while ``torch.Generator()`` defaults to CPU, which raised ``Expected a 'cuda' device type
for generator but found 'cpu'`` inside ``QADTrainer.__init__``.
"""

import unittest

import numpy as np
import pytest
import torch

from xtuner.v1.datasets.packing import ExpandSoftPackDataset


class _TinySizedDataset:
    """Minimal stand-in matching the attributes ``get_pack_infos`` consumers rely on."""

    def __init__(self) -> None:
        self.num_tokens = np.ones(5, dtype=np.int64)
        self.proxy_attn_flops = np.ones(5, dtype=np.float32)

    def __len__(self) -> int:
        return 5

    def __getitem__(self, index: int) -> dict:
        return {"data": index}


def _make_dataset(seed: int | None) -> ExpandSoftPackDataset:
    return ExpandSoftPackDataset(datasets=[_TinySizedDataset()], pack_max_length=4, seed=seed)


def _pack_infos(ds: ExpandSoftPackDataset):
    num_tokens = np.ones(len(ds), dtype=np.int64)
    proxy_attn_flops = np.ones(len(ds), dtype=np.float32)
    return ds.get_pack_infos(ds, dataset_id=0, num_tokens=num_tokens, proxy_attn_flops=proxy_attn_flops)


class TestPackGeneratorDeviceAlignment:
    def test_cpu_default_device_uses_stored_generator(self):
        """CPU default device keeps the legacy behavior: stored CPU generator drives randperm."""
        ds = _make_dataset(seed=42)
        infos_a = _pack_infos(ds)
        infos_b = _pack_infos(ds)
        assert infos_a["indices"] == infos_b["indices"], "same-seed CPU packing must be deterministic"

    def test_reseed_changes_cpu_permutation(self):
        a = _pack_infos(_make_dataset(seed=42))
        b = _pack_infos(_make_dataset(seed=43))
        assert a["indices"] != b["indices"], "different seeds must produce different permutations"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
    def test_cuda_default_device_does_not_crash_and_is_deterministic(self):
        """CUDA default device (FSDP2) must not trip the CPU-generator device assert."""
        prev = torch.get_default_device()
        try:
            torch.set_default_device("cuda")
            ds = _make_dataset(seed=42)
            infos_a = _pack_infos(ds)
            infos_b = _pack_infos(_make_dataset(seed=42))
            assert infos_a["indices"] == infos_b["indices"], "same-seed CUDA packing must be deterministic"
            covered = [i for info in infos_a for i in info["indices"]]
            assert sorted(covered) == list(range(len(ds))), "packing must be a full permutation"
        finally:
            torch.set_default_device(prev)


if __name__ == "__main__":
    unittest.main()
