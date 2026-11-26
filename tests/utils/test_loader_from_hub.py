import sys
from pathlib import Path
from typing import Any

import pytest

from xtuner.v1.utils.loader import download_model_from_hub


class _Recorder:
    def __init__(self, ret: str, name: str) -> None:
        self.ret = ret
        self.name = name
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return self.ret


class TestDownloadModelFromHubEnv:
    def _install_stubs(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[_Recorder, _Recorder]:
        hf = _Recorder(str(tmp_path / "hf"), "hf")
        ms = _Recorder(str(tmp_path / "ms"), "ms")

        hf_mod = type(sys)("huggingface_hub")
        hf_mod.snapshot_download = hf
        ms_mod = type(sys)("modelscope")
        ms_mod.snapshot_download = ms
        monkeypatch.setitem(sys.modules, "huggingface_hub", hf_mod)
        monkeypatch.setitem(sys.modules, "modelscope", ms_mod)
        return hf, ms

    def test_default_huggingface(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("XTUNER_FROM_HUB", raising=False)
        hf, ms = self._install_stubs(monkeypatch, tmp_path)
        res = download_model_from_hub("some/repo")
        assert res == str(tmp_path / "hf")
        assert hf.calls == [{"repo_id": "some/repo", "cache_dir": None}]
        assert ms.calls == []

    def test_env_modelscope_overrides_default(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("XTUNER_FROM_HUB", "modelscope")
        hf, ms = self._install_stubs(monkeypatch, tmp_path)
        res = download_model_from_hub("some/repo")
        assert res == str(tmp_path / "ms")
        assert ms.calls == [{"model_id": "some/repo", "cache_dir": None}]
        assert hf.calls == []

    def test_env_invalid_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("XTUNER_FROM_HUB", "foo")
        hf, ms = self._install_stubs(monkeypatch, tmp_path)
        with pytest.raises(NotImplementedError):
            download_model_from_hub("some/repo")
        assert hf.calls == [] and ms.calls == []

    def test_local_dir_ignores_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("XTUNER_FROM_HUB", "modelscope")
        hf, ms = self._install_stubs(monkeypatch, tmp_path)
        local = tmp_path / "local_model"
        local.mkdir()
        assert download_model_from_hub(str(local)) == str(local)
        assert hf.calls == [] and ms.calls == []

    def test_env_whitespace_treated_unset(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("XTUNER_FROM_HUB", "   ")
        hf, ms = self._install_stubs(monkeypatch, tmp_path)
        res = download_model_from_hub("some/repo")
        assert res == str(tmp_path / "hf")
        assert hf.calls == [{"repo_id": "some/repo", "cache_dir": None}]
        assert ms.calls == []
