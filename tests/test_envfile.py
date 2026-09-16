from scriba.envfile import find_local_model


def test_find_local_model_missing_returns_none(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBA_ENV_FILE", str(tmp_path / ".env"))
    monkeypatch.setattr("scriba.envfile.project_root", lambda: tmp_path)
    assert find_local_model("Qwen/Qwen3-ASR-1.7B", marker="config.json") is None


def test_find_local_model_resolves_project_root_config_models(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBA_ENV_FILE", str(tmp_path / "unrelated" / ".env"))
    monkeypatch.setattr("scriba.envfile.project_root", lambda: tmp_path)
    model_dir = tmp_path / "config" / "models" / "Qwen--Qwen3-ASR-1.7B"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    found = find_local_model("Qwen/Qwen3-ASR-1.7B", marker="config.json")
    assert found == model_dir


def test_find_local_model_resolves_env_file_sibling_models_dir(monkeypatch, tmp_path):
    env_dir = tmp_path / "data_config"
    monkeypatch.setenv("SCRIBA_ENV_FILE", str(env_dir / ".env"))
    monkeypatch.setattr("scriba.envfile.project_root", lambda: tmp_path / "elsewhere")
    model_dir = env_dir / "models" / "Qwen--Qwen3-ForcedAligner-0.6B"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    found = find_local_model("Qwen/Qwen3-ForcedAligner-0.6B", marker="config.json")
    assert found == model_dir
