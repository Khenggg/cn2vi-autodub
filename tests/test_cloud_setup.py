from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cloud_setup.sh"
GIT_BASH = Path("C:/Program Files/Git/bin/bash.exe")
BASH = (str(GIT_BASH) if GIT_BASH.is_file() else None) if os.name == "nt" else shutil.which("bash")


def _bash_platform() -> str:
    if not BASH:
        return ""
    result = subprocess.run([BASH, "-lc", "uname -s"], capture_output=True, text=True, check=False)
    return result.stdout.strip()


def _shell_path(path: Path, platform: str) -> str:
    raw = str(path)
    if os.name != "nt":
        return raw
    if platform.startswith("Linux"):
        drive, rest = os.path.splitdrive(raw)
        if not drive:
            raise ValueError(f"Expected a Windows drive path, got {raw}")
        return f"/mnt/{drive[0].lower()}/{rest.replace(os.sep, '/').lstrip('/')}"
    result = subprocess.run([BASH, "-lc", 'cygpath -u "$1"', "cloud-setup-test", raw],
                            capture_output=True, text=True, check=True)
    return result.stdout.strip()


@pytest.fixture
def shell_platform() -> str:
    if not BASH:
        pytest.skip("bash is not installed")
    platform = _bash_platform()
    if os.name == "nt" and not (platform.startswith("Linux") or platform.startswith(("MSYS", "MINGW"))):
        pytest.skip("bash shell integration needs WSL or Git Bash")
    return platform


def _fake_project(tmp_path: Path, platform: str) -> tuple[Path, Path, Path, Path, Path, Path]:
    project = tmp_path / "project"
    data = tmp_path / "data"
    venv = tmp_path / "venvs"
    project.joinpath("benchmarks").mkdir(parents=True)
    project.joinpath("scripts").mkdir()
    project.joinpath("config").mkdir()
    repo = SCRIPT.parent.parent
    fixture_runtime = json.dumps({
        "schema_version": 1,
        "profiles": ["asr", "tts", "vision", "separation"],
        "assets": [
            "whisper-asr", "alignment-en", "alignment-zh", "vieneu-turbo",
            "moss-torch", "lama-onnx", "rapidocr-v6", "roformer", "nltk-tokenizers"
        ],
        "validation": "PENDING_FRESH_CLOUD_INSTALL_AND_MODEL_TESTS"
    }, indent=2)
    project.joinpath("config", "cloud-runtime.json").write_text(fixture_runtime, encoding="utf-8")
    project.joinpath("scripts", "cloud_plan.py").write_text(
        repo.joinpath("scripts", "cloud_plan.py").read_text(encoding="utf-8"), encoding="utf-8")

    project.joinpath("benchmarks", "models.lock.json").write_text(
        repo.joinpath("benchmarks", "models.lock.json").read_text(encoding="utf-8"), encoding="utf-8")
    os_release = project / "test-os-release"
    os_release.write_text('ID=ubuntu\nVERSION_ID="24.04"\n', encoding="utf-8")
    fixture_script = SCRIPT.read_text(encoding="utf-8").replace(
        "/etc/os-release", _shell_path(os_release, platform)
    )
    project.joinpath("scripts", "cloud_setup.sh").write_text(fixture_script, encoding="utf-8")
    project.joinpath("scripts", "cloud_bootstrap.sh").write_text(
        """#!/usr/bin/env bash
printf 'called\\n' >> "$BOOTSTRAP_MARKER"
exit 0
""",
        encoding="utf-8",
    )
    call_log = tmp_path / "calls.log"
    host_log = tmp_path / "host-checks.log"
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_nvidia = fake_bin / "nvidia-smi"
    fake_nvidia.write_text(
        """#!/usr/bin/env bash
printf 'nvidia-smi\\n' >> "$HOST_CHECK_LOG"
[[ "${FAKE_GPU_FAIL:-0}" != 1 ]] || exit 9
printf '16000\\n'
""",
        encoding="utf-8",
    )
    fake_nvidia.chmod(0o755)
    fake_df = fake_bin / "df"
    fake_df.write_text(
        """#!/usr/bin/env bash
printf 'df\\n' >> "$HOST_CHECK_LOG"
printf 'Filesystem 1B-blocks Used Available Use%% Mounted on\\nfake 999999999999 0 999999999999 0%% /\\n'
""",
        encoding="utf-8",
    )
    fake_df.chmod(0o755)
    for command in ("sudo", "apt-get", "python3.12", "curl", "wget"):
        sentinel = fake_bin / command
        sentinel.write_text(
            "#!/usr/bin/env bash\nprintf '%s\\n' \"$0 $*\" >> \"$DANGEROUS_COMMAND_LOG\"\nexit 95\n",
            encoding="utf-8",
        )
        sentinel.chmod(0o755)
    fake_python = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$FAKE_CALL_LOG"
if [[ "$1 $2" == "-m autodub.preflight" ]]; then
  profile=''
  while (($#)); do
    if [[ "$1" == '--profile' ]]; then profile="$2"; shift 2; else shift; fi
  done
  [[ "${FAKE_FAIL_PROFILE:-}" != "$profile" ]] || exit 2
fi
exit 0
"""
    for environment in ("core", "asr", "tts", "vision", "separation"):
        executable = venv / environment / "bin" / "python"
        executable.parent.mkdir(parents=True)
        executable.write_text(fake_python, encoding="utf-8")
        executable.chmod(0o755)
    return project, data, venv, call_log, host_log, fake_bin


def _run_setup(project: Path, data: Path, venv: Path, call_log: Path, host_log: Path, fake_bin: Path,
               journal: Path, platform: str, *arguments: str, fail_profile: str = "",
               fail_host: bool = False) -> subprocess.CompletedProcess[str]:
    shell_script = _shell_path(project / "scripts" / "cloud_setup.sh", platform)
    bindings = {
        "PROJECT_ROOT": _shell_path(project, platform),
        "AUTODUB_DATA_ROOT": _shell_path(data, platform),
        "AUTODUB_VENV_ROOT": _shell_path(venv, platform),
        "AUTODUB_SETUP_JOURNAL": _shell_path(journal, platform),
        "FAKE_CALL_LOG": _shell_path(call_log, platform),
        "FAKE_FAIL_PROFILE": fail_profile,
        "BOOTSTRAP_MARKER": _shell_path(project.parent / "bootstrap.log", platform),
        "HOST_CHECK_LOG": _shell_path(host_log, platform),
        "FAKE_GPU_FAIL": "1" if fail_host else "0",
        "DANGEROUS_COMMAND_LOG": _shell_path(project.parent / "dangerous-commands.log", platform),
    }
    exports = " ".join(f"{key}={shlex.quote(value)}" for key, value in bindings.items())
    command = f"export {exports}; export PATH={shlex.quote(_shell_path(fake_bin, platform))}:/usr/bin:/bin; " \
        f"exec bash {shlex.quote(shell_script)} " + " ".join(shlex.quote(arg) for arg in arguments)
    return subprocess.run([BASH, "-c", command],
                          capture_output=True, text=True, check=False)


def test_dry_run_only_prints_plan_and_never_calls_bootstrap(tmp_path: Path, shell_platform: str) -> None:
    project, data, venv, call_log, host_log, fake_bin = _fake_project(tmp_path, shell_platform)
    journal = data / "results" / "dry-run.log"
    result = _run_setup(project, data, venv, call_log, host_log, fake_bin, journal, shell_platform, "--dry-run")

    assert result.returncode == 0
    assert "DRY RUN" in result.stdout
    assert "whisper-asr" in result.stdout and "roformer" in result.stdout
    assert "asr tts vision separation" in result.stdout
    assert "moss-onnx" not in result.stdout and "vieneu-turbo-onnx" not in result.stdout
    assert not call_log.exists()
    assert not host_log.exists()
    assert not (project.parent / "dangerous-commands.log").exists()
    assert not journal.exists()


def test_success_runs_all_steps_and_rerun_reaches_ready_again(tmp_path: Path, shell_platform: str) -> None:
    project, data, venv, call_log, host_log, fake_bin = _fake_project(tmp_path, shell_platform)
    for run_number in (1, 2):
        journal = data / "results" / f"run-{run_number}.log"
        result = _run_setup(project, data, venv, call_log, host_log, fake_bin, journal, shell_platform)
        assert result.returncode == 0, result.stderr
        contents = journal.read_text(encoding="utf-8")
        assert "run_status=ASSETS_VERIFIED" in contents
        assert "status=PASS" in contents
    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert (project.parent / "bootstrap.log").read_text(encoding="utf-8").splitlines() == ["called", "called"]
    assert host_log.read_text(encoding="utf-8").splitlines().count("nvidia-smi") == 2
    assert host_log.read_text(encoding="utf-8").splitlines().count("df") == 2
    assert not (project.parent / "dangerous-commands.log").exists()
    assert sum(line == "-m autodub.model_assets fetch" or line.startswith("-m autodub.model_assets fetch ")
               for line in calls) == 2
    assert sum(line.startswith("-m autodub.model_assets verify ") for line in calls) == 2
    for profile in ("asr", "tts", "vision", "separation"):
        assert sum(f"--profile {profile}" in line for line in calls) == 2
    assert "--model-manifest" in next(line for line in calls if "--profile asr" in line)


def test_failed_preflight_preserves_exit_code_and_stops_later_profiles(
    tmp_path: Path, shell_platform: str
) -> None:
    project, data, venv, call_log, host_log, fake_bin = _fake_project(tmp_path, shell_platform)
    journal = data / "results" / "failed.log"
    result = _run_setup(project, data, venv, call_log, host_log, fake_bin, journal, shell_platform,
                        fail_profile="vision")

    assert result.returncode == 2
    assert "BLOCKED step=preflight_vision exit_code=2" in result.stderr
    contents = journal.read_text(encoding="utf-8")
    assert "run_status=BLOCKED" in contents
    assert "failed_step=preflight_vision" in contents
    assert "run_status=ASSETS_VERIFIED" not in contents
    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert any("--profile vision" in line for line in calls)
    assert not any("--profile separation" in line for line in calls)
    assert not (project.parent / "dangerous-commands.log").exists()


def test_host_gate_runs_before_bootstrap_and_model_downloads(tmp_path: Path, shell_platform: str) -> None:
    project, data, venv, call_log, host_log, fake_bin = _fake_project(tmp_path, shell_platform)
    journal = data / "results" / "host-failed.log"
    result = _run_setup(project, data, venv, call_log, host_log, fake_bin, journal,
                        shell_platform, fail_host=True)

    assert result.returncode == 69
    assert "BLOCKED step=host_preflight exit_code=69" in result.stderr
    assert not (project.parent / "bootstrap.log").exists()
    assert not call_log.exists()
    assert not (project.parent / "dangerous-commands.log").exists()
    contents = journal.read_text(encoding="utf-8")
    assert "failed_step=host_preflight" in contents
    assert "run_status=ASSETS_VERIFIED" not in contents


def test_cloud_plan_supports_diarization_punctuation_indextts_and_explicit_profiles(tmp_path: Path) -> None:
    repo = SCRIPT.parent.parent
    project = tmp_path / "project"
    project.joinpath("config").mkdir(parents=True)
    project.joinpath("scripts").mkdir()
    project.joinpath("benchmarks").mkdir()
    fixture_runtime = json.dumps({
        "schema_version": 1,
        "profiles": ["asr", "tts", "vision", "separation"],
        "assets": [
            "whisper-asr", "alignment-en", "alignment-zh", "vieneu-turbo",
            "moss-torch", "lama-onnx", "rapidocr-v6", "roformer", "nltk-tokenizers"
        ],
        "validation": "PENDING_FRESH_CLOUD_INSTALL_AND_MODEL_TESTS"
    }, indent=2)
    project.joinpath("config", "cloud-runtime.json").write_text(fixture_runtime, encoding="utf-8")
    project.joinpath("scripts", "cloud_plan.py").write_text(
        repo.joinpath("scripts", "cloud_plan.py").read_text(encoding="utf-8"), encoding="utf-8")
    project.joinpath("benchmarks", "models.lock.json").write_text(
        repo.joinpath("benchmarks", "models.lock.json").read_text(encoding="utf-8"), encoding="utf-8")


    plan_py = project / "scripts" / "cloud_plan.py"

    # Test default profiles
    result = subprocess.run([sys.executable, str(plan_py), "--project", str(project), "--field", "profiles"],
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip().splitlines() == ["asr", "tts", "vision", "separation"]

    # Test explicit profiles with upcoming profiles: diarization, punctuation, indextts
    result = subprocess.run([sys.executable, str(plan_py), "--project", str(project),
                             "--field", "profiles", "--profiles", "diarization,punctuation,indextts"],
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip().splitlines() == ["diarization", "punctuation", "indextts"]

    # Test explicit single profile
    result = subprocess.run([sys.executable, str(plan_py), "--project", str(project),
                             "--field", "profiles", "--profile", "diarization"],
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip().splitlines() == ["diarization"]

    # Test invalid profile rejected
    invalid_result = subprocess.run([sys.executable, str(plan_py), "--project", str(project),
                                     "--field", "profiles", "--profile", "invalid/profile"],
                                    capture_output=True, text=True, check=False)
    assert invalid_result.returncode != 0


def test_dry_run_with_explicit_profiles(tmp_path: Path, shell_platform: str) -> None:
    project, data, venv, call_log, host_log, fake_bin = _fake_project(tmp_path, shell_platform)
    journal = data / "results" / "dry-run-profiles.log"
    result = _run_setup(project, data, venv, call_log, host_log, fake_bin, journal,
                        shell_platform, "--dry-run", "--profiles", "asr,tts")

    assert result.returncode == 0
    assert "DRY RUN" in result.stdout
    assert "Run --require-cloud preflight profiles: asr tts" in result.stdout
    assert "vision" not in result.stdout
    assert not journal.exists()


def test_journal_records_readiness_distinct_from_inference_quality(tmp_path: Path, shell_platform: str) -> None:
    project, data, venv, call_log, host_log, fake_bin = _fake_project(tmp_path, shell_platform)
    journal = data / "results" / "journal-check.log"
    result = _run_setup(project, data, venv, call_log, host_log, fake_bin, journal, shell_platform)
    assert result.returncode == 0, result.stderr
    contents = journal.read_text(encoding="utf-8")
    assert "run_status=ASSETS_VERIFIED" in contents
    assert "inference_quality=UNVERIFIED" in contents
    assert "model inference and video quality remain unverified" in result.stdout


def test_verify_startup_option_runs_and_journals_startup_verified(tmp_path: Path, shell_platform: str) -> None:
    project, data, venv, call_log, host_log, fake_bin = _fake_project(tmp_path, shell_platform)
    start_web = project / "scripts" / "start_web.sh"
    start_web.write_text("""#!/usr/bin/env bash
if [[ "${1:-}" == "start" ]]; then
  printf 'started\\n' >> "$START_WEB_LOG"
  exit 0
elif [[ "${1:-}" == "status" ]]; then
  exit 0
fi
exit 0
""", encoding="utf-8")
    start_web.chmod(0o755)
    journal = data / "results" / "startup-verified.log"
    result = _run_setup(project, data, venv, call_log, host_log, fake_bin, journal,
                        shell_platform, "--verify-startup")
    assert result.returncode == 0, result.stderr
    contents = journal.read_text(encoding="utf-8")
    assert "step=verify_startup status=PASS" in contents
    assert "run_status=STARTUP_VERIFIED" in contents
    assert "inference_quality=UNVERIFIED" in contents
