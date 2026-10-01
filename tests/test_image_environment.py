import os
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

import delta_svd_constants as c
from conftest import CONTAINER_FILES, REPO_ROOT


def dockerfile_env():
    """The key/value pairs of the final stage's ENV instruction."""
    text = (REPO_ROOT / "container" / "Dockerfile").read_text()
    block = text[text.index("\nENV ") + 5:]
    block = block[:block.index("\n\n")]
    return dict(re.findall(r'(\w+)="?([^"\s\\]*)"?', block))


def test_image_environment_matches_the_dockerfile():
    env = dockerfile_env()
    for name, value in c.IMAGE_ENVIRONMENT.items():
        assert env.get(name) == value, name


def test_entrypoint_isolates_the_interpreter():
    text = (REPO_ROOT / "container" / "Dockerfile").read_text()
    assert 'ENTRYPOINT [ "python", "-E", "-s", "-u", "/opt/scripts/delta-svd.py" ]' in text


# ---------------------------------------------------------------------------
# pin_image_environment(): host values forwarded by Apptainer must not survive.

def test_pin_overrides_host_tool_selection(delta_svd):
    environ = {
        "FSLDIR": "/home/user/fsl",
        "ANTSPATH": "/home/user/bin/ants/",
        "OPENBLAS_CORETYPE": "SkylakeX",
        "FSLOUTPUTTYPE": "NIFTI",
        "HOME": "/home/user",
    }
    delta_svd.pin_image_environment(environ, fslSubConfig=lambda d: None)
    for name, value in c.IMAGE_ENVIRONMENT.items():
        assert environ[name] == value
    assert environ["HOME"] == "/home/user"          # unrelated variables stay


def test_pin_drops_host_python_and_fsl_sub_variables(delta_svd):
    environ = {
        "PYTHONPATH": "/home/user/lib/python3.11/site-packages",
        "PYTHONHOME": "/usr",
        "PYTHONUSERBASE": "/home/user/.local",
        "FSLSUB_CONF": "/home/user/.fsl_sub.yml",
        "FSLSUB_PARALLEL": "8",
        "FSLSUB_PLUGINPATH": "/home/user/plugins",
    }
    delta_svd.pin_image_environment(environ, fslSubConfig=lambda d: None)
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE",
                 "FSLSUB_CONF", "FSLSUB_PARALLEL", "FSLSUB_PLUGINPATH"):
        assert name not in environ
    assert environ["PYTHONNOUSERSITE"] == "1"
    assert environ["PYTHONUNBUFFERED"] == "1"


def test_pin_sets_the_image_fsl_sub_config(delta_svd):
    environ = {"FSLSUB_CONF": "/home/user/.fsl_sub.yml"}
    seen = []
    delta_svd.pin_image_environment(
        environ, fslSubConfig=lambda d: seen.append(d) or "/opt/conda/etc/fslconf/fsl_sub.yml")
    assert seen == [c.IMAGE_ENVIRONMENT["FSLDIR"]]
    assert environ["FSLSUB_CONF"] == "/opt/conda/etc/fslconf/fsl_sub.yml"


def test_fsl_sub_config_prefers_the_fsldir_file(delta_svd, tmp_path, monkeypatch):
    conf = tmp_path / "etc" / "fslconf" / "fsl_sub.yml"
    conf.parent.mkdir(parents=True)
    conf.write_text("method: shell\n")
    monkeypatch.setattr(delta_svd.importlib.util, "find_spec", lambda name: None)
    assert delta_svd.fsl_sub_config(str(tmp_path)) == str(conf)


def test_fsl_sub_config_falls_back_to_the_package_shell_config(delta_svd, tmp_path, monkeypatch):
    # an empty file is skipped, as fsl_sub itself skips it
    empty = tmp_path / "fsl" / "etc" / "fslconf" / "fsl_sub.yml"
    empty.parent.mkdir(parents=True)
    empty.touch()
    shell = tmp_path / "pkg" / "fsl_sub" / "plugins" / "fsl_sub_shell.yml"
    shell.parent.mkdir(parents=True)
    shell.write_text("method: shell\n")
    spec = SimpleNamespace(submodule_search_locations=[str(tmp_path / "pkg" / "fsl_sub")])
    monkeypatch.setattr(delta_svd.importlib.util, "find_spec", lambda name: spec)
    assert delta_svd.fsl_sub_config(str(tmp_path / "fsl")) == str(shell)


def test_fsl_sub_config_none_without_any_file(delta_svd, tmp_path, monkeypatch):
    monkeypatch.setattr(delta_svd.importlib.util, "find_spec", lambda name: None)
    assert delta_svd.fsl_sub_config(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# Interpreter isolation.

def test_running_in_image(delta_svd):
    assert delta_svd.running_in_image(c.IMAGE_SCRIPT_DIR + "/delta-svd.py")
    assert not delta_svd.running_in_image(str(CONTAINER_FILES / "delta-svd.py"))


@pytest.mark.parametrize("ignoreEnv, noUserSite, restart", [
    (0, 0, True), (1, 0, True), (0, 1, True), (1, 1, False)])
def test_isolated_interpreter_argv(delta_svd, ignoreEnv, noUserSite, restart):
    flags = SimpleNamespace(ignore_environment=ignoreEnv, no_user_site=noUserSite)
    argv = delta_svd.isolated_interpreter_argv(flags, ["--dwi", "x.nii.gz"], "/opt/scripts/delta-svd.py")
    if restart:
        assert argv == [sys.executable, "-E", "-s", "-u", "/opt/scripts/delta-svd.py",
                        "--dwi", "x.nii.gz"]
    else:
        assert argv is None


def _image_copy(tmp_path):
    """A copy of the scripts whose IMAGE_SCRIPT_DIR names the copy's own folder,
    so it behaves as the copy installed in the image."""
    scripts = tmp_path / "scripts"
    shutil.copytree(CONTAINER_FILES, scripts)
    constants = scripts / "delta_svd_constants.py"
    constants.write_text(constants.read_text().replace(
        f'IMAGE_SCRIPT_DIR = "{c.IMAGE_SCRIPT_DIR}"', f'IMAGE_SCRIPT_DIR = "{scripts}"'))
    return scripts


def test_image_copy_ignores_a_host_pythonpath(tmp_path):
    # Started as a plain 'python delta-svd.py', without -E -s, the image copy has
    # to re-exec itself isolated before numpy is imported.
    scripts = _image_copy(tmp_path)
    host = tmp_path / "host"
    (host / "numpy").mkdir(parents=True)
    (host / "numpy" / "__init__.py").write_text('raise SystemExit("HOST NUMPY IMPORTED")\n')
    env = dict(os.environ, PYTHONPATH=str(host))

    # without isolation, the planted numpy is what gets imported
    plain = subprocess.run([sys.executable, "-c", "import numpy"], env=env,
                           capture_output=True, text=True)
    assert "HOST NUMPY IMPORTED" in plain.stderr

    out = subprocess.run([sys.executable, str(scripts / "delta-svd.py"), "--version"],
                         env=env, capture_output=True, text=True)
    assert "HOST NUMPY IMPORTED" not in out.stdout + out.stderr
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("DELTA-SVD ")


def test_image_copy_hands_the_pinned_environment_to_subprocesses(tmp_path):
    scripts = _image_copy(tmp_path)
    env = dict(os.environ, FSLDIR="/host/fsl", ANTSPATH="/host/ants",
               OPENBLAS_CORETYPE="Haswell", FSLSUB_CONF="/host/.fsl_sub.yml",
               PYTHONUSERBASE="/host/.local")
    code = (
        "import importlib.util, subprocess, sys\n"
        f"sys.path.insert(0, {str(scripts)!r})\n"
        f"spec = importlib.util.spec_from_file_location('dsvd', {str(scripts / 'delta-svd.py')!r})\n"
        "spec.loader.exec_module(importlib.util.module_from_spec(spec))\n"
        "subprocess.run(['env'])\n")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    child = dict(line.split("=", 1) for line in out.stdout.splitlines() if "=" in line)
    for name, value in c.IMAGE_ENVIRONMENT.items():
        assert child[name] == value, name
    assert child.get("FSLSUB_CONF") != "/host/.fsl_sub.yml"
    assert "PYTHONUSERBASE" not in child
