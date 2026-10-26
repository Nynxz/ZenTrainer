"""Settings > Versions (server/routes/common.py _zt_version): what's running, a newer commit on disk,
and what the upstream has, from throwaway git repos (nothing outside tmp_path is touched)."""

import subprocess

from zentrainer.server.routes import common


def _git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


def _checkout(tmp_path, monkeypatch):
    up = tmp_path / "up"
    up.mkdir()
    _git(up, "init", "-q", "-b", "main")
    (up / "a.txt").write_text("1")
    _git(up, "add", "a.txt")
    _git(up, "commit", "-qm", "first")
    _git(up, "tag", "v0.3.0")
    _git(tmp_path, "clone", "-q", str(up), "zt")
    zt = tmp_path / "zt"
    monkeypatch.setattr(common, "REPO", zt)
    monkeypatch.setattr(common, "_LOADED_COMMIT", _git(zt, "rev-parse", "HEAD"))
    return up, zt


def test_up_to_date_then_new_commits_upstream(tmp_path, monkeypatch):
    up, zt = _checkout(tmp_path, monkeypatch)
    v = common._zt_version(True)
    assert v["describe"] == "v0.3.0" and v["branch"] == "main" and not v["dirty"] and "on_disk" not in v
    assert v["upstream"]["behind"] == 0 and v["upstream"]["ref"] == "origin/main"
    (up / "a.txt").write_text("2")
    _git(up, "commit", "-qam", "Captioning: hide masked areas")
    v = common._zt_version(True)
    assert v["upstream"]["behind"] == 1 and v["upstream"]["commits"][0].endswith(" Captioning: hide masked areas")
    assert _git(zt, "rev-parse", "HEAD") == v["commit"]  # only fetched: the checkout didn't move
    assert "upstream" not in common._zt_version(False)  # no fetch unless asked


def test_a_pull_while_running_and_local_changes(tmp_path, monkeypatch):
    up, zt = _checkout(tmp_path, monkeypatch)
    (up / "a.txt").write_text("2")
    _git(up, "commit", "-qam", "second")
    _git(zt, "pull", "-q")
    (zt / "a.txt").write_text("edited")
    v = common._zt_version(False)
    assert v["on_disk"]["commit"] == _git(zt, "rev-parse", "HEAD") != v["commit"]  # restart to load it
    assert v["dirty"]


def test_a_detached_checkout_or_no_git_says_so(tmp_path, monkeypatch):
    _, zt = _checkout(tmp_path, monkeypatch)
    _git(zt, "checkout", "-q", "--detach")
    v = common._zt_version(True)
    assert v["branch"] is None and "isn't on a branch" in v["upstream_error"]
    monkeypatch.setattr(common, "_LOADED_COMMIT", "")
    assert common._zt_version(True) == {"version": common._zt_version(False)["version"]}
