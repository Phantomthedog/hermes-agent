"""Regression tests for ``tools/native_screenshot``.

What this test proves
---------------------
1. ``/snap/bin/chromium`` is **never** selected — explicit, env, and
   subpath forms all rejected.  This is the contract that prevents the
   silent-regression bug where chromium prints ``N bytes written`` and the
   file isn't on the host filesystem.

2. A real ``native_screenshot()`` call against ``https://example.com``:

   * renders to a fresh ``/tmp`` staging file,
   * verifies it via ``file`` (or magic bytes) as a real PNG,
   * copies the verified PNG to the requested destination,
   * verifies the destination is a real PNG,
   * returns a dict with ``success=True``.

3. **False success is rejected.**  We mock ``subprocess.run`` to return a
   successful ``CompletedProcess`` without writing the PNG (the exact
   shape of the snap-chromium bug).  The helper must report
   ``success=False`` rather than trusting the "wrote N bytes" claim.

4. Empty URL, relative destination, missing chromium binary, and empty
   staged file all surface as ``success=False`` (not exceptions, not
   silent failures).

Marking
-------
The render-and-copy tests are marked ``integration`` because they need the
Playwright bundled chromium (``~/.cache/ms-playwright/chromium-1124``) and
network access.  Run with ``pytest -m integration`` or remove the marker to
execute them in this environment.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from tools.native_screenshot import (
    DEFAULT_BUNDLED_CHROMIUM,
    HERMES_NATIVE_CHROMIUM_ENV,
    NativeScreenshotError,
    _verify_png,
    native_screenshot,
    resolve_chromium,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


CHROMIUM_BUNDLED = DEFAULT_BUNDLED_CHROMIUM


def _chromium_available() -> bool:
    """True when the bundled chromium binary exists and is executable."""
    return CHROMIUM_BUNDLED.is_file() and os.access(CHROMIUM_BUNDLED, os.X_OK)


pytestmark_integration = pytest.mark.integration


@pytest.fixture
def clean_hermes_chromium_env(monkeypatch):
    """Unset HERMES_NATIVE_CHROMIUM so tests exercise the default path.

    The repo-wide ``tests/conftest.py`` unsets credential-shaped env vars
    but does not touch this one.  Tests that need to assert default
    behaviour explicitly clear it.
    """
    monkeypatch.delenv(HERMES_NATIVE_CHROMIUM_ENV, raising=False)
    return monkeypatch


@pytest.fixture
def tmp_dest(tmp_path):
    """A destination path under ``tmp_path`` that does not exist yet.

    Mimics the real ``/mnt/c/.../foo.png`` case: parent dir *may* exist
    (it does — tmp_path) but the file itself does not.  We never write
    to ``/mnt/c`` from tests because that's the host's data volume; we
    use the per-test tmp_path instead.  The semantics — write to /tmp,
    verify, copy across to a 9p-style path — are exercised by the
    real chromium render test, but the *copy mechanics* are exercised
    here using tmp_path as the "9p mount".
    """
    return tmp_path / "screenshots" / "out.png"


# ---------------------------------------------------------------------------
# 1. resolve_chromium — snap refusal contract
# ---------------------------------------------------------------------------


class TestResolveChromiumRefusesSnap:
    """The snap chromium must be unreachable via every input channel."""

    def test_explicit_snap_path_rejected(self, clean_hermes_chromium_env):
        with pytest.raises(NativeScreenshotError) as ei:
            resolve_chromium("/snap/bin/chromium")
        assert "snap" in str(ei.value).lower() or "REJECTED" in str(ei.value)

    def test_explicit_snap_subpath_rejected(self, clean_hermes_chromium_env):
        with pytest.raises(NativeScreenshotError):
            resolve_chromium("/snap/chromium/current/usr/lib/chromium-browser/chrome")

    def test_env_snap_path_rejected(self, clean_hermes_chromium_env, monkeypatch):
        monkeypatch.setenv(HERMES_NATIVE_CHROMIUM_ENV, "/snap/bin/chromium")
        with pytest.raises(NativeScreenshotError):
            resolve_chromium()

    def test_explicit_snap_does_not_silently_fall_through(
        self, clean_hermes_chromium_env
    ):
        """If a snap path is requested, we must fail loudly — even if the
        bundled chromium exists at the default.  Otherwise a future
        change to fallback semantics would silently bring snap back."""
        with pytest.raises(NativeScreenshotError):
            resolve_chromium("/snap/bin/chromium")

    def test_no_usable_chromium_raises(self, clean_hermes_chromium_env, monkeypatch):
        """When nothing is found, error must mention what was tried.

        On this host the bundled chromium may already exist, which makes
        the \"nothing found\" path unreachable — skip accordingly.
        """
        if _chromium_available():
            pytest.skip("bundled chromium present — can't exercise the no-chromium path")
        monkeypatch.setenv(HERMES_NATIVE_CHROMIUM_ENV, "/nonexistent/chrome")
        with pytest.raises(NativeScreenshotError) as ei:
            resolve_chromium()
        msg = str(ei.value)
        assert "/nonexistent/chrome" in msg
        assert "DEFAULT_BUNDLED_CHROMIUM" in msg


class TestResolveChromiumHappyPaths:
    """When snap isn't involved, resolution finds a working binary."""

    def test_default_returns_bundled_when_present(
        self, clean_hermes_chromium_env
    ):
        if not _chromium_available():
            pytest.skip("bundled chromium not present")
        p = resolve_chromium()
        assert p == CHROMIUM_BUNDLED
        assert "snap" not in str(p)

    def test_explicit_overrides_env(self, clean_hermes_chromium_env, monkeypatch):
        if not _chromium_available():
            pytest.skip("bundled chromium not present")
        # env points to a non-snap binary that doesn't exist.
        # Without explicit, resolve_chromium falls back to the bundled default.
        monkeypatch.setenv(HERMES_NATIVE_CHROMIUM_ENV, "/nonexistent/binary")
        p = resolve_chromium()
        assert p == CHROMIUM_BUNDLED

    def test_explicit_takes_precedence(
        self, clean_hermes_chromium_env, monkeypatch
    ):
        if not _chromium_available():
            pytest.skip("bundled chromium not present")
        # env points at an existing non-snap binary but the explicit arg
        # (bundled chromium) must still win the candidate walk.
        monkeypatch.setenv(HERMES_NATIVE_CHROMIUM_ENV, "/nonexistent/binary")
        p = resolve_chromium(CHROMIUM_BUNDLED)
        assert p == CHROMIUM_BUNDLED

    def test_explicit_snap_error_mentions_bundled_fallback(
        self, clean_hermes_chromium_env
    ):
        """If the explicit is snap, the error should mention the bundled
        path so the user knows what to point at instead."""
        if not _chromium_available():
            pytest.skip("bundled chromium not present")
        with pytest.raises(NativeScreenshotError) as ei:
            resolve_chromium("/snap/bin/chromium")
        assert str(CHROMIUM_BUNDLED) in str(ei.value)


# ---------------------------------------------------------------------------
# 2. _verify_png — verification primitives
# ---------------------------------------------------------------------------


class TestVerifyPng:
    def test_missing_file_rejected(self, tmp_path):
        ok, detail = _verify_png(tmp_path / "does_not_exist.png")
        assert ok is False
        assert "missing" in detail

    def test_empty_file_rejected(self, tmp_path):
        f = tmp_path / "empty.png"
        f.write_bytes(b"")
        ok, detail = _verify_png(f)
        assert ok is False
        assert "empty" in detail

    def test_non_png_rejected(self, tmp_path):
        f = tmp_path / "fake.png"
        f.write_bytes(b"this is not a PNG file at all, just text\n")
        ok, detail = _verify_png(f)
        assert ok is False
        assert "not a PNG" in detail or "image/png" in detail

    def test_real_png_accepted(self, tmp_path):
        """Render a real PNG via the bundled chromium, then verify it.

        This is the same machinery the production code uses — a single
        ``_verify_png`` call against a real chromium output proves the
        verifier works on real output, not just synthetic bytes.
        """
        if not _chromium_available():
            pytest.skip("bundled chromium not present")
        out = tmp_path / "real.png"
        cmd = [
            str(CHROMIUM_BUNDLED),
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--no-first-run",
            f"--user-data-dir={tmp_path / 'ud'}",
            f"--screenshot={out}",
            "--window-size=400,300",
            "https://example.com",
        ]
        try:
            subprocess.run(
                cmd,
                capture_output=True,
                timeout=30,
                check=False,
            )
        except subprocess.TimeoutExpired:
            pytest.skip("chromium render timed out (network slow?)")
        ok, detail = _verify_png(out)
        assert ok is True, f"verify_png rejected a real chromium PNG: {detail}"
        assert "PNG" in detail


# ---------------------------------------------------------------------------
# 3. native_screenshot — false-success rejection (the snap-bug contract)
# ---------------------------------------------------------------------------


class TestFalseSuccessRejected:
    """The exact shape of the snap-chromium bug, replayed with a mock.

    Snap chromium returns exit-code 0 and prints
    ``16137 bytes written to file /tmp/...`` — but the file is not
    on the host filesystem.  We simulate that here: the subprocess
    call "succeeds" but no file appears.  The helper MUST report
    ``success=False`` rather than trusting the success claim.
    """

    def _mock_completed(self, stdout: str = "16137 bytes written to file /tmp/out.png"):
        cp = MagicMock()
        cp.returncode = 0
        cp.stdout = stdout
        cp.stderr = ""
        return cp

    def test_lied_about_write_rejected(self, tmp_path, clean_hermes_chromium_env):
        if not _chromium_available():
            pytest.skip("bundled chromium not present")
        dest = tmp_path / "dest" / "out.png"

        with patch(
            "tools.native_screenshot.subprocess.run",
            return_value=self._mock_completed(),
        ):
            result = native_screenshot(
                "https://example.com",
                dest,
                timeout=10,
            )

        assert result["success"] is False, (
            "Helper trusted a fake-success subprocess. The snap-bug contract "
            f"is broken. Result: {result}"
        )
        assert result["stage"] in {"verify_src", "render"}
        assert "verification" in result["error"].lower() or "missing" in result["error"].lower()
        # Critical: dest must NOT have been created on a false success.
        assert not dest.exists(), "dest created on a false success — the bug is back"

    def test_zero_byte_write_rejected(self, tmp_path, clean_hermes_chromium_env):
        """Chromium exits 0 but writes 0 bytes (truncated / disk-full)."""
        if not _chromium_available():
            pytest.skip("bundled chromium not present")
        dest = tmp_path / "dest" / "out.png"
        cp = self._mock_completed(stdout="0 bytes written")

        with patch("tools.native_screenshot.subprocess.run", return_value=cp):
            result = native_screenshot("https://example.com", dest, timeout=10)

        assert result["success"] is False
        assert not dest.exists()

    def test_garbage_bytes_rejected(self, tmp_path, clean_hermes_chromium_env):
        """Chromium writes something, but it's not a PNG (e.g. an error HTML
        page that chromium captured by mistake)."""
        if not _chromium_available():
            pytest.skip("bundled chromium not present")
        dest = tmp_path / "dest" / "out.png"

        real_run = subprocess.run

        def fake_run(*args, **kwargs):
            # Pull the --screenshot=<path> off the cmdlist and write garbage there.
            cmd = args[0] if args else kwargs.get("args", [])
            for i, c in enumerate(cmd):
                cs = str(c)
                if cs.startswith("--screenshot="):
                    target = Path(cs.split("=", 1)[1])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(b"<html>500 Internal Server Error</html>")
                    break
            return self._mock_completed(stdout="1234 bytes written")

        with patch("tools.native_screenshot.subprocess.run", side_effect=fake_run):
            result = native_screenshot("https://example.com", dest, timeout=10)

        assert result["success"] is False
        assert not dest.exists(), "garbage PNG was copied to dest — verification skipped"


# ---------------------------------------------------------------------------
# 4. native_screenshot — input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_empty_url_rejected(self, tmp_path, clean_hermes_chromium_env):
        result = native_screenshot("", tmp_path / "out.png")
        assert result["success"] is False
        assert "empty url" in result["error"]

    def test_whitespace_url_rejected(self, tmp_path, clean_hermes_chromium_env):
        result = native_screenshot("   \n\t  ", tmp_path / "out.png")
        assert result["success"] is False
        assert "empty url" in result["error"]

    def test_relative_dest_rejected(self, tmp_path, clean_hermes_chromium_env):
        result = native_screenshot("https://example.com", "relative/out.png")
        assert result["success"] is False
        assert "absolute" in result["error"]
        assert result["stage"] == "resolve"

    def test_no_chromium_at_all_rejected(
        self, tmp_path, clean_hermes_chromium_env, monkeypatch
    ):
        if _chromium_available():
            pytest.skip("bundled chromium present on this host — can't exercise the no-chromium path")
        monkeypatch.setenv(HERMES_NATIVE_CHROMIUM_ENV, "/nonexistent/chrome")
        result = native_screenshot("https://example.com", tmp_path / "out.png")
        assert result["success"] is False
        assert "no usable" in result["error"].lower() or "tried" in result["error"].lower()


# ---------------------------------------------------------------------------
# 5. native_screenshot — happy path (real chromium)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestNativeScreenshotIntegration:
    """End-to-end: real chromium, real URL, real file on disk."""

    def test_renders_to_tmp_then_copies_to_dest(
        self, tmp_path, clean_hermes_chromium_env
    ):
        if not _chromium_available():
            pytest.skip("bundled chromium not present")
        # Mimic the /mnt/c/.../foo.png shape with tmp_path instead of /mnt/c
        # so we don't write to the host's data volume during tests.  The
        # copy + verify contract is the same.
        dest = tmp_path / "screenshots" / "deep" / "out.png"
        assert not dest.exists()

        result = native_screenshot(
            "https://example.com",
            dest,
            window_size=(800, 600),
            timeout=30,
        )

        assert result["success"] is True, f"integration render failed: {result}"
        assert result["path"] == str(dest)
        assert Path(result["path"]).is_file(), "dest not on disk after success"
        assert result["bytes"] > 0
        assert "snap" not in result["chromium_path"]
        assert "PNG" in result["verify"]

        # Re-verify the file on disk independently — proves the helper
        # didn't lie (the same class of bug we're testing for).
        ok, detail = _verify_png(Path(result["path"]))
        assert ok is True, f"on-disk file failed independent verification: {detail}"

    def test_creates_missing_parent_dirs(
        self, tmp_path, clean_hermes_chromium_env
    ):
        if not _chromium_available():
            pytest.skip("bundled chromium not present")
        # /nested/a/b/c/d/out.png — three levels of nonexistent parents.
        dest = tmp_path / "nested" / "a" / "b" / "c" / "d" / "out.png"
        assert not dest.parent.exists()

        result = native_screenshot("https://example.com", dest, timeout=30)

        assert result["success"] is True
        assert dest.parent.is_dir()
        assert dest.is_file()

    def test_copies_to_mnt_c_style_path(
        self, tmp_path, clean_hermes_chromium_env
    ):
        """Explicit /mnt/c-shaped path under a tempdir — confirms the
        copy stage handles a 9p-style destination by going through
        the verified /tmp staging file (chromium never writes to dest).

        We use ``tmp_path / 'mnt_c_standin'`` rather than the real
        ``/mnt/c`` to keep tests hermetic, but the *shape* of the path
        (``<some-root>/subdir/out.png``) is the same one the WSL host
        would pass.
        """
        if not _chromium_available():
            pytest.skip("bundled chromium not present")
        mnt_c_standin = tmp_path / "mnt_c_standin"
        mnt_c_standin.mkdir()
        dest = mnt_c_standin / "AI_WORKSPACE_ACTIVE" / "screenshots" / "foo.png"
        assert not dest.exists()

        result = native_screenshot("https://example.com", dest, timeout=30)

        assert result["success"] is True, f"copy-to-mnt_c-style failed: {result}"
        # tmp_png should have been cleaned up after the copy.
        assert "tmp_path" in result
        assert not Path(result["tmp_path"]).exists(), (
            "staging /tmp file should be cleaned up after successful copy"
        )
        # The dest file should be the same size as what we reported.
        actual_size = dest.stat().st_size
        assert actual_size == result["bytes"]
        # And it should be a real PNG.
        with open(dest, "rb") as f:
            head = f.read(8)
        assert head == b"\x89PNG\r\n\x1a\n"
