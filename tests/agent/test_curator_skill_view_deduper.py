"""Tests for CuratorSkillViewDeduper — skill_view deduplication guard.

Verifies that the curator's LLM consolidation pass cannot repeatedly call
skill_view on the same skill.  Uses the real deduper class directly and
patches the underlying skill resolution for fast execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def deduper():
    """Fresh deduper instance per test."""
    from tools.skills_tool import _CuratorSkillViewDeduper
    d = _CuratorSkillViewDeduper()
    return d


@pytest.fixture
def fake_skill_dirs(tmp_path: Path, monkeypatch):
    """Set up a minimal skills directory with one test skill and wire
    SKILLS_DIR in skills_tool to point at it."""
    skills = tmp_path / "skills"
    skills.mkdir()
    # Create tailscale-setup skill
    d = skills / "tailscale-setup"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: tailscale-setup\ndescription: test\n---\n# Tailscale Setup\n",
        encoding="utf-8",
    )
    # Create openclash-openwrt skill
    d2 = skills / "openclash-openwrt"
    d2.mkdir()
    (d2 / "SKILL.md").write_text(
        "---\nname: openclash-openwrt\ndescription: test\n---\n# OpenClash\n",
        encoding="utf-8",
    )
    # Patch SKILLS_DIR
    import tools.skills_tool
    monkeypatch.setattr(tools.skills_tool, "SKILLS_DIR", skills)
    # Patch get_external_skills_dirs at the import site used by skill_view
    monkeypatch.setattr(
        "agent.skill_utils.get_external_skills_dirs",
        lambda: [],
    )
    return skills


# ---------------------------------------------------------------------------
# Deduper unit tests
# ---------------------------------------------------------------------------

class TestCuratorSkillViewDeduper:

    def test_not_active_by_default(self, deduper):
        assert not deduper.is_active
        assert not deduper.should_skip("any-skill")

    def test_activate_then_deactivate(self, deduper):
        deduper.activate()
        assert deduper.is_active
        deduper.deactivate()
        assert not deduper.is_active

    def test_first_two_calls_allowed(self, deduper):
        deduper.activate()
        # 1st call → allowed
        assert not deduper.should_skip("tailscale-setup")
        # 2nd call → allowed
        assert not deduper.should_skip("tailscale-setup")

    def test_third_call_skipped(self, deduper):
        deduper.activate()
        deduper.should_skip("tailscale-setup")  # 1
        deduper.should_skip("tailscale-setup")  # 2
        assert deduper.should_skip("tailscale-setup")  # 3 → skip

    def test_fourth_call_also_skipped(self, deduper):
        deduper.activate()
        for _ in range(4):
            deduper.should_skip("tailscale-setup")
        # All beyond 2 should be True
        for _ in range(3):
            assert deduper.should_skip("tailscale-setup")

    def test_different_skills_independent(self, deduper):
        deduper.activate()
        # View tailscale-setup twice
        deduper.should_skip("tailscale-setup")
        deduper.should_skip("tailscale-setup")
        # openclash-openwrt should still be allowed
        assert not deduper.should_skip("openclash-openwrt")
        assert not deduper.should_skip("openclash-openwrt")
        # Third call on each should skip
        assert deduper.should_skip("tailscale-setup")
        assert deduper.should_skip("openclash-openwrt")

    def test_skipped_result_format(self, deduper):
        deduper.activate()
        deduper.should_skip("tailscale-setup")  # 1
        deduper.should_skip("tailscale-setup")  # 2
        deduper.should_skip("tailscale-setup")  # 3
        result = deduper.skipped_result("tailscale-setup")
        data = json.loads(result)
        assert data["success"] is True
        assert data["skipped"] is True
        assert "SKIPPED_DUPLICATE_SKILL_VIEW" in data["message"]
        assert "tailscale-setup" in data["message"]

    def test_inactive_deduper_never_skips(self, deduper):
        # Not activated — should always return False
        for _ in range(10):
            assert not deduper.should_skip("anything")

    def test_reactivate_resets_count(self, deduper):
        deduper.activate()
        deduper.should_skip("tailscale-setup")  # 1
        deduper.should_skip("tailscale-setup")  # 2
        assert deduper.should_skip("tailscale-setup")  # 3 → skip after reset
        deduper.deactivate()
        deduper.activate()
        # Counter should be reset
        assert not deduper.should_skip("tailscale-setup")

    def test_concurrent_threads_independent(self, deduper):
        """Thread-local state means two threads don't interfere."""
        import threading
        results = []

        deduper.activate()
        deduper.should_skip("tailscale-setup")  # 1 on main

        def worker():
            d2 = _make_fresh_deduper()
            d2.activate()
            results.append(d2.should_skip("tailscale-setup"))  # 1 on worker
            results.append(d2.should_skip("tailscale-setup"))  # 2 on worker
            results.append(d2.should_skip("tailscale-setup"))  # 3 → skip on worker

        def _make_fresh_deduper():
            from tools.skills_tool import _CuratorSkillViewDeduper
            return _CuratorSkillViewDeduper()

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        # Main thread should still be on view 2, not 3
        assert not deduper.should_skip("tailscale-setup")  # 2 on main
        # Worker's third call was skipped
        assert results == [False, False, True]


# ---------------------------------------------------------------------------
# Integration tests: skill_view + deduper with fake skill directories
# ---------------------------------------------------------------------------

class TestSkillViewWithDeduper:

    def test_real_skill_view_twice_then_skip(self, fake_skill_dirs, deduper):
        """skill_view should succeed twice, then return synthetic skip."""
        deduper.activate()

        with patch(
            "tools.skills_tool._curator_skill_view_deduper",
            deduper,
        ):
            # 1st call — real content
            r1 = _view("tailscale-setup")
            d1 = json.loads(r1)
            assert d1.get("success") is True

            # 2nd call — real content
            r2 = _view("tailscale-setup")
            d2 = json.loads(r2)
            assert d2.get("success") is True

            # 3rd call — synthetic skip (dedup)
            r3 = _view("tailscale-setup")
            d3 = json.loads(r3)
            assert d3.get("skipped") is True
            assert "SKIPPED_DUPLICATE_SKILL_VIEW" in d3.get("message", "")

    def test_different_skills_not_blocked(self, fake_skill_dirs, deduper):
        """Viewing tailscale-setup should not affect openclash-openwrt."""
        deduper.activate()
        with patch(
            "tools.skills_tool._curator_skill_view_deduper",
            deduper,
        ):
            # Exhaust tailscale-setup views
            for _ in range(3):
                _view("tailscale-setup")

            # openclash-openwrt should still be viewable
            r = _view("openclash-openwrt")
            d = json.loads(r)
            assert d.get("success") is True
            assert d.get("skipped") is not True

    def test_non_curator_skill_view_unaffected(self, fake_skill_dirs, deduper):
        """Without activation, deduper never fires — normal usage is untouched."""
        # deduper is NOT activated
        with patch(
            "tools.skills_tool._curator_skill_view_deduper",
            deduper,
        ):
            for _ in range(5):
                r = _view("tailscale-setup")
                d = json.loads(r)
                assert d.get("success") is True
                assert d.get("skipped") is not True

    def test_curator_functions_integration(self, fake_skill_dirs):
        """Test the public activate/deactivate functions work with real skill_view."""
        from tools.skills_tool import (
            curator_activate_skill_view_deduper,
            curator_deactivate_skill_view_deduper,
            curator_skill_view_deduper_is_active,
            skill_view,
        )

        assert not curator_skill_view_deduper_is_active()

        curator_activate_skill_view_deduper()
        assert curator_skill_view_deduper_is_active()

        # View twice normally
        r1 = json.loads(skill_view("tailscale-setup"))
        assert r1.get("success") is True

        r2 = json.loads(skill_view("tailscale-setup"))
        assert r2.get("success") is True

        # Third view should skip
        r3 = json.loads(skill_view("tailscale-setup"))
        assert r3.get("skipped") is True
        assert "SKIPPED_DUPLICATE_SKILL_VIEW" in r3.get("message", "")

        curator_deactivate_skill_view_deduper()
        assert not curator_skill_view_deduper_is_active()

        # After deactivation, viewing works again
        r4 = json.loads(skill_view("tailscale-setup"))
        assert r4.get("success") is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _view(name: str) -> str:
    """Call skill_view with minimal arguments."""
    from tools.skills_tool import skill_view
    return skill_view(name)
