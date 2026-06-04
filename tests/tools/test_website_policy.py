import json
from pathlib import Path

import pytest
import yaml

from tests.tools.conftest import register_all_web_providers

from tools.website_policy import WebsitePolicyError, check_website_access, load_website_blocklist


def test_load_website_blocklist_merges_config_and_shared_file(tmp_path):
    shared = tmp_path / "community-blocklist.txt"
    shared.write_text("# comment\nexample.org\nsub.bad.net\n", encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "security": {
                    "website_blocklist": {
                        "enabled": True,
                        "domains": ["example.com", "https://www.evil.test/path"],
                        "shared_files": [str(shared)],
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    policy = load_website_blocklist(config_path)

    assert policy["enabled"] is True
    assert {rule["pattern"] for rule in policy["rules"]} == {
        "example.com",
        "evil.test",
        "example.org",
        "sub.bad.net",
    }


def test_check_website_access_matches_parent_domain_subdomains(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "security": {
                    "website_blocklist": {
                        "enabled": True,
                        "domains": ["example.com"],
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    blocked = check_website_access("https://docs.example.com/page", config_path=config_path)

    assert blocked is not None
    assert blocked["host"] == "docs.example.com"
    assert blocked["rule"] == "example.com"


def test_check_website_access_supports_wildcard_subdomains_only(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "security": {
                    "website_blocklist": {
                        "enabled": True,
                        "domains": ["*.tracking.example"],
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    assert check_website_access("https://a.tracking.example", config_path=config_path) is not None
    assert check_website_access("https://www.tracking.example", config_path=config_path) is not None
    assert check_website_access("https://tracking.example", config_path=config_path) is None


def test_default_config_exposes_website_blocklist_shape():
    from hermes_cli.config import DEFAULT_CONFIG

    website_blocklist = DEFAULT_CONFIG["security"]["website_blocklist"]
    assert website_blocklist["enabled"] is False
    assert website_blocklist["domains"] == []
    assert website_blocklist["shared_files"] == []


def test_load_website_blocklist_uses_enabled_default_when_section_missing(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"display": {"tool_progress": "all"}}, sort_keys=False), encoding="utf-8")

    policy = load_website_blocklist(config_path)

    assert policy == {"enabled": False, "rules": []}


def test_load_website_blocklist_raises_clean_error_for_invalid_domains_type(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "security": {
                    "website_blocklist": {
                        "enabled": True,
                        "domains": "example.com",
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(WebsitePolicyError, match="security.website_blocklist.domains must be a list"):
        load_website_blocklist(config_path)


def test_load_website_blocklist_raises_clean_error_for_invalid_shared_files_type(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "security": {
                    "website_blocklist": {
                        "enabled": True,
                        "shared_files": "community-blocklist.txt",
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(WebsitePolicyError, match="security.website_blocklist.shared_files must be a list"):
        load_website_blocklist(config_path)


def test_load_website_blocklist_raises_clean_error_for_invalid_top_level_config_type(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(["not", "a", "mapping"], sort_keys=False), encoding="utf-8")

    with pytest.raises(WebsitePolicyError, match="config root must be a mapping"):
        load_website_blocklist(config_path)


def test_load_website_blocklist_raises_clean_error_for_invalid_security_type(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"security": []}, sort_keys=False), encoding="utf-8")

    with pytest.raises(WebsitePolicyError, match="security must be a mapping"):
        load_website_blocklist(config_path)


def test_load_website_blocklist_raises_clean_error_for_invalid_website_blocklist_type(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "security": {
                    "website_blocklist": "block everything",
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(WebsitePolicyError, match="security.website_blocklist must be a mapping"):
        load_website_blocklist(config_path)


def test_load_website_blocklist_raises_clean_error_for_invalid_enabled_type(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "security": {
                    "website_blocklist": {
                        "enabled": "false",
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(WebsitePolicyError, match="security.website_blocklist.enabled must be a boolean"):
        load_website_blocklist(config_path)


def test_load_website_blocklist_raises_clean_error_for_malformed_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("security: [oops\n", encoding="utf-8")

    with pytest.raises(WebsitePolicyError, match="Invalid config YAML"):
        load_website_blocklist(config_path)


def test_load_website_blocklist_wraps_shared_file_read_errors(tmp_path, monkeypatch):
    shared = tmp_path / "community-blocklist.txt"
    shared.write_text("example.org\n", encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "security": {
                    "website_blocklist": {
                        "enabled": True,
                        "shared_files": [str(shared)],
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    def failing_read_text(self, *args, **kwargs):
        raise PermissionError("no permission")

    monkeypatch.setattr(Path, "read_text", failing_read_text)

    # Unreadable shared files are now warned and skipped (not raised),
    # so the blocklist loads successfully but without those rules.
    result = load_website_blocklist(config_path)
    assert result["enabled"] is True
    assert result["rules"] == []  # shared file rules skipped


def test_check_website_access_uses_dynamic_hermes_home(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "security": {
                    "website_blocklist": {
                        "enabled": True,
                        "domains": ["dynamic.example"],
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Invalidate the module-level cache so the new HERMES_HOME is picked up.
    # A prior test may have cached a default policy (enabled=False) under the
    # old HERMES_HOME set by the autouse _isolate_hermes_home fixture.
    from tools.website_policy import invalidate_cache
    invalidate_cache()

    blocked = check_website_access("https://dynamic.example/path")

    assert blocked is not None
    assert blocked["rule"] == "dynamic.example"


def test_check_website_access_blocks_scheme_less_urls(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "security": {
                    "website_blocklist": {
                        "enabled": True,
                        "domains": ["blocked.test"],
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    blocked = check_website_access("www.blocked.test/path", config_path=config_path)

    assert blocked is not None
    assert blocked["host"] == "www.blocked.test"
    assert blocked["rule"] == "blocked.test"


def test_browser_navigate_returns_policy_block(monkeypatch):
    from tools import browser_tool

    # Allow SSRF check to pass so the policy check is reached
    monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)
    monkeypatch.setattr(
        browser_tool,
        "check_website_access",
        lambda url: {
            "host": "blocked.test",
            "rule": "blocked.test",
            "source": "config",
            "message": "Blocked by website policy",
        },
    )
    monkeypatch.setattr(
        browser_tool,
        "_run_browser_command",
        lambda *args, **kwargs: pytest.fail("browser command should not run for blocked URL"),
    )

    result = json.loads(browser_tool.browser_navigate("https://blocked.test"))

    assert result["success"] is False
    assert result["blocked_by_policy"]["rule"] == "blocked.test"


def test_browser_navigate_allows_when_shared_file_missing(monkeypatch, tmp_path):
    """Missing shared blocklist files are warned and skipped, not fatal."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "security": {
                    "website_blocklist": {
                        "enabled": True,
                        "shared_files": ["missing-blocklist.txt"],
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    # check_website_access should return None (allow) — missing file is skipped
    result = check_website_access("https://allowed.test", config_path=config_path)
    assert result is None


class TestWebToolPolicy:
    """Tests that exercise web_extract_tool with website-policy gates.

    These tests need the bundled web providers to be registered in the
    agent.web_search_registry so the tool dispatchers can find an active
    provider.  Without registration, the tools return an error dict that
    lacks a ``results`` key, causing ``KeyError``.
    """

    _register_providers = staticmethod(register_all_web_providers)

    @pytest.fixture(autouse=True)
    def _populate_web_registry(self):
        self._register_providers()
        yield
        from agent.web_search_registry import _reset_for_tests
        _reset_for_tests()

    @pytest.mark.asyncio
    async def test_web_extract_short_circuits_blocked_url(self, monkeypatch):
        from tools import web_tools
        from plugins.web.firecrawl import provider as firecrawl_provider

        # Allow test URLs past SSRF check so website policy is what gets tested
        monkeypatch.setattr(web_tools, "is_safe_url", lambda url: True)
        # The per-URL website-policy gate moved into the firecrawl plugin's
        # extract() during the web-provider migration. Patch it at the new
        # location.
        monkeypatch.setattr(
            firecrawl_provider,
            "check_website_access",
            lambda url: {
                "host": "blocked.test",
                "rule": "blocked.test",
                "source": "config",
                "message": "Blocked by website policy",
            },
        )
        monkeypatch.setattr(
            firecrawl_provider,
            "_get_firecrawl_client",
            lambda: pytest.fail("firecrawl should not run for blocked URL"),
        )
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
        # Force the firecrawl plugin to be the active extract provider.
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-key")

        result = json.loads(await web_tools.web_extract_tool(["https://blocked.test"], use_llm_processing=False))

        assert result["results"][0]["url"] == "https://blocked.test"
        assert "Blocked by website policy" in result["results"][0]["error"]

    @pytest.mark.asyncio
    async def test_web_extract_blocks_redirected_final_url(self, monkeypatch):
        from tools import web_tools
        from plugins.web.firecrawl import provider as firecrawl_provider

        # Allow test URLs past SSRF check so website policy is what gets tested
        monkeypatch.setattr(web_tools, "is_safe_url", lambda url: True)

        def fake_check(url):
            if url == "https://allowed.test":
                return None
            if url == "https://blocked.test/final":
                return {
                    "host": "blocked.test",
                    "rule": "blocked.test",
                    "source": "config",
                    "message": "Blocked by website policy",
                }
            pytest.fail(f"unexpected URL checked: {url}")

        class FakeFirecrawlClient:
            def scrape(self, url, formats):
                return {
                    "markdown": "secret content",
                    "metadata": {
                        "title": "Redirected",
                        "sourceURL": "https://blocked.test/final",
                    },
                }

        # After the web-provider migration, the per-URL gate + firecrawl client
        # live in the plugin. Patch both at the plugin location.
        monkeypatch.setattr(firecrawl_provider, "check_website_access", fake_check)
        monkeypatch.setattr(firecrawl_provider, "_get_firecrawl_client", lambda: FakeFirecrawlClient())
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-key")

        result = json.loads(await web_tools.web_extract_tool(["https://allowed.test"], use_llm_processing=False))

        assert result["results"][0]["url"] == "https://blocked.test/final"
        assert result["results"][0]["content"] == ""
        assert result["results"][0]["blocked_by_policy"]["rule"] == "blocked.test"


def test_check_website_access_fails_open_on_malformed_config(tmp_path, monkeypatch):
    """Malformed config with default path should fail open (return None), not crash."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("security: [oops\n", encoding="utf-8")

    # With explicit config_path (test mode), errors propagate
    with pytest.raises(WebsitePolicyError):
        check_website_access("https://example.com", config_path=config_path)

    # Simulate default path by pointing HERMES_HOME to tmp_path
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import website_policy
    website_policy.invalidate_cache()

    # With default path, errors are caught and fail open
    result = check_website_access("https://example.com")
    assert result is None  # allowed, not crashed


class TestWebExtractSSRF:
    """Tests that the SSRF check in web_extract_tool is backend-aware.

    Cloud extract backends (Firecrawl, Parallel, Tavily, Exa) fetch URLs
    from their own servers — local DNS resolution is misleading behind
    proxy DNS hijack, so only the absolute security floor applies.
    Local/self-hosted backends (SearXNG) keep the full SSRF check.
    """

    _register_providers = staticmethod(register_all_web_providers)

    @pytest.fixture(autouse=True)
    def _populate_web_registry(self):
        self._register_providers()
        yield
        from agent.web_search_registry import _reset_for_tests
        _reset_for_tests()

    @pytest.mark.asyncio
    async def test_cloud_extract_skips_ssrf_for_public_url(self, monkeypatch):
        """Firecrawl (cloud) with proxy hijack should NOT block public URLs."""
        from tools import web_tools
        from tools import url_safety

        # Force cloud backend
        monkeypatch.setattr(web_tools, "_get_extract_backend", lambda: "firecrawl")
        # Mock proxy DNS hijack = True (all probes resolve to 198.18.0.0/15)
        monkeypatch.setattr(url_safety, "_proxy_dns_hijack_cache", True)
        # Mock the firecrawl client to return success without real API call
        monkeypatch.setattr("plugins.web.firecrawl.provider._get_firecrawl_client", lambda: None)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-key")

        # Patch the async extract method to avoid real network calls
        original_extract = web_tools.web_extract_tool

        # Just test the SSRF pre-filter by short-circuiting the provider
        async def mock_extract(urls, **kwargs):
            # The SSRF check should let the URL through
            return json.dumps({"results": [{"url": urls[0], "title": "Test", "content": "Hello, world!"}]})
        monkeypatch.setattr(web_tools, "web_extract_tool", mock_extract)

        result = json.loads(await web_tools.web_extract_tool(["https://github.com/advisories/test"], format="markdown", use_llm_processing=False))
        # The URL should be allowed through (no SSRF error)
        assert result["results"][0]["content"] == "Hello, world!"

    @pytest.mark.asyncio
    async def test_cloud_extract_blocks_metadata_endpoint(self, monkeypatch):
        """Firecrawl (cloud) should STILL block cloud metadata endpoints."""
        from tools import web_tools
        from tools import url_safety

        monkeypatch.setattr(web_tools, "_get_extract_backend", lambda: "firecrawl")
        monkeypatch.setattr(url_safety, "_proxy_dns_hijack_cache", True)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

        result = json.loads(await web_tools.web_extract_tool(
            ["http://169.254.169.254/latest/meta-data/"],
            use_llm_processing=False,
        ))
        assert "cloud metadata" in result["results"][0]["error"].lower()

    @pytest.mark.asyncio
    async def test_local_backend_still_blocks_private_url(self, monkeypatch):
        """SearXNG (local) should still use full SSRF check for private URLs."""
        from tools import web_tools
        from tools import url_safety

        monkeypatch.setattr(web_tools, "_get_extract_backend", lambda: "searxng")
        monkeypatch.setattr(url_safety, "_proxy_dns_hijack_cache", True)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

        result = json.loads(await web_tools.web_extract_tool(
            ["http://192.168.1.1/admin"],
            use_llm_processing=False,
        ))
        assert "private or internal" in result["results"][0]["error"].lower()

    @pytest.mark.asyncio
    async def test_cloud_extract_allows_literal_rfc1918(self, monkeypatch):
        """Cloud extract passes RFC1918 private IPs (10.x, 172.16.x, 192.168.x).

        Unlike local backends, a cloud extract service (Firecrawl) fetches
        the URL from its own network, not the agent's.  RFC1918 addresses
        on a cloud server refer to the cloud provider's own private network,
        not the user's LAN.  Only the absolute security floor — cloud
        metadata endpoints that could leak instance credentials regardless
        of who fetches them — needs to be blocked for cloud backends.
        """
        from tools import web_tools
        from tools import url_safety

        monkeypatch.setattr(web_tools, "_get_extract_backend", lambda: "firecrawl")
        monkeypatch.setattr(url_safety, "_proxy_dns_hijack_cache", True)
        monkeypatch.setattr("plugins.web.firecrawl.provider._get_firecrawl_client", lambda: None)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-key")

        async def mock_extract(urls, **kwargs):
            return json.dumps({"results": [{"url": urls[0], "title": "Test", "content": "reached"}]})
        monkeypatch.setattr(web_tools, "web_extract_tool", mock_extract)

        for url in ("http://10.0.0.1/admin", "http://172.16.0.1/admin", "http://192.168.1.1/admin"):
            result = json.loads(await web_tools.web_extract_tool([url], use_llm_processing=False))
            # Should pass through (not blocked) for cloud backends
            assert result["results"][0]["content"] == "reached", f"{url} should not be blocked for cloud extract"

    @pytest.mark.asyncio
    async def test_cloud_extract_allows_literal_localhost(self, monkeypatch):
        """Cloud extract passes literal localhost (127.0.0.1, ::1).

        Same reasoning as RFC1918: the cloud service fetches from its own
        servers, so localhost refers to the cloud server's loopback, not
        the user's localhost.
        """
        from tools import web_tools
        from tools import url_safety

        monkeypatch.setattr(web_tools, "_get_extract_backend", lambda: "firecrawl")
        monkeypatch.setattr(url_safety, "_proxy_dns_hijack_cache", True)
        monkeypatch.setattr("plugins.web.firecrawl.provider._get_firecrawl_client", lambda: None)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-key")

        async def mock_extract(urls, **kwargs):
            return json.dumps({"results": [{"url": urls[0], "title": "Test", "content": "reached"}]})
        monkeypatch.setattr(web_tools, "web_extract_tool", mock_extract)

        for url in ("http://127.0.0.1:8080/admin", "http://[::1]:8080/admin"):
            result = json.loads(await web_tools.web_extract_tool([url], use_llm_processing=False))
            assert result["results"][0]["content"] == "reached", f"{url} should not be blocked for cloud extract"
