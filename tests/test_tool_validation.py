import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool


class SampleTool(Tool):
    @property
    def name(self) -> str:
        return "sample"

    @property
    def description(self) -> str:
        return "sample tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 2},
                "count": {"type": "integer", "minimum": 1, "maximum": 10},
                "mode": {"type": "string", "enum": ["fast", "full"]},
                "meta": {
                    "type": "object",
                    "properties": {
                        "tag": {"type": "string"},
                        "flags": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["tag"],
                },
            },
            "required": ["query", "count"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return "ok"


class TimeoutRequiredTool(Tool):
    @property
    def name(self) -> str:
        return "timeout_required"

    @property
    def description(self) -> str:
        return "tool with required timeout parameter"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "timeout": {"type": "integer", "minimum": 1},
            },
            "required": ["timeout"],
        }

    async def execute(self, timeout: int, **kwargs: Any) -> str:
        return f"timeout={timeout}"


class SlowToolWithOwnTimeoutParam(Tool):
    @property
    def name(self) -> str:
        return "slow_with_timeout_param"

    @property
    def description(self) -> str:
        return "slow tool that owns timeout param semantics"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "timeout": {"type": "integer", "minimum": 1},
                "delay": {"type": "number"},
            },
        }

    async def execute(self, timeout: int = 1, delay: float = 0.03, **kwargs: Any) -> str:
        await asyncio.sleep(delay)
        return f"inner-timeout={timeout}"


class SlowToolWithoutTimeoutParam(Tool):
    @property
    def name(self) -> str:
        return "slow_no_timeout_param"

    @property
    def description(self) -> str:
        return "slow tool without timeout schema"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "delay": {"type": "number"},
            },
        }

    async def execute(self, delay: float = 0.2, **kwargs: Any) -> str:
        await asyncio.sleep(delay)
        return "done"


class InternallyTimedTool(Tool):
    @property
    def name(self) -> str:
        return "internally_timed"

    @property
    def description(self) -> str:
        return "tool that owns timeout handling without exposing timeout schema"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "delay": {"type": "number"},
            },
        }

    @property
    def manages_own_timeout(self) -> bool:
        return True

    async def execute(self, delay: float = 0.03, **kwargs: Any) -> str:
        await asyncio.sleep(delay)
        return "managed-timeout-ok"


def test_validate_params_missing_required() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi"})
    assert "missing required count" in "; ".join(errors)


def test_validate_params_type_and_range() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi", "count": 0})
    assert any("count must be >= 1" in e for e in errors)

    errors = tool.validate_params({"query": "hi", "count": "2"})
    assert any("count should be integer" in e for e in errors)


def test_validate_params_enum_and_min_length() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "h", "count": 2, "mode": "slow"})
    assert any("query must be at least 2 chars" in e for e in errors)
    assert any("mode must be one of" in e for e in errors)


def test_validate_params_nested_object_and_array() -> None:
    tool = SampleTool()
    errors = tool.validate_params(
        {
            "query": "hi",
            "count": 2,
            "meta": {"flags": [1, "ok"]},
        }
    )
    assert any("missing required meta.tag" in e for e in errors)
    assert any("meta.flags[0] should be string" in e for e in errors)


def test_validate_params_ignores_unknown_fields() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi", "count": 2, "extra": "x"})
    assert errors == []


async def test_registry_returns_validation_error() -> None:
    reg = ToolRegistry()
    reg.register(SampleTool())
    result = await reg.execute("sample", {"query": "hi"})
    assert "Invalid parameters" in result


async def test_registry_preserves_timeout_param_when_tool_schema_uses_it() -> None:
    reg = ToolRegistry(default_timeout=1)
    reg.register(TimeoutRequiredTool())
    result = await reg.execute("timeout_required", {"timeout": 3})
    assert result == "timeout=3"


async def test_registry_does_not_wrap_timeout_aware_tools_with_global_wait_for() -> None:
    reg = ToolRegistry(default_timeout=0.01)
    reg.register(SlowToolWithOwnTimeoutParam())
    result = await reg.execute("slow_with_timeout_param", {"timeout": 5, "delay": 0.03})
    assert result == "inner-timeout=5"


async def test_registry_keeps_top_level_timeout_override_for_timeout_unaware_tools() -> None:
    reg = ToolRegistry(default_timeout=1)
    reg.register(SlowToolWithoutTimeoutParam())
    result = await reg.execute("slow_no_timeout_param", {"delay": 0.2, "timeout": 0.01})
    assert "timed out after 0.01 seconds" in result


async def test_registry_skips_global_wait_for_when_tool_manages_own_timeout() -> None:
    reg = ToolRegistry(default_timeout=0.01)
    reg.register(InternallyTimedTool())
    result = await reg.execute("internally_timed", {"delay": 0.03, "timeout": 999})
    assert result == "managed-timeout-ok"


def test_exec_extract_absolute_paths_keeps_full_windows_path() -> None:
    cmd = r"type C:\user\workspace\txt"
    paths = ExecTool._extract_absolute_paths(cmd)
    assert paths == [r"C:\user\workspace\txt"]


def test_exec_extract_absolute_paths_ignores_relative_posix_segments() -> None:
    cmd = ".venv/bin/python script.py"
    paths = ExecTool._extract_absolute_paths(cmd)
    assert "/bin/python" not in paths


def test_exec_extract_absolute_paths_captures_posix_absolute_paths() -> None:
    cmd = "cat /tmp/data.txt > /tmp/out.txt"
    paths = ExecTool._extract_absolute_paths(cmd)
    assert "/tmp/data.txt" in paths
    assert "/tmp/out.txt" in paths


async def test_exec_cancellation_cleans_up_subprocess() -> None:
    tool = ExecTool(timeout=10)
    process = MagicMock()
    process.returncode = None
    process.communicate = AsyncMock(side_effect=asyncio.CancelledError())
    process.wait = AsyncMock(return_value=0)

    with patch("nanobot.agent.tools.shell.asyncio.create_subprocess_shell", AsyncMock(return_value=process)):
        with pytest.raises(asyncio.CancelledError):
            await tool.execute(command="sleep 100")

    process.kill.assert_called_once()
    process.wait.assert_awaited_once()
