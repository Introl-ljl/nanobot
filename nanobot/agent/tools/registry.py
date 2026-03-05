"""Tool registry for dynamic tool management."""

import asyncio
from collections.abc import Mapping
from typing import Any

from nanobot.agent.tools.base import Tool


class ToolRegistry:
    """
    Registry for agent tools.

    Allows dynamic registration and execution of tools.
    """

    def __init__(self, default_timeout: int = 30):
        self._tools: dict[str, Tool] = {}
        self._default_timeout = default_timeout

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions in OpenAI format."""
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute(self, name: str, params: dict[str, Any]) -> str:
        """Execute a tool by name with given parameters."""
        _hint = "\n\n[Analyze the error above and try a different approach.]"

        tool = self._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found. Available: {', '.join(self.tool_names)}"

        try:
            if not isinstance(params, Mapping):
                return (
                    f"Error: Invalid parameters for tool '{name}': expected object"
                    + _hint
                )

            tool_params = dict(params)
            tool_schema = tool.parameters if isinstance(tool.parameters, dict) else {}
            tool_properties = tool_schema.get("properties", {})
            tool_accepts_timeout = isinstance(tool_properties, dict) and "timeout" in tool_properties
            tool_manages_timeout = tool.manages_own_timeout

            timeout = self._default_timeout
            if not tool_accepts_timeout:
                # Reserve top-level timeout override only for tools that do not define
                # timeout in their own schema.
                timeout = tool_params.pop("timeout", self._default_timeout)

            errors = tool.validate_params(tool_params)
            if errors:
                return f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors) + _hint

            if tool_accepts_timeout or tool_manages_timeout:
                result = await tool.execute(**tool_params)
            else:
                try:
                    result = await asyncio.wait_for(
                        tool.execute(**tool_params),
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    return f"Error: Tool '{name}' timed out after {timeout} seconds" + _hint

            if isinstance(result, str) and result.startswith("Error"):
                return result + _hint
            return result
        except Exception as e:
            return f"Error executing {name}: {str(e)}" + _hint

    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
