#!/usr/bin/env python3
"""Test MCP stdio transport functionality"""

import asyncio
import json
import sys
from mcp.server import Server
from mcp import Resource, Tool


# Test MCP server with stdio transport
class TestMCPServer:
    def __init__(self):
        self.server = Server("test-mcp")
        self._setup_handlers()

    def _setup_handlers(self):
        @self.server.list_resources()
        async def handle_list_resources() -> list[Resource]:
            return [
                Resource(
                    uri="test://example",
                    name="Test Resource",
                    description="A test resource",
                    mimeType="text/plain"
                )
            ]
        
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            if uri == "test://example":
                return "This is a test resource content"
            else:
                raise ValueError(f"Unknown resource: {uri}")

        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            return [
                Tool(
                    name="test_tool",
                    description="A test tool",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "message": {"type": "string"}
                        },
                        "required": ["message"]
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list:
            if name == "test_tool":
                message = arguments.get("message", "No message provided")
                return [{"type": "text", "text": f"Echo: {message}"}]
            else:
                raise ValueError(f"Unknown tool: {name}")

    async def run_stdio(self):
        """Run the server using stdio transport"""
        async with self.server as server:
            await server.run(
                read_stream=asyncio.StreamReader(),
                write_stream=sys.stdout,
                initialization_options=InitializationOptions(
                    server_name="test-mcp",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions()
                    )
                )
            )


async def main():
    """Test the MCP stdio server"""
    print("Starting Test MCP Server with stdio transport...", file=sys.stderr)
    
    server = TestMCPServer()
    try:
        await server.run_stdio()
    except KeyboardInterrupt:
        print("\nShutting down server...", file=sys.stderr)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
