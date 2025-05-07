from mcp.server.fastmcp import FastMCP
# This is the shared MCP server instance
mcp = FastMCP("mix_server")

if __name__ == "__main__":
    mcp.run()