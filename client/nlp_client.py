#!/usr/bin/env python3

import contextlib
import os
import json
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging
import asyncio
from contextlib import asynccontextmanager
from mcp.types import CallToolResult
from google import genai
from typing import NotRequired, TypedDict, List, Dict

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class ServerParams(TypedDict):
    command: str
    args: List[str]
    env: NotRequired[Dict[str, str]]

# Configuration
PORT = os.getenv("NLP_CLIENT_PORT", "8006")
server_sessions = {}
tools = []
server_params: Dict[str, ServerParams] = {
    "tuya": {
        "command": "python",
        "args": ["server/tuya_server.py"]
    },
    "browsermcp": {
        "command": "npx",
        "args": ["@browsermcp/mcp@latest"]
    },
    "websearch": {
        "command": "npx",
        "args": ["websearch-mcp"],
        "env": {
            "API_URL": "http://localhost:3001",
            "MAX_SEARCH_RESULT": "5" 
        }
    }
}


# Configure Gemini
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
model = "gemini-2.0-flash"


def ask_gemini_for_tool(user_input, tools, device_mappings=None,
                        command_count=0, conversation_history=None):
    if conversation_history is None:
        conversation_history = []

    # Start with the initial prompt
    prompt = f"""You are a home automation assistant that can call any MCP tool. The available tools are:
{json.dumps(tools, indent=2)}

Given the user's request, you must follow these rules:

1. For light control requests (on/off/dim), you MUST follow this exact two-step process:
   a. First call: {{"tool": "tuya/get_device_mappings", "arguments": {{}}}}
   b. Second call: {{"tool": "tuya/control_light", "arguments": {{"device_id": "device_id_from_mappings", "command": "on/off"}}}}

2. For other requests, use the appropriate tool directly.

3. You can handle multiple commands in a single request. For each command:
   - If it's a light control command, follow the two-step process
   - If it's another type of command, use the appropriate tool directly
   - Always use one tool call at a time.

4. CRITICAL: After processing ALL commands in the user's request, you MUST return {{"tool": "done", "arguments": {{"message": "message_to_user"}}}} to signal completion.
   - Do NOT make any more tool calls after returning "done"
   - Do NOT make any tool calls if you've already processed all commands
   - Do NOT make any tool calls if you've already returned "done"
   - The message_to_user should be the final message to the user.
   - If the request just requires information, set message_to_user to the answer.

Let's start with the user's request:
Human: {user_input}
"""

    # Add conversation history
    for entry in conversation_history:
        if entry["role"] == "assistant":
            prompt += f"\nAssistant: {json.dumps(entry['content'])}"
        else:
            prompt += f"\nSystem: {json.dumps(entry['content'])}"

    # Add device mappings if available
    if device_mappings:
        prompt += f"""
System: Here are the device mappings you requested:
{json.dumps(device_mappings, indent=2)}

Please use these mappings to control the lights.
"""

    prompt += "\nAssistant:"

    response = client.models.generate_content(contents=[prompt], model=model)
    try:
        if not response.text:
            return None
        start = response.text.find('{')
        end = response.text.rfind('}') + 1
        json_str = response.text[start:end]
        logger.info(f"Gemini output: {json_str}")
        return json.loads(json_str)
    except Exception as e:
        logger.error(f"Failed to parse Gemini output: {e}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app"""
    # Startup
    logger.info("Starting application lifespan...")

    async with contextlib.AsyncExitStack() as stack:
        # First create all stdio clients
        stdio_pairs = {}
        for name, params in server_params.items():
            stdio_client_ctx = stdio_client(StdioServerParameters(**params))
            stdio_pairs[name] = await stack.enter_async_context(stdio_client_ctx)

        for name, (read, write) in stdio_pairs.items():
            session_ctx = ClientSession(read, write)
            session = await stack.enter_async_context(session_ctx)
            await session.initialize()
            server_sessions[name] = session
            tools_result = await session.list_tools()

            global tools
            # Add Tuya tools
            for tool in tools_result.tools:
                tools.append({
                    "name": f"{name}/{tool.name}",
                    "description": tool.description,
                    "inputSchema": tool.inputSchema
                })
        yield


async def call_mcp_tool(tool_name, arguments):
    print(tool_name)
    if server_sessions is None:
        return {"status": "error", "message": "Server session not initialized"}
    if tool_name == "done":
        return json.dumps({"status": "success", "arguments": arguments})
    server_name = tool_name.split("/")[0]
    tool = tool_name.split("/")[1]
    result: CallToolResult = await server_sessions[server_name].call_tool(tool, arguments=arguments)
    from mcp.types import TextContent
    content0 = result.content[0]
    if isinstance(content0, TextContent):
        return content0.text
    return str(content0)

# Global server session
server_lock = asyncio.Lock()


class CommandRequest(BaseModel):
    command: str


# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)


@app.post("/command")
async def process_command(request: CommandRequest):
    """Process a natural language command"""
    try:
        conversation_history = []
        logger.info(f"Processing command: {request.command}")

        # Store device mappings if we get them
        device_mappings = None
        results = []
        command_count = 0
        MAX_COMMANDS = 10  # Safety limit

        while command_count < MAX_COMMANDS:
            # Get tool call from Gemini
            gemini_result = ask_gemini_for_tool(
                request.command,
                tools,
                device_mappings,
                command_count,
                conversation_history
            )
            logger.info(f"Command {command_count}: {gemini_result}")

            if not gemini_result or "tool" not in gemini_result or "arguments" not in gemini_result:
                break

            tool_name = gemini_result["tool"]
            arguments = gemini_result["arguments"]
            mcp_result = await call_mcp_tool(tool_name, arguments)

            # Add the tool call and its result to conversation history
            conversation_history.append({
                "role": "assistant",
                "content": {"tool": tool_name, "arguments": arguments}
            })
            conversation_history.append({
                "role": "system",
                "content": mcp_result
            })

            results.append(mcp_result)
            command_count += 1

            # If this was a get_device_mappings call, store the mappings
            if tool_name == "tuya/get_device_mappings":
                if isinstance(mcp_result, str):
                    result = json.loads(mcp_result)
                else:
                    result = mcp_result
                if result["status"] == "success":
                    device_mappings = result["mappings"]

            # If we've hit the command limit or received done, break
            if command_count >= MAX_COMMANDS or tool_name == "done":
                break

        # Return the last result
        return {"status": "success", "message": json.loads(results[-1])["arguments"]["message"]} if results else {
            "status": "error", "message": "No results"}
    except Exception as e:
        logger.error(f"Error processing command: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(PORT))
