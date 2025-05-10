#!/usr/bin/env python3

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

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
PORT = os.getenv("NLP_CLIENT_PORT", "8006")
server_session = None
tools = []
server_params = StdioServerParameters(command="python", args=["server/tuya_server.py"])
SERVER_TIMEOUT = 30


# Configure Gemini
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
model = "gemini-2.0-flash"

def ask_gemini_for_tool(user_input, tools, device_mappings=None, command_count=0, conversation_history=None):
    if conversation_history is None:
        conversation_history = []
        
    # Start with the initial prompt
    prompt = f"""You are a home automation assistant that can call any MCP tool. The available tools are:
{json.dumps(tools, indent=2)}

Given the user's request, you must follow these rules:

1. For light control requests (on/off/dim), you MUST follow this exact two-step process:
   a. First call: {{"tool": "get_device_mappings", "arguments": {{}}}}
   b. Second call: {{"tool": "control_light", "arguments": {{"device_id": "device_id_from_mappings", "command": "on/off"}}}}

2. For other requests, use the appropriate tool directly.

3. You can handle multiple commands in a single request. For each command:
   - If it's a light control command, follow the two-step process
   - If it's another type of command, use the appropriate tool directly

4. CRITICAL: After processing ALL commands in the user's request, you MUST return {{"tool": "done", "arguments": {{}}}} to signal completion.
   - Do NOT make any more tool calls after returning "done"
   - Do NOT make any tool calls if you've already processed all commands
   - Do NOT make any tool calls if you've already returned "done"

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
        return json.loads(json_str)
    except Exception as e:
        logger.error(f"Failed to parse Gemini output: {e}")
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app"""
    # Startup
    logger.info("Starting application lifespan...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            global server_session
            await session.initialize()
            server_session = session
            tools_result = await session.list_tools()
            global tools
            for tool in tools_result.tools:
                tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema
                })
            yield

async def call_mcp_tool(tool_name, arguments):
    if server_session is None:
        return {"status": "error", "message": "Server session not initialized"}
    result: CallToolResult = await server_session.call_tool(tool_name, arguments=arguments)
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
        
        # First, get the initial tool call from Gemini
        gemini_result = ask_gemini_for_tool(request.command, tools, conversation_history=conversation_history)
        if not gemini_result or "tool" not in gemini_result or "arguments" not in gemini_result:
            raise HTTPException(status_code=400, detail="Could not parse tool/arguments from Gemini")
        
        # Store device mappings if we get them
        device_mappings = None
        results = []
        command_count = 0
        MAX_COMMANDS = 10  # Safety limit
        
        # Process the first tool call
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
        
        # If this was a get_device_mappings call, store the mappings and make the control_light call
        if tool_name == "get_device_mappings":
            if isinstance(mcp_result, str):
                result = json.loads(mcp_result)
            else:
                result = mcp_result
                
            if result["status"] == "success":
                device_mappings = result["mappings"]
                
                # Now get the next tool call from Gemini with the mappings
                while command_count < MAX_COMMANDS:
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
                    if tool_name == "done":
                        break
                        
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
                    
                    # If we've hit the command limit, force a "done" response
                    if command_count >= MAX_COMMANDS:
                        break
        
        # Return the last result or all results if there were multiple
        if len(results) > 1:
            return {"status": "success", "results": results}
        return results[-1] if results else {"status": "error", "message": "No results"}
    except Exception as e:
        logger.error(f"Error processing command: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(PORT)) 