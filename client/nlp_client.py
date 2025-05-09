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
model = "gemini-2.5-flash-preview-04-17"

def ask_gemini_for_tool(user_input, tools, device_mappings=None):
    prompt = f"""
You are a home automation assistant that can call any MCP tool. The available tools are:
{json.dumps(tools, indent=2)}

Given the user's request, you must follow these rules:

1. For light control requests (on/off/dim), you MUST follow this exact two-step process:
   a. First call: {{"tool": "get_device_mappings", "arguments": {{}}}}
   b. Second call: {{"tool": "control_light", "arguments": {{"device_id": "device_id_from_mappings", "command": "on/off"}}}}

2. For other requests, use the appropriate tool directly.

Examples:
1. For "turn off living room light":
   First response: {{"tool": "get_device_mappings", "arguments": {{}}}}
   Second response: {{"tool": "control_light", "arguments": {{"device_id": "eb585e7b9c3a346ab4mcwb_1", "command": "off"}}}}

2. For "turn on bedroom light":
   First response: {{"tool": "get_device_mappings", "arguments": {{}}}}
   Second response: {{"tool": "control_light", "arguments": {{"device_id": "eb43453d39775b28a7vnsh", "command": "on"}}}}

3. For "list devices":
   Response: {{"tool": "list_devices", "arguments": {{}}}}
"""

    if device_mappings:
        prompt += f"""
IMPORTANT: You have just received the device mappings. Now you MUST use the control_light tool with the appropriate device ID.
Available device mappings:
{json.dumps(device_mappings, indent=2)}

For the living room light, use device_id: {device_mappings['sala']['id']}
For the bedroom light, use device_id: {device_mappings['quarto']['id']}
For the door light, use device_id: {device_mappings['porta']['id']}
For the backlight, use device_id: {device_mappings['backlight']['id']}
"""

    prompt += f"\nUser request: {user_input}\nJSON:"
    
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
        # First, get the initial tool call from Gemini
        gemini_result = ask_gemini_for_tool(request.command, tools)
        if not gemini_result or "tool" not in gemini_result or "arguments" not in gemini_result:
            raise HTTPException(status_code=400, detail="Could not parse tool/arguments from Gemini")
        
        # Store device mappings if we get them
        device_mappings = None
        
        # Process the first tool call
        tool_name = gemini_result["tool"]
        arguments = gemini_result["arguments"]
        mcp_result = await call_mcp_tool(tool_name, arguments)
        
        # If this was a get_device_mappings call, store the mappings and make the control_light call
        if tool_name == "get_device_mappings":
            if isinstance(mcp_result, str):
                result = json.loads(mcp_result)
            else:
                result = mcp_result
                
            if result["status"] == "success":
                device_mappings = result["mappings"]
                
                # Now get the next tool call from Gemini with the mappings
                gemini_result = ask_gemini_for_tool(request.command, tools, device_mappings)
                
                if gemini_result and "tool" in gemini_result and "arguments" in gemini_result:
                    tool_name = gemini_result["tool"]
                    arguments = gemini_result["arguments"]
                    mcp_result = await call_mcp_tool(tool_name, arguments)
        
        if isinstance(mcp_result, str):
            result = json.loads(mcp_result)
        else:
            result = mcp_result
            
        return result
    except Exception as e:
        logger.error(f"Error processing command: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(PORT)) 