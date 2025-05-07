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
    level=logging.DEBUG,  # Changed to DEBUG for more detailed logs
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

DEVICE_MAPPINGS = {
    "quarto": {
        "id": "eb43453d39775b28a7vnsh",
        "aliases": ["quarto", "bedroom", "room", "bed room", "bed-room"]
    },
    "sala": {
        "id": "eb585e7b9c3a346ab4mcwb_1",
        "aliases": ["sala", "living room", "livingroom", "living-room", "lounge"]
    },
    "porta": {
        "id": "eb585e7b9c3a346ab4mcwb_2",
        "aliases": ["porta", "door", "entrance"]
    },
    "backlight": {
        "id": "eb585e7b9c3a346ab4mcwb_16",
        "aliases": ["backlight", "back light", "back-light", "ambient"]
    }
}



def ask_gemini_for_tool(user_input, tools):
    prompt = f"""
You are a home automation assistant that can call any MCP tool. The available tools are:
{json.dumps(tools, indent=2)}

If it's about lighting, you should use these devices:
{json.dumps(DEVICE_MAPPINGS, indent=2)}

Given the user's request, output a JSON object with:
- tool: the tool name to use
- arguments: a dictionary of arguments for the tool

Example:
{{
  "tool": "control_light",
  "arguments": {{"device_id": "living_room", "command": "on", "brightness": 50}}
}}

User request: {user_input}
JSON:
"""
    response = client.models.generate_content(contents=[prompt], model=model)
    # Extract the JSON from the response
    try:
        if not response.text:
            return None
        start = response.text.find('{')
        end = response.text.rfind('}') + 1
        json_str = response.text[start:end]
        return json.loads(json_str)
    except Exception as e:
        print("Failed to parse Gemini output as JSON:", e)
        print("Gemini output was:", response.text)
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
        gemini_result = ask_gemini_for_tool(request.command, tools)
        if not gemini_result or "tool" not in gemini_result or "arguments" not in gemini_result:
            print("Could not parse tool/arguments from Gemini.")
            raise HTTPException(status_code=400, detail="Could not parse tool/arguments from Gemini.")
        
        tool_name = gemini_result["tool"]
        arguments = gemini_result["arguments"]
        mcp_result = await call_mcp_tool(tool_name, arguments)
        if isinstance(mcp_result, str):
            result = json.loads(mcp_result)
        if result["status"] == "error":
            return result["status"]
        return result["status"]
    except Exception as e:
        logger.error(f"Error processing command: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(PORT)) 