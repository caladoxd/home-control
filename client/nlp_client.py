#!/usr/bin/env python3

from mcp.types import CallToolResult
import os
import json
import re
import psutil
from typing import Dict, Any, List, Optional
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
server_params = StdioServerParameters(command="python", args=["server/tuya_server.py"])
SERVER_TIMEOUT = 30

# Global server session
server_lock = asyncio.Lock()

# Device mappings with aliases
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

class CommandRequest(BaseModel):
    command: str

def parse_natural_command(command: str):
    """Parse natural language command into device commands"""
    commands = []
    command = command.lower()
    
    # Extract brightness if present
    brightness = None
    brightness_match = re.search(r'(\d+)\s*(?:percent|%)', command)
    if brightness_match:
        brightness = int(brightness_match.group(1))
        command = command.replace(brightness_match.group(0), '')

    # Determine the action
    action = None
    if any(word in command for word in ['on', 'turn on', 'switch on', 'enable']):
        action = 'on'
    elif any(word in command for word in ['off', 'turn off', 'switch off', 'disable']):
        action = 'off'
    elif 'set' in command or 'brightness' in command:
        action = 'set'

    # Find matching devices
    for device_name, device_info in DEVICE_MAPPINGS.items():
        if any(alias in command for alias in device_info['aliases']):
            commands.append({
                "device_id": device_info['id'],
                "command": action or 'on',  # Default to 'on' if no action specified
                "brightness": brightness
            })

    return commands

def find_server_process():
    """Find existing tuya_server.py process"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'].lower():
                cmdline = proc.info.get('cmdline', [])
                if any('tuya_server.py' in cmd for cmd in cmdline):
                    return proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None

def kill_server_process(pid):
    """Kill the server process with the given PID"""
    try:
        if pid:
            proc = psutil.Process(pid)
            logger.info(f"Terminating existing server process (PID: {pid})")
            proc.terminate()
            try:
                proc.wait(timeout=5)  # Wait up to 5 seconds for graceful termination
            except psutil.TimeoutExpired:
                logger.warning(f"Server process did not terminate gracefully, killing it (PID: {pid})")
                proc.kill()
            return True
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        logger.warning(f"Error terminating server process (PID: {pid}): {e}")
    return False


async def send_command(command: str) -> Dict[str, Any]:
    """Send a command to the Tuya MCP server"""
    global server_session
    
    if server_session is None:
        return {"status": "error", "message": "Server session not initialized"}
    
    try:
        async with asyncio.timeout(SERVER_TIMEOUT):
            # Parse the natural language command
            device_commands = parse_natural_command(command)
            if not device_commands:
                return {"status": "error", "message": "No matching devices found in command"}

            results = []
            
            # Process each device command
            for cmd in device_commands:
                try:
                    result: CallToolResult = await server_session.call_tool(
                        "control_light",
                        arguments={
                            "device_id": cmd["device_id"],
                            "command": cmd["command"],
                            "brightness": cmd.get("brightness")
                        }
                    )
                    # Handle the result
                    result_dict = json.loads(result.content[0].text)
                except Exception as e:
                    logger.error(f"Error executing command: {str(e)}")
                    result_dict = {
                        "status": "error",
                        "message": f"Failed to execute command: {str(e)}"
                    }
                results.append(result_dict)

            # Check if any commands failed
            print(f"Results: {results}")
            failed_commands = [r for r in results if r["status"] == "error"]
            if failed_commands:
                return {
                    "status": "error",
                    "message": "Some commands failed",
                    "results": results
                }

            return {
                "status": "success",
                "message": "Commands executed successfully",
                "results": results
            }
    except asyncio.TimeoutError:
        logger.error("Command execution timed out")
        return {"status": "error", "message": "Command execution timed out"}
    except Exception as e:
        logger.error(f"Error in send_command: {str(e)}")
        return {"status": "error", "message": str(e)}

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
            yield

# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

@app.post("/command")
async def process_command(request: CommandRequest):
    """Process a natural language command"""
    try:
        result = await send_command(request.command)
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        return result["status"]
    except Exception as e:
        logger.error(f"Error processing command: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(PORT)) 