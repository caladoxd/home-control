from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent
from typing import Dict, Any, List, Optional
import os
from dotenv import load_dotenv
import httpx
import asyncio
import sys


# # Load environment variables
load_dotenv()

# # Create MCP server instance
mcp = FastMCP("tuya_server")

# Tuya service configuration
TUYA_SERVICE_URL = f"http://localhost:{os.getenv('PORT', '8005')}"
TIMEOUT = 30  # seconds

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

async def check_service_availability():
    """Check if the Tuya service is available"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{TUYA_SERVICE_URL}/devices", timeout=5.0)
            response.raise_for_status()
            return True
    except Exception as e:
        print(f"Tuya service is not available: {str(e)}")
        return False

@mcp.tool()
async def control_light(device_id: str, command: str, brightness: Optional[int] = None, 
                 color_temp: Optional[int] = None, color: Optional[str] = None) -> Dict[str, Any]:
    """
    Control a Tuya light device
    
    Args:
        device_id: The ID of the device to control
        command: The command to execute (on, off, set)
        brightness: Optional brightness level (0-100)
        color_temp: Optional color temperature
        color: Optional color value
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{TUYA_SERVICE_URL}/control_light",
                json={
                    "device_id": device_id,
                    "command": command,
                    "brightness": brightness,
                    "color_temp": color_temp,
                    "color": color
                },
                timeout=TIMEOUT
            )
            response.raise_for_status()
            result = response.json()
            return result
    
    except httpx.TimeoutException:
        return {"status": "error", "message": "Request timed out"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def list_devices() -> Dict[str, Any]:
    """List all available Tuya devices"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{TUYA_SERVICE_URL}/devices", timeout=TIMEOUT)
            response.raise_for_status()
            result = response.json()
            return result
    except httpx.TimeoutException:
        return {"status": "error", "message": "Request timed out"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def get_device_mappings() -> Dict[str, Any]:
    """Get the mapping of device names to their IDs and aliases"""
    return {"status": "success", "mappings": DEVICE_MAPPINGS}

@mcp.prompt()
def device_control_help() -> str:
    """Get help on how to control devices"""
    return """I can help you control your Tuya smart devices. Here are some examples of what you can do:

1. Turn devices on/off:
   - "Turn on the living room light"
   - "Turn off the bedroom light"

2. Adjust brightness:
   - "Set living room brightness to 50%"
   - "Dim the bedroom lights to 30%"
"""

if __name__ == "__main__":
    mcp.run()
    # # Check if Tuya service is available before starting
    # if not asyncio.run(check_service_availability()):
    #     sys.exit(1)
    
    # # Run the MCP server with persistent connection
    # try:
    #     while True:  # Keep server alive
    #         mcp.run()
    # except KeyboardInterrupt:
    # except Exception as e:
    #     sys.exit(1)