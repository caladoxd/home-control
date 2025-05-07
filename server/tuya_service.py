from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from tuyapy import TuyaApi
import os
from dotenv import load_dotenv
import logging
import json
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(title="Tuya Service")

# Tuya configuration
TUYA_USERNAME = os.getenv("TUYA_USERNAME")
TUYA_PASSWORD = os.getenv("TUYA_PASSWORD")
PORT = os.getenv("PORT", 8005)

# Initialize Tuya client as a global variable
tuya = None

def get_tuya_client():
    """Get or create Tuya client"""
    global tuya
    
    if tuya is None:
        # Create new client and authenticate
        logger.info("Initializing Tuya client...")
        tuya = TuyaApi()
        try:
            tuya.init(
                username=TUYA_USERNAME,
                password=TUYA_PASSWORD,
                countryCode="1",  # Default to US, change as needed
                bizType="smart_life"
            )
            logger.info("Tuya client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Tuya client: {str(e)}")
            raise
    
    return tuya

class LightControlRequest(BaseModel):
    device_id: str
    command: str
    brightness: Optional[int] = None
    color_temp: Optional[int] = None
    color: Optional[str] = None

@app.post("/control_light")
async def control_light(request: LightControlRequest) -> Dict[str, Any]:
    """Control a Tuya light device"""
    try:
        tuya_client = get_tuya_client()
        
        if request.command == "on":
            response = tuya_client.device_control(request.device_id, "turnOnOff", {"value": "1"})
        elif request.command == "off":
            response = tuya_client.device_control(request.device_id, "turnOnOff", {"value": "0"})
        elif request.command == "set":
            params = {}
            if request.brightness is not None:
                params["brightness"] = request.brightness
            if request.color_temp is not None:
                params["colorTemp"] = request.color_temp
            if request.color is not None:
                params["color"] = request.color
            if params:
                response = tuya_client.device_control(request.device_id, "setBrightness", params)
            else:
                raise HTTPException(status_code=400, detail="No parameters provided for set command")
        else:
            raise HTTPException(status_code=400, detail="Invalid command")
        
        # Check if response is valid
        if response:
            try:
                if isinstance(response, str):
                    response = json.loads(response)
                return {"status": "success", "message": f"Light {request.command} command executed", "response": response}
            except json.JSONDecodeError:
                return {"status": "success", "message": f"Light {request.command} command executed"}
        else:
            return {"status": "success", "message": f"Light {request.command} command executed"}
    
    except Exception as e:
        logger.error(f"Error controlling light: {str(e)}")
        # If it's an auth error, reset the client
        if "authorization" in str(e).lower():
            global tuya
            tuya = None
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/devices")
async def list_devices() -> Dict[str, Any]:
    """List all available Tuya devices"""
    try:
        tuya_client = get_tuya_client()
        devices = tuya_client.get_all_devices()
        
        if devices is None:
            return {"status": "success", "devices": []}
        
        # Convert devices to dictionaries
        device_list = []
        for device in devices:
            device_dict = {
                "id": device.obj_id,
                "name": device.obj_name,
                "type": device.dev_type,
                "online": device.data.get("online", False) if device.data else False,
            }
            device_list.append(device_dict)
            
        return {"status": "success", "devices": device_list}
    except Exception as e:
        logger.error(f"Error listing devices: {str(e)}")
        # If it's an auth error, reset the client
        if "authorization" in str(e).lower():
            global tuya
            tuya = None
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Initialize Tuya client when server starts
    try:
        get_tuya_client()
    except Exception as e:
        logger.error(f"Failed to initialize Tuya client on startup: {str(e)}")
    
    # Run the FastAPI server
    uvicorn.run(app, host="localhost", port=int(PORT)) 