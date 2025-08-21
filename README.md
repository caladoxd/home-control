# MCP (Model Context Protocol) Smart Home Control

This project implements a Model Context Protocol system for controlling smart home devices through Siri and Tuya integration.

## Components

- **MCP Client**: A client application that can be triggered by Siri to send commands to the MCP server
- **MCP Server**: A server application that handles Tuya integration for controlling smart lights

## Setup

### Prerequisites

- Python 3.8+
- Tuya Developer Account
- iOS device (Siri and Shortcuts app)

### Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure credentials in `.env`
5. Configure Siri integration

## Configuration

1. Create a Tuya developer account and get your API credentials
2. Get credentials from Gemini Studio
3. Get credentials from Google Programmable Search Engine
4. Update `.env` with your credentials
5. Configure Siri shortcuts to trigger the client application (send HTTP request with command to your app's URL)

## Usage

1.  Start Tuya service:
   ```bash
   python server/tuya_service.py
   ```
2. Start client:
   ```bash
   python client/mcp_client.py
   ```
3. Use Siri to trigger commands through the client
4. The server will process commands and control your devices

## Security

- All communication between client and server is encrypted
- API keys and credentials are stored securely
- Local network communication only 
