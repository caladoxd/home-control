# MCP (Model Context Protocol) Smart Home Control

This project implements a Model Context Protocol system for controlling smart home devices through Siri and Tuya integration.

## Components

- **MCP Client**: A client application that can be triggered by Siri to send commands to the MCP server
- **MCP Server**: A server application that handles Tuya integration for controlling smart lights

## Setup

### Prerequisites

- Python 3.8+
- Tuya Developer Account
- iOS device with Siri
- Shortcuts app (for Siri integration)

### Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure your Tuya credentials in `config.yaml`
4. Start the server:
   ```bash
   python server/main.py
   ```
5. Configure the client for Siri integration

## Configuration

1. Create a Tuya developer account and get your API credentials
2. Update `config.yaml` with your Tuya credentials
3. Configure Siri shortcuts to trigger the client application

## Usage

1. Start the server
2. Use Siri to trigger commands through the client
3. The server will process commands and control your Tuya devices

## Security

- All communication between client and server is encrypted
- API keys and credentials are stored securely
- Local network communication only 