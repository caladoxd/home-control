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
from typing import Any, NotRequired, TypedDict, List, Dict
import chromadb
from chromadb.utils import embedding_functions

# Configure logging
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
PORT = os.getenv("MCP_CLIENT_PORT", "8006")
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

# Gemini setup
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
model = "gemini-2.5-flash-lite"
embedding_model = "gemini-embedding-001"

# Initialize Chroma client and collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("conversation_memory")

# Embedding helper using Gemini
def embed_text(text: str) -> list[float]:
    """Generate vector embedding for given text"""
    if not text:
        return []
    try:
        response = client.models.embed_content(model=embedding_model, contents=[text])
        return response.embeddings[0].values or [] if response.embeddings else []
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return []

def store_conversation(user_input: str, llm_output: str):
    """Store command/response pair in Chroma"""
    try:
        embedding = embed_text(user_input + llm_output)
        # Skip storing if embedding is empty
        if not embedding:
            logger.warning("Skipping conversation storage: empty embedding")
            return
        
        uid = f"conv_{hash(user_input + llm_output)}"
        collection.add(
            ids=[uid],
            documents=[f"User: {user_input}\nAssistant: {llm_output}"],
            embeddings=[embedding],
        )
        # logger.info("Conversation stored in memory.")
    except Exception as e:
        logger.error(f"Failed to store conversation: {e}")

def retrieve_similar_conversations(query: str, n_results: int = 3):
    """Retrieve the most similar past interactions"""
    try:
        query_vec = embed_text(query)
        if not query_vec:
            return []
        results = collection.query(query_embeddings=[query_vec], n_results=n_results)
        logger.info(f"Memory retrieval results: {results['documents']}")
        return results["documents"][0] if results["documents"] else []
    except Exception as e:
        logger.error(f"Memory retrieval error: {e}")
        return []

def ask_gemini_for_tool(user_input, tools, device_mappings=None,
                        command_count=0, conversation_history=None):
    if conversation_history is None:
        conversation_history = []

    # Retrieve similar memory
    similar_contexts = retrieve_similar_conversations(user_input)
    memory_context = "\n\n".join(similar_contexts) if similar_contexts else "None"

    # Prompt with memory
    prompt = f"""You are a home automation assistant that can call any MCP tool.
The available tools are:
{json.dumps(tools, indent=2)}

Relevant past memory:
{memory_context}

Given the user's request, follow these rules:

0. Output format requirements (STRICT):
   - You MUST return EXACTLY ONE JSON object and NOTHING ELSE.
   - Do NOT return multiple JSON objects, do NOT return arrays, do NOT include prose or markdown.
   - If multiple actions are needed, return ONLY the next action; you will be called again.
   - JSON schema: {{"tool": "<server>/<tool_name>", "arguments": {{ ... }} }}

1. For light control (on/off/dim), follow this exact two-step process:
   a. {{"tool": "tuya/get_device_mappings", "arguments": {{}}}}
   b. {{"tool": "tuya/control_light", "arguments": {{"device_id": "device_id_from_mappings", "command": "on/off"}}}}

2. For other requests, use the appropriate tool directly.
3. After all commands, return: {{"tool": "done", "arguments": {{"message": "message_to_user"}}}}.

User: {user_input}
"""

    # Add conversation history if any
    for entry in conversation_history:
        role = entry["role"]
        prompt += f"\n{role.capitalize()}: {json.dumps(entry['content'])}"

    # Add device mappings if applicable
    if device_mappings:
        prompt += f"\nSystem: Device mappings:\n{json.dumps(device_mappings, indent=2)}"

    prompt += "\nAssistant:"

    response = client.models.generate_content(contents=[prompt], model=model)
    try:
        if not response.text:
            return None
        if isinstance(response, dict):
            return response
        start = response.text.find("{")
        end = response.text.rfind("}") + 1
        json_str = response.text[start:end]
        return json.loads(json_str)
    except Exception as e:
        logger.error(f"Failed to parse Gemini output: {json_str}\n{e}")
        raise e

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application lifespan...")
    async with contextlib.AsyncExitStack() as stack:
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
            for tool in tools_result.tools:
                tools.append({
                    "name": f"{name}/{tool.name}",
                    "description": tool.description,
                    "inputSchema": tool.inputSchema,
                })
        yield

async def call_mcp_tool(tool_name, arguments):
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

class CommandRequest(BaseModel):
    command: str

app = FastAPI(lifespan=lifespan)

@app.post("/")
async def process_command(request: CommandRequest):
    """Process a natural language command"""
    try:
        conversation_history = []
        logger.info(f"Processing command: {request.command}")

        device_mappings = None
        results = []
        command_count = 0
        MAX_COMMANDS = 50

        while command_count < MAX_COMMANDS:
            gemini_result = ask_gemini_for_tool(
                request.command, tools, device_mappings, command_count, conversation_history
            )
            logger.info(f"Command {command_count}: {gemini_result}")

            if not gemini_result or "tool" not in gemini_result or "arguments" not in gemini_result:
                break

            tool_name = gemini_result["tool"]
            arguments = gemini_result["arguments"]
            mcp_result = await call_mcp_tool(tool_name, arguments)

            conversation_history.append({"role": "assistant", "content": {"tool": tool_name, "arguments": arguments}})
            conversation_history.append({"role": "system", "content": mcp_result})

            results.append(mcp_result)
            command_count += 1

            if tool_name == "tuya/get_device_mappings":
                result = json.loads(mcp_result) if isinstance(mcp_result, str) else mcp_result
                if result.get("status") == "success":
                    device_mappings = result["mappings"]

            if command_count >= MAX_COMMANDS or tool_name == "done":
                break

        # Store conversation in memory
        if results:
            final_message = json.loads(results[-1])["arguments"]["message"]
            store_conversation(request.command, final_message)
            return {"status": "success", "message": final_message}
        else:
            return {"status": "error", "message": "No results"}
    except Exception as e:
        logger.error(f"Error processing command: {gemini_result} {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(PORT))
