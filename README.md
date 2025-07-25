# Cortana - AI Assistant with MCP Support

A powerful AI assistant built with pydantic-ai, featuring MCP (Model Context Protocol) server integration, Google tools, code execution capabilities, and advanced memory management.

## 🚀 Features

- **Multi-LLM Support**: Compatible with Google's Gemini and OpenAI models
- **MCP Server Integration**: Connect to external tools and services via MCP protocol
- **Google Tools**: Image search and code execution capabilities
- **Memory Management**: Automatic conversation summarization for long sessions
- **Media Support**: Handle audio, images, and PDF files
- **Async/Await**: Full asynchronous support for better performance
- **Extensible**: Easy to add custom tools and integrations

## 📦 Installation

### Using UV (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd Cortana

# Install using UV
uv sync
```

### Using pip

```bash
pip install -e .
```

### Using pip with requirements.txt

```bash
pip install -r requirements.txt
```

## 🔧 Dependencies

- **pydantic-ai >= 0.4.0**: Core AI framework
- **tavily-python >= 0.5.1**: Web search capabilities
- **ipykernel >= 6.30.0**: Jupyter notebook support

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
from cortana.cortana_agent import Cortana_agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

# Initialize with Google Gemini
llm = GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key="your-api-key"))
cortana = Cortana_agent(llm=llm)

# Simple chat
async def main():
    async with cortana:
        response = await cortana.chat(["Hello, what can you help me with?"])
        print(f"UI Version: {response.ui_version}")
        print(f"Voice Version: {response.voice_version}")

asyncio.run(main())
```

### With OpenAI

```python
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

llm = OpenAIModel('gpt-4-mini', provider=OpenAIProvider(api_key="your-openai-key"))
cortana = Cortana_agent(llm=llm)
```

## 🛠️ Configuration Options

### Cortana_agent Parameters

```python
cortana = Cortana_agent(
    llm=your_llm,                              # Required: pydantic-ai compatible model
    tools=[],                                  # Optional: List of custom tools
    mcp_servers=[],                            # Optional: List of MCP servers
    summarizer=False,                          # Optional: Enable conversation summarization
    custom_summarizer_agent=None,              # Optional: Custom summarizer agent
    memory_length=20,                          # Optional: Messages before summarization
    memory_summarizer_length=15                # Optional: Messages to summarize
)
```

## 🔗 MCP Server Integration

### Adding MCP Servers

```python
from cortana.utils.helper_functions import MCP_server_helper
from pydantic_ai.mcp import MCPServerStreamableHTTP, MCPServerSSE, MCPServerStdio

# Using helper class
mcp_helper = MCP_server_helper()
mcp_helper.add_mpc_server(type='http', mpc_server_url='https://mcp.notion.com/mcp')
mcp_helper.add_mpc_server(type='sse', mpc_server_url='https://mcp.notion.com/sse')
mcp_helper.add_mpc_server(type='stdio', command='npx', args=['-y', 'mcp-remote', 'https://mcp.notion.com/mcp'])

# Initialize Cortana with MCP servers
cortana = Cortana_agent(llm=llm, mcp_servers=mcp_helper.get_mpc_servers())
```

### Direct MCP Server Setup

```python
mcp_servers = [
    MCPServerStreamableHTTP(url='https://mcp.notion.com/mcp', headers=None),
    MCPServerSSE(url='https://mcp.notion.com/sse', headers=None),
    MCPServerStdio(command='npx', args=['-y', 'mcp-remote', 'https://mcp.notion.com/mcp'], env=None)
]

cortana = Cortana_agent(llm=llm, mcp_servers=mcp_servers)
```

## 🛠️ Google Tools Integration

### Image Search Tool

```python
from cortana.PrebuiltTools.google_tools import search_images_tool

# Setup image search
image_tool = search_images_tool(
    api_key="your-google-api-key",
    search_engine_id="your-custom-search-engine-id"
)

cortana = Cortana_agent(llm=llm, tools=[image_tool])

# Usage
response = await cortana.chat(["Find me an image of a sunset"])
```

### Code Execution Tool

```python
from cortana.PrebuiltTools.google_tools import code_execution_tool

# Setup code execution
code_tool = code_execution_tool(api_key="your-gemini-api-key")

cortana = Cortana_agent(llm=llm, tools=[code_tool])

# Usage
response = await cortana.chat(["Calculate the factorial of 10 using Python"])
```

### Combined Tools Example

```python
tools = [
    search_images_tool(api_key=google_api_key, search_engine_id=search_engine_id),
    code_execution_tool(api_key=google_api_key)
]

cortana = Cortana_agent(llm=llm, tools=tools)
```

## 💾 Memory Management

### Enable Automatic Summarization

```python
cortana = Cortana_agent(
    llm=llm,
    summarizer=True,                    # Enable summarization
    memory_length=20,                   # Summarize after 20 messages
    memory_summarizer_length=15         # Summarize oldest 15 messages
)
```

### Custom Summarizer Agent

```python
from pydantic_ai import Agent

custom_summarizer = Agent(
    llm, 
    instructions='Create detailed technical summaries focusing on code and solutions.'
)

cortana = Cortana_agent(
    llm=llm,
    summarizer=True,
    custom_summarizer_agent=custom_summarizer
)
```

### Accessing Memory and State

```python
# Access conversation history
messages = cortana.memory.messages

# Access agent state
deps = cortana.deps
user_name = cortana.deps.user
agents_output = cortana.deps.agents_output

# Reset memory
cortana.reset()
```

## 📱 Media Support

### Text Input

```python
response = await cortana.chat(["What's the weather like today?"])
```

### Image Input

```python
from pydantic_ai.messages import BinaryContent

# From file
with open("image.png", "rb") as f:
    image_data = f.read()

response = await cortana.chat([
    "What do you see in this image?",
    BinaryContent(data=image_data, media_type='image/png')
])
```

### Audio Input

```python
# Audio file
with open("audio.wav", "rb") as f:
    audio_data = f.read()

response = await cortana.chat([
    "Transcribe this audio",
    BinaryContent(data=audio_data, media_type='audio/wav')
])
```

### PDF Input

```python
# PDF file
with open("document.pdf", "rb") as f:
    pdf_data = f.read()

response = await cortana.chat([
    "Summarize this document",
    BinaryContent(data=pdf_data, media_type='application/pdf')
])
```

## 🔧 Advanced Usage

### Context Manager (Recommended)

```python
async def main():
    async with Cortana_agent(llm=llm, mcp_servers=mcp_servers) as cortana:
        # MCP servers are automatically connected
        response = await cortana.chat(["Help me with my Notion workspace"])
        print(response.ui_version)
        # MCP servers are automatically disconnected
```

### Manual Connection Management

```python
cortana = Cortana_agent(llm=llm, mcp_servers=mcp_servers)

# Connect manually
await cortana.connect()

try:
    response = await cortana.chat(["Hello"])
finally:
    # Disconnect manually
    await cortana.disconnect()
```

### Custom Tools

```python
from pydantic_ai.tools import Tool

def custom_weather_tool(location: str) -> str:
    """Get weather information for a location"""
    # Your weather API logic here
    return f"Weather in {location}: Sunny, 25°C"

weather_tool = Tool(
    custom_weather_tool,
    name='get_weather',
    description='Get current weather for any location'
)

cortana = Cortana_agent(llm=llm, tools=[weather_tool])
```

## 📝 Complete Example

```python
import asyncio
import os
from dotenv import load_dotenv

from cortana.cortana_agent import Cortana_agent
from cortana.utils.helper_functions import MCP_server_helper
from cortana.PrebuiltTools.google_tools import search_images_tool, code_execution_tool

from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.messages import BinaryContent

# Load environment variables
load_dotenv()

async def main():
    # Setup LLM
    llm = GoogleModel('gemini-2.5-flash', 
                     provider=GoogleProvider(api_key=os.getenv('GOOGLE_API_KEY')))
    
    # Setup MCP servers
    mcp_helper = MCP_server_helper()
    mcp_helper.add_mpc_server(type='stdio', command='npx', 
                             args=['-y', '@modelcontextprotocol/server-filesystem', '/tmp'])
    
    # Setup tools
    tools = [
        search_images_tool(
            api_key=os.getenv('GOOGLE_API_KEY'),
            search_engine_id=os.getenv('GOOGLE_SEARCH_ENGINE_ID')
        ),
        code_execution_tool(api_key=os.getenv('GOOGLE_API_KEY'))
    ]
    
    # Initialize Cortana
    cortana = Cortana_agent(
        llm=llm,
        tools=tools,
        mcp_servers=mcp_helper.get_mpc_servers(),
        summarizer=True,
        memory_length=20
    )
    
    # Use context manager for automatic connection handling
    async with cortana:
        # Set user name
        cortana.deps.user = "Alice"
        
        # Text conversation
        response = await cortana.chat(["Hello Cortana, what can you help me with?"])
        print("Cortana:", response.voice_version)
        
        # Math problem with code execution
        response = await cortana.chat(["Calculate the sum of squares from 1 to 100"])
        print("Math Result:", response.ui_version)
        
        # Image search
        response = await cortana.chat(["Find me an image of a beautiful landscape"])
        print("Image Search:", response.ui_version)
        
        # Check conversation history
        print(f"Total messages in memory: {len(cortana.memory.messages)}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🧪 Testing

Run the included Jupyter notebooks to test different features:

- `notebooks/cortana_test.ipynb`: Basic functionality testing
- `notebooks/cort_mcp_test.ipynb`: MCP server integration testing
- `notebooks/cortana_voice_test.ipynb`: Voice/audio capabilities testing
- `notebooks/memory_handling.ipynb`: Memory management testing

## 🔑 Environment Variables

Create a `.env` file in your project root:

```env
GOOGLE_API_KEY=your_google_api_key
GOOGLE_SEARCH_ENGINE_ID=your_custom_search_engine_id
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the notebooks in the `notebooks/` directory for examples
2. Review the docstrings in the source code
3. Open an issue on GitHub
