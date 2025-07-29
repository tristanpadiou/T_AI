# T_AI - AI Agent Framework

A powerful framework for building AI agents with **MCP (Model Context Protocol)** integration, **tools**, **memory management**, and **dependency handling**. T_AI simplifies the creation of sophisticated AI agents by providing a unified interface for multiple LLMs, external tools, and advanced conversation management.
This framework is build on top of pydantic_ai, go check it out at:

[https://ai.pydantic.dev/](https://ai.pydantic.dev/)

## 🚀 Key Features

- **🔗 MCP Integration**: Seamless connection to external tools and services via Model Context Protocol
- **🛠️ Tool System**: Built-in Google tools (image search, code execution) and easy custom tool integration  
- **🧠 Memory Management**: Intelligent conversation summarization for long-running sessions
- **📦 Dependency Management**: Clean state management and user context handling
- **🤖 Multi-LLM Support**: Compatible with Google Gemini, OpenAI, and Anthropic models
- **📱 Media Support**: Handle text, audio, images, and PDF files seamlessly
- **⚡ Async/Await**: Full asynchronous support for optimal performance
- **🔌 Extensible**: Easy to extend with custom tools and integrations

## 📦 Installation

### Available at pypi.com
```bash
pip install t-ai-project
```

### Using UV (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd T_AI

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

## 🔧 Core Dependencies

- **pydantic-ai >= 0.4.0**: Core AI framework
- **tavily-python >= 0.5.1**: Web search capabilities

## 🚀 Quick Start

### Basic Agent Creation

```python
import asyncio
from t_ai.t_agent import TAgent, Deps
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

# Initialize with Google Gemini
llm = GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key="your-api-key"))
agent = TAgent(llm=llm)

# Simple conversation
async def main():
    async with agent:
        response = await agent.chat(["Hello, what can you help me with?"])
        print(f"UI Version: {response.ui_version}")
        print(f"Voice Version: {response.voice_version}")

asyncio.run(main())
```

### With OpenAI

```python
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

llm = OpenAIModel('gpt-4-mini', provider=OpenAIProvider(api_key="your-openai-key"))
agent = TAgent(llm=llm)
```

## 🛠️ Framework Configuration

### TAgent Parameters

```python
agent = TAgent(
    llm=your_llm,                              # Required: pydantic-ai compatible model
    deps=custom_deps,                          # Optional: Custom dependencies object
    instructions="Custom instructions",        # Optional: Agent instructions
    tools=[],                                  # Optional: List of custom tools
    mcp_servers=[],                            # Optional: List of MCP servers
    summarizer=False,                          # Optional: Enable conversation summarization
    custom_summarizer_agent=None,              # Optional: Custom summarizer agent
    memory_length=20,                          # Optional: Messages before summarization
    memory_summarizer_length=15,               # Optional: Messages to summarize
    use_memory=True                            # Optional: Enable/disable memory
)
```

## 🔗 MCP Server Integration

### Using MCP Helper (Recommended)

```python
from t_ai.utils.helper_functions import MCP_server_helper
from t_ai.t_agent import TAgent

# Create MCP helper
mcp_helper = MCP_server_helper()

# Add different types of MCP servers
mcp_helper.add_mcp_server(type='http', mcp_server_url='https://mcp.notion.com/mcp')
mcp_helper.add_mcp_server(type='sse', mcp_server_url='https://mcp.notion.com/sse')
mcp_helper.add_mcp_server(type='stdio', command='npx', args=['-y', 'mcp-remote', 'https://mcp.notion.com/mcp'])

# Initialize agent with MCP servers
agent = TAgent(llm=llm, mcp_servers=mcp_helper.get_mpc_servers())
```

### Direct MCP Server Setup

```python
from pydantic_ai.mcp import MCPServerStreamableHTTP, MCPServerSSE, MCPServerStdio

mcp_servers = [
    MCPServerStreamableHTTP(url='https://mcp.notion.com/mcp'),
    MCPServerSSE(url='https://mcp.notion.com/sse'),
    MCPServerStdio(command='npx', args=['-y', 'mcp-remote', 'https://mcp.notion.com/mcp'])
]

agent = TAgent(llm=llm, mcp_servers=mcp_servers)
```

## 🛠️ Built-in Tools

### Google Image Search Tool

```python
from t_ai.PrebuiltTools.google_tools import search_images_tool

# Setup image search
image_tool = search_images_tool(
    api_key="your-google-api-key",
    search_engine_id="your-custom-search-engine-id"
)

agent = TAgent(llm=llm, tools=[image_tool])

# Usage
response = await agent.chat(["Find me an image of a sunset"])
```

### Google Code Execution Tool

```python
from t_ai.PrebuiltTools.google_tools import code_execution_tool

# Setup code execution  
code_tool = code_execution_tool(api_key="your-gemini-api-key")

agent = TAgent(llm=llm, tools=[code_tool])

# Usage
response = await agent.chat(["Calculate the factorial of 10 using Python"])
```

### Combined Tools Example

```python
from t_ai.PrebuiltTools.google_tools import search_images_tool, code_execution_tool

tools = [
    search_images_tool(api_key=google_api_key, search_engine_id=search_engine_id),
    code_execution_tool(api_key=google_api_key)
]

agent = TAgent(llm=llm, tools=tools)
```

## 💾 Memory Management

### Enable Automatic Summarization

```python
agent = TAgent(
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

agent = TAgent(
    llm=llm,
    summarizer=True,
    custom_summarizer_agent=custom_summarizer
)
```

### Memory and State Management

```python
# Access conversation history
messages = agent.memory.messages

# Access agent dependencies
deps = agent.deps
user_name = agent.deps.user
agents_output = agent.deps.agents_output

# Reset agent state
agent.reset()
```

## 📱 Media Support

### Text Input

```python
response = await agent.chat(["What's the weather like today?"])
```

### Image Input

```python
from pydantic_ai.messages import BinaryContent

# From file
with open("image.png", "rb") as f:
    image_data = f.read()

response = await agent.chat([
    "What do you see in this image?",
    BinaryContent(data=image_data, media_type='image/png')
])
```

### Audio Input

```python
# Audio file
with open("audio.wav", "rb") as f:
    audio_data = f.read()

response = await agent.chat([
    "Transcribe this audio",
    BinaryContent(data=audio_data, media_type='audio/wav')
])
```

### PDF Input

```python
# PDF file
with open("document.pdf", "rb") as f:
    pdf_data = f.read()

response = await agent.chat([
    "Summarize this document",
    BinaryContent(data=pdf_data, media_type='application/pdf')
])
```

## 🔧 Advanced Usage

### Context Manager (Recommended)

```python
async def main():
    async with TAgent(llm=llm, mcp_servers=mcp_servers) as agent:
        # MCP servers are automatically connected
        response = await agent.chat(["Help me with my Notion workspace"])
        print(response.ui_version)
        # MCP servers are automatically disconnected
```

### Manual Connection Management

```python
agent = TAgent(llm=llm, mcp_servers=mcp_servers)

# Connect manually
await agent.connect()

try:
    response = await agent.chat(["Hello"])
finally:
    # Disconnect manually
    await agent.disconnect()
```

### Custom Dependencies

```python
from t_ai.t_agent import Deps

# Create custom dependencies
custom_deps = Deps(
    agents_output={"previous_results": []},
    user="Alice"
)

agent = TAgent(llm=llm, deps=custom_deps)
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

agent = TAgent(llm=llm, tools=[weather_tool])
```

## 📝 Complete Framework Example

```python
import asyncio
import os
from dotenv import load_dotenv

from t_ai.t_agent import TAgent, Deps
from t_ai.utils.helper_functions import MCP_server_helper
from t_ai.PrebuiltTools.google_tools import search_images_tool, code_execution_tool

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
    mcp_helper.add_mcp_server(type='stdio', command='npx', 
                             args=['-y', '@modelcontextprotocol/server-filesystem', '/tmp'])
    
    # Setup tools
    tools = [
        search_images_tool(
            api_key=os.getenv('GOOGLE_API_KEY'),
            search_engine_id=os.getenv('GOOGLE_SEARCH_ENGINE_ID')
        ),
        code_execution_tool(api_key=os.getenv('GOOGLE_API_KEY'))
    ]
    
    # Setup custom dependencies
    deps = Deps(agents_output={}, user="Alice")
    
    # Initialize T_AI agent
    agent = TAgent(
        llm=llm,
        deps=deps,
        tools=tools,
        mcp_servers=mcp_helper.get_mcp_servers(),
        summarizer=True,
        memory_length=20,
        instructions="You are a helpful AI assistant with access to various tools and services."
    )
    
    # Use context manager for automatic connection handling
    async with agent:
        # Text conversation
        response = await agent.chat(["Hello, what can you help me with?"])
        print("Agent:", response.voice_version)
        
        # Math problem with code execution
        response = await agent.chat(["Calculate the sum of squares from 1 to 100"])
        print("Math Result:", response.ui_version)
        
        # Image search
        response = await agent.chat(["Find me an image of a beautiful landscape"])
        print("Image Search:", response.ui_version)
        
        # Check conversation history
        print(f"Total messages in memory: {len(agent.memory.messages)}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🧪 Testing and Development

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

## 🏗️ Architecture

T_AI is built with a modular architecture:

- **Core Agent (`TAgent`)**: Main framework class handling LLM interactions, memory, and coordination
- **MCP Integration**: Support for Model Context Protocol servers (HTTP, SSE, stdio)
- **Tool System**: Extensible tool framework with built-in Google tools
- **Memory Management**: Intelligent conversation summarization and state management
- **Dependencies**: Clean dependency injection for user context and shared state

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the notebooks in the `notebooks/` directory for examples
2. Review the docstrings in the source code  
3. Open an issue on GitHub

## 🙏 Acknowledgments

- Built on top of [pydantic-ai](https://github.com/pydantic/pydantic-ai)
- MCP (Model Context Protocol) integration
- Google AI and OpenAI API support
