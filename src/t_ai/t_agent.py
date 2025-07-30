from __future__ import annotations
import asyncio
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest,ModelResponse

from dataclasses import dataclass
from datetime import datetime
from pydantic import Field



@dataclass
class Message_state:
    messages: list[ModelMessage]

@dataclass
class Deps:
    agents_output: dict
    user:str
    
class TAgent:
    def __init__(self,llm:any, deps:Deps = None,voice:bool = False, instructions:str = None, tools:list = [], mcp_servers:list = [], summarizer:bool = False, custom_summarizer_agent:Agent = None, memory_length:int = 20, memory_summarizer_length:int = 15, use_memory:bool = True, retries:int = 3):
        """
        ## Args:
        ### llm
            llm (any): The LLM to use as a model has to be a pydantic_ai model
            \n
            **example:**
            ```python
            GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key=api_keys['google_api_key']))
            OpenAIModel('gpt-4.1-mini',provider=OpenAIProvider(api_key=api_keys['openai_api_key'])) (recommended for mcp servers)
            AnthropicModel('claude-3-5-sonnet-20240620', provider=AnthropicProvider(api_key=api_keys['anthropic_api_key'])) (recommended for mcp servers but more expensive)
            ```
            \n
        ### voice
            voice (bool): Whether to use the voice or not default is False
            \n
        ### deps
            deps (Deps): The deps to use for the agent, if not provided, the default deps will be used
            \n
            **example:**
            ```python
            @dataclass
            class Deps:
                agents_output: dict
                user:str
            ```
            \n
        ### instructions
            instructions (str): The instructions to use for the agent, if not provided, the default instructions will be used
            \n
            **example:**
            ```python
            "You are a helpful assistant capable of handling a wide range of tasks."
            ```
            \n
        ### summarizer
            summarizer (bool): Whether to use the summarizer agent or not default is False
            \n
        ### custom_summarizer_agent
            custom_summarizer_agent (Agent): The custom summarizer agent to use, if not provided, the main agent will be used, it has to be a pydantic_ai agent
            \n
        ### memory_length
            memory_length (int): The number of messages to keep in memory before summarizing, default is 20
            \n
        ### memory_summarizer_length
            memory_summarizer_length (int): The number of messages to summarize, default is 15
            \n
        ### use_memory
            use_memory (bool): Whether to use the built in memory or not default is True
            \n
        ### mpc_servers
            mcp_servers (list): The list of MCP servers to use, they have to be a list of pydantic_ai MCP servers:
            \n
            **example:**
                example:
                ```python
                [
                    MCPServerStreamableHTTP(url='https://mcp.notion.com/mcp', headers=None),
                    MCPServerSSE(url='https://mcp.notion.com/sse', headers=None),
                    MCPServerStdio(command='npx', args=['-y', 'mcp-remote', 'https://mcp.notion.com/mcp'], env=None)
                ]
                ```
            \n
            or you can use the helper function to add a mcp server:
            ```python
            from cortana.utils.helper_functions import MCP_server_helper

            mcp_server_helper=MCP_server_helper()

            mcp_server_helper.add_mcp_server(type='http', mcp_server_url='https://mcp.notion.com/mcp', headers=None)
           
            mcp_server_helper.get_mcp_servers()
            ```
            \n
        ### tools
            tools (list): The list of tools to use as functions:
              \n
              **example:**
              example:
              ```python

              def tool_1():
                return "tool_1"
              def tool_2():
                return "tool_2"
              def tool_3():
                return "tool_3"
              
              [
                tool_1,
                tool_2,
                tool_3
              ]
              ```
        """
        
        self.llm=llm
        self.tools=tools
        self.mcp_servers = mcp_servers
        self.voice=voice
        self.retries=retries
        #deps
        if deps:
            self.deps=deps
        else:
            self.deps=Deps(agents_output={}, user='')

        #memory
        self.use_memory=use_memory
        self.memory=Message_state(messages=[])
        #summarize old messages
        self.summarize=summarizer
        self.memory_length=memory_length
        self.memory_summarizer_length=memory_summarizer_length
        self.custom_summarizer_agent=custom_summarizer_agent
        if self.summarize:
            if not self.custom_summarizer_agent:
                self.summarize_agent=Agent(llm,instructions='Summarize this conversation, omitting small talk and unrelated topics. Focus on the technical discussion and next steps.')
            else:
                self.summarize_agent=self.custom_summarizer_agent

               

        self._mcp_context_manager = None
        self._is_connected = False
        
        #agent
        @dataclass
        class TAgent_output:
            ui_version: str= Field(description='a markdown format version of the answer for displays if necessary')
            voice_version: str = Field(description='a conversationnal version of the answer for text to voice')
        

        if instructions:
            self.instructions = instructions
        else:
            self.instructions = """
        # You are a helpful AI Assistant

        ## Your Role:
        You are a helpful assistant capable of handling a wide range of tasks.

        ## Available Information:
        - Current time and date
        - User query and context
        - User's name (always refer to them by their first name)
        - Various tools and capabilities
        - always ask the user for confirmation before using any tool

        """
        
        self.agent=Agent(
            self.llm, 
            output_type= str if not self.voice else TAgent_output, 
            tools=self.tools,
            retries=self.retries,
            mcp_servers=self.mcp_servers, 
            instructions=self.instructions
        )
        
        
    
    async def connect(self):
        """Establish persistent connection to MCP server"""
        if not self._is_connected:
            print("Initializing MCP server connection...")
            self._mcp_context_manager = self.agent.run_mcp_servers()
            try:
                print("Connecting to MCP server (this may take a moment on first run)...")
                await self._mcp_context_manager.__aenter__()
                self._is_connected = True
                return "Connected to MCP server"
            except Exception as e:
                print(f"Failed to connect to MCP server: {e}")
                # Clean up on failure
                self._mcp_context_manager = None
                self._is_connected = False
                raise e

    async def disconnect(self):
        """Close the MCP server connection"""
        if self._is_connected and self._mcp_context_manager:
            try:
                await self._mcp_context_manager.__aexit__(None, None, None)
            except Exception as e:
                # Log the error but continue cleanup
                print(f"Warning: Error during MCP server disconnect: {e}")
            finally:
                self._is_connected = False
                self._mcp_context_manager = None
            return "Disconnected from MCP server"
        elif self._mcp_context_manager:
            # Handle case where context manager exists but connection flag is wrong
            self._mcp_context_manager = None
            self._is_connected = False
            return "Cleaned up MCP server resources"
        else:
            return "No MCP server connection to disconnect"
    #summarize old messages
    async def summarizer(self,result):
        """
        function to summarize memory.messages when it is too long
        args:
            result: the models output
        
        """
        if len(result.all_messages()) > self.memory_length:
            oldest_messages=[]
            for i in result.all_messages()[:self.memory_summarizer_length]:
                
                if isinstance(i,ModelRequest):
                    if isinstance(i.parts[0].content,list):
                        oldest_messages.append({'user_query':i.parts[0].content[0]})
                    else:
                        oldest_messages.append({'user_query':i.parts[0].content})
                elif isinstance(i,ModelResponse):
                    oldest_messages.append({'model_response':i})
            summary = await self.summarize_agent.run(f'oldest messages: {str(oldest_messages)}')
            # Return the last message and the summary
            self.memory.messages=summary.new_messages() + result.new_messages()
        else:
            self.memory.messages=result.all_messages()
    async def chat(self, query:list):
        """
        # Chat Function Documentation

        This function enables interaction with the user through various types of input.

        ## Parameters

        - `query`: The input to process. A list of inputs of the following types:
        - String: Direct text input passed to the agent
        - Binary content: Special format for media files (see below)

        ## Binary Content Types

        The function supports different types of media through `BinaryContent` objects:

        ### Audio
        ```python
        cortana_agent.chat([
            'optional string message',
            BinaryContent(data=audio, media_type='audio/wav')
        ])
        ```

        ### PDF Files
        ```python
        cortana_agent.chat([
            'optional string message',
            BinaryContent(data=pdf_path.read_bytes(), media_type='application/pdf')
        ])
        ```

        ### Images
        ```python
        cortana_agent.chat([
            'optional string message',
            BinaryContent(data=image_response.content, media_type='image/png')
        ])
        ```

        ## Returns

        - `Cortana_output`: as a pydantic object, the ui_version and voice_version are the two fields of the object if voice is True, otherwise it is a string

        ## Extra Notes
        The deps and message_history of cortana can be accessed using the following code:
        ```python
        cortana_agent.deps
        cortana_agent.memory.messages
        ```
        """
        if not self._is_connected:
            try:
                await self.connect()
            except Exception as e:
                return f"Failed to connect to MCP server before chat: {e}"
            
        try:
            if self.use_memory:
                result=await self.agent.run(query, deps=self.deps, message_history=self.memory.messages)
                if self.summarize:
                    asyncio.create_task(self.summarizer(result))
                else:
                    self.memory.messages=result.all_messages()
            else:
                result=await self.agent.run(query, deps=self.deps)
        except Exception as e:
            print(f"Error during chat: {e}")
            # Try to reconnect on connection errors
            if "connection" in str(e).lower() or "timeout" in str(e).lower():
                print("Attempting to reconnect...")
                self._is_connected = False
                try:
                    await self.connect()
                    if self.use_memory:
                        result=await self.agent.run(query, deps=self.deps, message_history=self.memory.messages)
                        if self.summarize:
                            asyncio.create_task(self.summarizer(result))
                        else:
                            self.memory.messages=result.all_messages()
                    else:
                        result=await self.agent.run(query, deps=self.deps)
                    return result
                except Exception as reconnect_error:
                    return f"Chat failed and reconnection failed: {reconnect_error}"
       
        return result
    
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
    
    async def reset(self):
        """Reset the agent state"""

        if self.use_memory:
            if self._is_connected:
                await self.disconnect()
            self.memory.messages=[]
       
            return "Agent reset"
        else:
            return "no memory to reset"
        