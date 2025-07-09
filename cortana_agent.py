from __future__ import annotations
import requests
import httpx
from pydantic_ai import Agent, RunContext, format_as_xml
from pydantic_ai.common_tools.tavily import tavily_search_tool
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.mcp import MCPServerStreamableHTTP, MCPServerSSE, MCPServerStdio
from dataclasses import dataclass
from datetime import datetime
from pydantic import Field
from google import genai
from google.genai import types


@dataclass
class Api_keys:
    api_keys: dict


@dataclass
class Message_state:
    messages: list[ModelMessage]

@dataclass
class Deps:
    agents_output: dict
    user:str
    
class Cortana_agent:
    def __init__(self, api_keys:dict, mpc_server_urls:list = [], mpc_stdio_commands:list = []):
        """
        Args:
            api_keys (dict): The API keys to use as a dictionary, the keys are:
            \n
            example:
            {
                'google_api_key': (str): The Google API key,

                'openai_api_key': (str): The OpenAI API key,

                'tavily_key': (str): The Tavily API key,

                'pse': (str): The Google Custom Search Engine ID,

                
            }
            mpc_server_urls (list): The list of dicts containing the url\n
            and the name of the mpc server and the type of connection, \n
            and the bearer token if necessary
              \n
              example:
              [
                {
                  'url': 'http://localhost:8000',
                  'name': 'mcp_server_1',
                  'type': 'http','SSE'
                  'headers': {'Authorization': 'Bearer 1234567890'} #optional or None
                }
              ]
            mpc_stdio_commands (list): The list of commands to use with the stdio mpc server
              \n
              example:
              [
                {
                  'name': 'memory',
                  'command': 'npx', 'docker', 'npm', 'python'
                  'args': ['-y', '@modelcontextprotocol/server-memory']
                }
              ]
        """
        
        self.api_keys=Api_keys(api_keys=api_keys)
        
        self.mpc_server_urls = mpc_server_urls
        self.mpc_stdio_commands = mpc_stdio_commands
        # tools
        llms={'pydantic_llm':GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key=self.api_keys.api_keys['google_api_key'])),
              'mcp_llm':OpenAIModel('gpt-4.1-mini',provider=OpenAIProvider(api_key=self.api_keys.api_keys['openai_api_key']))}
    
        
        async def find_images_tool(ctx: RunContext[Deps],query:str):
            """Search for images using this tool, this tool can search one image at a time
            args: query
            return: image url
            """
            # Define the API endpoint for Google Custom Search
            url = "https://www.googleapis.com/customsearch/v1"
            

            params = {
                "q": query,
                "cx": self.api_keys.api_keys['pse'],
                "key": self.api_keys.api_keys['google_api_key'],
                "searchType": "image",  # Search for images
                "num": 1  # Number of results to fetch
            }

            # Make the request to the Google Custom Search API
            response = requests.get(url, params=params)
            data = response.json()

            # Check if the response contains image results
            if 'items' in data:
                # Extract the first image result
                image_url = data['items'][0]['link']
                res=f'image url for {query} : {image_url}'

                if not ctx.deps.agents_output.get('find_images_tool'):
                    ctx.deps.agents_output['find_images_tool']=[]

                ctx.deps.agents_output['find_images_tool'].append(image_url)

                if len(ctx.deps.agents_output['find_images_tool'])>5:
                    del ctx.deps.agents_output['find_images_tool'][0]
                return f'image url for {query} : {image_url}'
            else:
                return 'no image found'
                
        async def code_execution_tool(ctx: RunContext[Deps],query:str):
            """
            Use this tool to answer math questions or any other questions that require code execution, it can handle complex math problems and code execution.
            args: query (str): the detailed query 
            return: the result of the code execution
            """
            client = genai.Client(api_key=self.api_keys.api_keys['google_api_key'])

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=query,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(code_execution=types.ToolCodeExecution)]
                ),
            )
            
            
            res={}
            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    res['text']=part.text
                    
                if part.executable_code is not None:
                    res['code']=part.executable_code.code
                if part.code_execution_result is not None:
                    res['output']=part.code_execution_result.output
            ctx.deps.agents_output['code_execution_tool']=res

            
            return f'the result of the code execution is {res.get("output") if res.get("output") else "no result"}'
        

        #summarize old messages
        self.summarize_agent=Agent(llms['pydantic_llm'],instructions='Summarize this conversation, omitting small talk and unrelated topics. Focus on the technical discussion and next steps.')


        
        #mpc servers
        self.mpc_servers=[]
        for mpc_server_url in self.mpc_server_urls:
            if mpc_server_url['type'] == 'http':
                if mpc_server_url['headers'] is not None:
                    self.mpc_servers.append(MCPServerStreamableHTTP(url=mpc_server_url['url'], headers=mpc_server_url['headers']))
                else:
                    self.mpc_servers.append(MCPServerStreamableHTTP(mpc_server_url['url']))
            elif mpc_server_url['type'] == 'SSE':
                if mpc_server_url['headers'] is not None:
                    self.mpc_servers.append(MCPServerSSE(url=mpc_server_url['url'], headers=mpc_server_url['headers']))
                else:
                    self.mpc_servers.append(MCPServerSSE(mpc_server_url['url']))
        for mpc_stdio_command in self.mpc_stdio_commands:
            self.mpc_servers.append(MCPServerStdio(mpc_stdio_command['command'], mpc_stdio_command['args']))                 
        
        self._mcp_context_manager = None
        self._is_connected = False
        #agent
        @dataclass
        class Cortana_output:
            ui_version: str= Field(description='a markdown format version of the answer for displays if necessary')
            voice_version: str = Field(description='a conversationnal version of the answer for text to voice')

        instructions = """
        # You are Cortana - A Helpful AI Assistant

        ## Your Role:
        You are Cortana, a helpful assistant capable of handling a wide range of tasks.

        ## Available Information:
        - Current time and date
        - User query and context
        - User's name (always refer to them by their first name)
        - Various tools and capabilities

        ## Tool Usage Guidelines:
        1. Before using any tools, explain what you plan to do and ask for user confirmation
        2. Only proceed with tool usage after the user says yes

        ## Communication Style:
        - Always address the user by their first name
        - Be clear and concise in explanations
        - Make suggestions on what to do next
        """

        self.agent=Agent(
            llms['mcp_llm'], 
            output_type=Cortana_output, 
            tools=[tavily_search_tool(self.api_keys.api_keys['tavily_key']), find_images_tool, code_execution_tool],
            mcp_servers=self.mpc_servers, 
            instructions=instructions
        )
        self.memory=Message_state(messages=[])
        self.deps=Deps(agents_output={}, user='')
    
    async def connect(self):
        """Establish persistent connection to MCP server"""
        if not self._is_connected:
            self._mcp_context_manager = self.agent.run_mcp_servers()
            await self._mcp_context_manager.__aenter__()
            self._is_connected = True
            print("Connected to MCP server")

    async def disconnect(self):
        """Close the MCP server connection"""
        if self._is_connected and self._mcp_context_manager:
            try:
                await self._mcp_context_manager.__aexit__(None, None, None)
                print("Disconnected from MCP server")
            except RuntimeError as e:
                if "Attempted to exit cancel scope in a different task" in str(e):
                    # This is expected when disconnecting from a different task context
                    print("MCP server disconnected (task context changed)")
                else:
                    raise e
            except Exception as e:
                print(f"Error during MCP disconnect: {e}")
            finally:
                self._is_connected = False
                self._mcp_context_manager = None
                
    async def chat(self, query:any):
        """
        # Chat Function Documentation

        This function enables interaction with the user through various types of input.

        ## Parameters

        - `query`: The input to process. Can be one of the following types:
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

        - `Cortana_output`: as a pydantic object, the ui_version and voice_version are the two fields of the object

        ## Extra Notes
        The deps and message_history of cortana can be accessed using the following code:
        ```python
        cortana_agent.deps
        cortana_agent.memory.messages
        ```
        """
        if not self._is_connected:
            await self.connect()
            
        result=await self.agent.run(query, deps=self.deps, message_history=self.memory.messages)
        
        #summarize old messages
        if len(result.all_messages()) > 20:
                    oldest_messages = result.all_messages()[:15]
                    summary = await self.summarize_agent.run(f'oldest messages: {oldest_messages}')
                    # Return the last message and the summary
                    self.memory.messages=summary.new_messages() + result.new_messages()
        else:
            self.memory.messages=result.all_messages()
        
        return result.output
    
    def reset(self):
        """
        Resets the Cortana agent to its initial state.

        Returns:
            str: A confirmation message indicating that the agent has been reset.
        """
        self.memory.messages=[]
        self.deps=Deps(agents_output={}, user='')
        return f'Cortana has been reset'
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
