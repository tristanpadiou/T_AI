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
from pydantic_ai.mcp import MCPServerStreamableHTTP
from dataclasses import dataclass
from datetime import datetime
from pydantic import Field
from google import genai
from google.genai import types
import json
import os
from dotenv import load_dotenv
import asyncio
load_dotenv()
# import nest_asyncio
# nest_asyncio.apply()
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_openai import ChatOpenAI


@dataclass
class Api_keys:
    api_keys: dict


@dataclass
class Message_state:
    messages: list[ModelMessage]

@dataclass
class Deps:
    agents_output: dict
    mail_inbox: dict
    google_agent_output: dict
    
class Cortana_agent:
    def __init__(self, api_keys:dict, mpc_server_urls:dict = {}):
        """
        Args:
            
            api_keys (dict): The API keys to use as a dictionary
            google_agent_api_url (str): The URL of the Google Agent API
            outlook_mpc_url (str): The URL of the Outlook Agent API
            notion_agent_mpc_url (str): The URL of the Notion Agent API
            
        """
        GEMINI_MODEL='gemini-2.0-flash'
        self.api_keys=Api_keys(api_keys=api_keys)
        
        self.mpc_server_urls = mpc_server_urls
       
        # tools
        llms={'pydantic_llm':GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key=self.api_keys.api_keys['google_api_key'])),
              'langchain_llm':ChatGoogleGenerativeAI(google_api_key=self.api_keys.api_keys['google_api_key'], model=GEMINI_MODEL, temperature=0.3),
              'openai_llm':ChatOpenAI(model='gpt-4.1-nano',api_key=self.api_keys.api_keys['openai_api_key']),
              'mcp_llm':OpenAIModel('gpt-4.1-mini',provider=OpenAIProvider(api_key=self.api_keys.api_keys['openai_api_key']))}
        
        async def Memory_tool(ctx: RunContext[Deps], query:str,tool:str):
            """
            Use this tool to dive into the memory database of the agents to answer questions based on previous information provided from previous tool calls
            this tool can also be used to get more details about a specific email or task.
            The database is dictionary.
            Args:
                query (str): The query related to the Memory_tool and its capabilities
                tool (str): The tool that was used to get the information either google_agent_tool or outlook_agent_tool
                
            Returns:
                str: The response from the Memory_tool
            """
            history=ctx.deps.agents_output.get(tool)
            
            
            answer_question_agent=Agent(llms['pydantic_llm'], instructions="answer the question based on the information provided")
            result= answer_question_agent.run_sync(f"answer the question based on the information provided: {history} and the query: {query}")
            return result.output



      
        
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
        
        #mpc servers
        mpc_servers=[]
        for mpc_server_url in self.mpc_server_urls:
            mpc_servers.append(MCPServerStreamableHTTP(self.mpc_server_urls[mpc_server_url]))
                    
        
        self._mcp_context_manager = None
        self._is_connected = False
        #agent
        @dataclass
        class Cortana_output:
            ui_version: str= Field(description='a markdown format version of the answer for displays if necessary')
            voice_version: str = Field(description='a conversationnal version of the answer for text to voice')

        self.agent=Agent(llms['mcp_llm'], output_type=Cortana_output, tools=[tavily_search_tool(self.api_keys.api_keys['tavily_key']), Memory_tool, find_images_tool, code_execution_tool], mcp_servers=mpc_servers, system_prompt="you are Cortana, a helpful assistant that can help with a wide range of tasks,\
                          you have the current time and the user query, you can use the tools provided to you if necessary to help the user with their queries, ask how you can help the user, sometimes the user will ask you not to use the tools, in this case you should not use the tools")
        self.memory=Message_state(messages=[])
        self.deps=Deps(agents_output={}, google_agent_output={},mail_inbox={})
    
    async def connect(self):
        """Establish persistent connection to MCP server"""
        if not self._is_connected:
            self._mcp_context_manager = self.agent.run_mcp_servers()
            await self._mcp_context_manager.__aenter__()
            self._is_connected = True
            return "Connected to MCP server"

    async def disconnect(self):
        """Close the MCP server connection"""
        if self._is_connected and self._mcp_context_manager:
            await self._mcp_context_manager.__aexit__(None, None, None)
            self._is_connected = False
            self._mcp_context_manager = None
            return "Disconnected from MCP server"
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
        self.memory.messages=result.all_messages()
        return result.output
    
    def reset(self):
        """
        Resets the Cortana agent to its initial state.

        Returns:
            str: A confirmation message indicating that the agent has been reset.
        """
        self.memory.messages=[]
        self.deps=Deps(agents_output={}, google_agent_output={},mail_inbox={})
        return f'Cortana has been reset'
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
