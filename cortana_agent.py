from __future__ import annotations
import requests
import httpx
from pydantic_ai import Agent, RunContext
from pydantic_ai.common_tools.tavily import tavily_search_tool
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from dataclasses import dataclass
from datetime import datetime
from pydantic import Field
from google import genai
from google.genai import types
import json
import nest_asyncio
nest_asyncio.apply()
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
    def __init__(self, api_keys:dict, google_agent_api_url:str = "https://wolf1997-google-agent-api.hf.space", outlook_agent_api_url:str = "https://wolf1997-outlook-agent-api.hf.space"):
        """
        Args:
            
            api_keys (dict): The API keys to use as a dictionary
            google_agent_api_url (str): The URL of the Google Agent API
            outlook_agent_api_url (str): The URL of the Outlook Agent API
            
        """
        GEMINI_MODEL='gemini-2.0-flash'
        self.api_keys=Api_keys(api_keys=api_keys)
        self.google_agent_api_url = google_agent_api_url
        self.outlook_agent_api_url = outlook_agent_api_url
       
        # tools
        llms={'pydantic_llm':GoogleModel('gemini-2.5-flash-preview-05-20', provider=GoogleProvider(api_key=self.api_keys.api_keys['google_api_key'])),
              'langchain_llm':ChatGoogleGenerativeAI(google_api_key=self.api_keys.api_keys['google_api_key'], model=GEMINI_MODEL, temperature=0.3),
              'openai_llm':ChatOpenAI(model='gpt-4.1-nano',api_key=self.api_keys.api_keys['openai_api_key'])}
        async def google_agent_tool(ctx:RunContext[Deps],query:str):
            """
            # Google Agent Interaction Function

            ## Purpose
            This function provides an interface to interact with a Google agent that can perform multiple Google-related tasks simultaneously but it cannot do 
            websearches.

            ## Capabilities
            The agent can:
            - Manage user emails 
            - Manage Google tasks 
            - get contact list (get contact details)
            - Manage Google Calendar
            - List available tools
            - Improve planning based on user feedback with planning notes
            - Improve its query based on user feedback with query notes

            ## Parameters
            - `query` (str): A complete query string describing the desired Google agent actions
            - The query should include all necessary details for the requested operations
            - Multiple actions can be specified in a single query

            ## Returns
            - `str`: The agent's response to the query

            ## Important Notes
            - The agent can process multiple actions in a single query
            - User feedback can be provided to help improve the agent's planning and query
            - All Google-related operations should be included in the query string

            """
            try:
                # Make API call to Google Agent
                data = {
                    'query': query,
                    'google_api_key': self.api_keys.api_keys['google_api_key'],
                    'openai_api_key': self.api_keys.api_keys['openai_api_key'],
                    'composio_key': self.api_keys.api_keys['composio_key'],
                    'pse': self.api_keys.api_keys.get('pse', '')
                }
                
                response = requests.post(f"{self.google_agent_api_url}/chat", data=data)
                response.raise_for_status()
                result = response.json()
                
                # Store the response in context
                ctx.deps.agents_output['google_agent_tool'] = result.get('response').get('node_messages_dict')
                
                return str(result.get('response').get('node_messages_list')[-1])
                
            except Exception as e:
                return f"Error calling Google Agent API: {str(e)}"
        
        async def reset_google_agent_tool():
            """
            Use this tool to reset the google agent when it is not working as expected
            """
            try:
                response = requests.post(f"{self.google_agent_api_url}/reset")
                response.raise_for_status()
                result = response.json()
                return result.get('message')
            except Exception as e:
                return f"Error resetting Google Agent: {str(e)}"

        async def outlook_agent_tool(ctx:RunContext[Deps], query:str):
            """
            # Outlook Agent Interaction Function

            ## Purpose
            This function provides an interface to interact with an Outlook agent that can perform Microsoft 365 tasks.

            ## Capabilities
            The agent can:
            - Read and manage Outlook emails
            - Create and manage Microsoft Tasks/To-Do
            - Manage Outlook Calendar events
            - Manage Outlook Contacts
            - Access Microsoft Graph API services

            ## Parameters
            - `query` (str): A complete query string describing the desired Outlook agent actions
            - The query should include all necessary details for the requested operations
            - Multiple actions can be specified in a single query

            ## Returns
            - `str`: The agent's response to the query

            ## Important Notes
            - The agent can process multiple actions in a single query
            - All Microsoft 365 operations should be included in the query string
            """
            try:
                # Make API call to Outlook Agent
                data = {
                    'query': query,
                    'google_api_key': self.api_keys.api_keys['google_api_key'],
                    'openai_api_key': self.api_keys.api_keys['openai_api_key'],
                    'composio_key': self.api_keys.api_keys['composio_key']
                }
                
                response = requests.post(f"{self.outlook_agent_api_url}/chat", data=data)
                response.raise_for_status()
                result = response.json()
                
                # Store the response in context
                ctx.deps.agents_output['outlook_agent_tool'] = result.get('response').get('node_messages_dict')
                
                return str(result.get('response').get('node_messages_list')[-1])
            except Exception as e:
                return f"Error calling Outlook Agent API: {str(e)}"

        async def reset_outlook_agent_tool():
            """
            Use this tool to reset the outlook agent when it is not working as expected
            """
            try:
                response = requests.post(f"{self.outlook_agent_api_url}/reset")
                response.raise_for_status()
                result = response.json()
                return result.get('message')
            except Exception as e:
                return f"Error resetting Outlook Agent: {str(e)}"

        
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

        async def get_current_time_tool():
            """
            Use this tool to get the current time.
            Returns:
                str: The current time in a formatted string
            """
            return f"The current time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
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
            Use this tool to execute code to answer math questions or any other questions that require code execution
            args: query
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

            
            return f'the result of the code execution for {query} is {res}'
        @dataclass
        class Cortana_output:
            ui_version: str= Field(description='a markdown format version of the answer for displays if necessary')
            voice_version: str = Field(description='a conversationnal version of the answer for text to voice')
        self.agent=Agent(llms['pydantic_llm'], output_type=Cortana_output, tools=[tavily_search_tool(self.api_keys.api_keys['tavily_key']), google_agent_tool, Memory_tool, get_current_time_tool, reset_google_agent_tool, outlook_agent_tool, reset_outlook_agent_tool, find_images_tool, code_execution_tool], system_prompt="you are Cortana, a helpful assistant that can help with a wide range of tasks,\
                          you can use the tools provided to you if necessary to help the user with their queries, ask how you can help the user, sometimes the user will ask you not to use the tools, in this case you should not use the tools")
        self.memory=Message_state(messages=[])
        self.deps=Deps(agents_output={}, google_agent_output={},mail_inbox={})
    
    def chat(self, query:any):
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

        result=self.agent.run_sync(query, deps=self.deps, message_history=self.memory.messages)
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
