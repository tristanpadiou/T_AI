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
import json

from langchain_google_genai import ChatGoogleGenerativeAI
import nest_asyncio
nest_asyncio.apply()
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
            - Search for images
            - Manage user emails
            - Manage Google tasks
            - get contact list
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
                ctx.deps.agents_output['google_agent_tool'] = result
                
                return str(result.get('response').get('node_messages_list')[-1])
                
            except Exception as e:
                return f"Error calling Google Agent API: {str(e)}"
        
        async def reset_google_agent_tool(ctx:RunContext[Deps]):
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
                ctx.deps.agents_output['outlook_agent_tool'] = result
                
                return str(result.get('response').get('node_messages_list')[-1])
            except Exception as e:
                return f"Error calling Outlook Agent API: {str(e)}"

        async def reset_outlook_agent_tool(ctx:RunContext[Deps]):
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

        
        async def Memory_tool(ctx: RunContext[Deps], query:str):
            """
            Use this tool to dive into the memory of the agents to answer questions based on previous information provided from previous tool calls.
            Args:
                query (str): The query related to the Memory_tool and its capabilities
                
            Returns:
                str: The response from the Memory_tool
            """
            history=ctx.deps.agents_output

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
                

        @dataclass
        class Cortana_output:
            ui_version: str= Field(description='a markdown format version of the answer for displays if necessary')
            voice_version: str = Field(description='a conversationnal version of the answer for text to voice')
        self.agent=Agent(llms['pydantic_llm'], output_type=Cortana_output, tools=[tavily_search_tool(self.api_keys.api_keys['tavily_key']), google_agent_tool, Memory_tool, get_current_time_tool, reset_google_agent_tool, outlook_agent_tool, reset_outlook_agent_tool], system_prompt="you are Cortana, a helpful assistant that can help with a wide range of tasks,\
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
