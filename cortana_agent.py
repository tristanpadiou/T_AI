from __future__ import annotations
from google_agent import Google_agent

from pydantic_ai import Agent, RunContext
from pydantic_ai.common_tools.tavily import tavily_search_tool
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from composio_langgraph import ComposioToolSet
from dataclasses import dataclass
from datetime import datetime
from pydantic import Field

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
    def __init__(self, api_keys:dict):
        """
        Args:
            
            api_keys (dict): The API keys to use as a dictionary
            
        """
        self.toolset=ComposioToolSet(api_key=api_keys['composio_key'])
        GEMINI_MODEL='gemini-2.0-flash'
        self.api_keys=Api_keys(api_keys=api_keys)
       
        # tools
        llms={'pydantic_llm':GoogleModel('gemini-2.5-flash-preview-05-20', provider=GoogleProvider(api_key=self.api_keys.api_keys['google_api_key'])),
              'langchain_llm':ChatGoogleGenerativeAI(google_api_key=self.api_keys.api_keys['google_api_key'], model=GEMINI_MODEL, temperature=0.3),
              'openai_llm':ChatOpenAI(api_key=self.api_keys.api_keys['openai_api_key'])}
        
        
        google_agent=Google_agent(llms,self.api_keys.api_keys, self.toolset)
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

            res=google_agent.chat(query)
            if google_agent.state.mail_inbox:
                ctx.deps.mail_inbox=google_agent.state.mail_inbox
            ctx.deps.google_agent_output=google_agent.state
            ctx.deps.agents_output['google_agent_tool']=google_agent.state.node_messages_dict
            try:
                return res.node_messages[-1]
            except:
                return res
        
        async def reset_google_agent_tool(ctx:RunContext[Deps]):
            """
            Use this tool to reset the google agent when it is not working as expected
            """
            google_agent.reset()
            return 'Google agent has been reset'


        async def web_search_tool(ctx: RunContext[Deps], query:str):
            """
            Use this tool to do a quick web search on a topic, to gather informations and data.
            Args:
                query (str): The query related to the web_search_tool and its capabilities
                
            Returns:
                str: The response from the web_search_tool
            """
            quick_research_agent=Agent(llms['pydantic_llm'], tools=[tavily_search_tool(self.api_keys.api_keys['tavily_key'])], instructions="do a websearch based on the query")
            result= quick_research_agent.run_sync(query)
            ctx.deps.agents_output['web_search_tool']=result.output
            return result.output
        
        async def Memory_tool(ctx: RunContext[Deps], query:str):
            """
            Use this tool to dive into the memory of the agents to answer questions based on previous information provided from previous tool calls.
            Args:
                query (str): The query related to the Memory_tool and its capabilities
                
            Returns:
                str: The response from the Memory_tool
            """
            history=ctx.deps.agents_output

            answer_question_agent=Agent(llms['pydantic_llm'], tools=[tavily_search_tool(self.api_keys.api_keys['tavily_key'])], instructions="answer the question based on the information provided")
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
        self.agent=Agent(llms['pydantic_llm'], output_type=Cortana_output, tools=[google_agent_tool, web_search_tool, Memory_tool, get_current_time_tool, reset_google_agent_tool], system_prompt="you are Cortana, a helpful assistant that can help with a wide range of tasks,\
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
