from google_agent import Google_agent
from deep_research import Deep_research_engine
from pydantic_ai import Agent, RunContext
from pydantic_ai.common_tools.tavily import tavily_search_tool
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from dotenv import load_dotenv
from dataclasses import dataclass
from datetime import datetime
import os
from tavily import TavilyClient
load_dotenv()
google_api_key=os.getenv('google_api_key')
tavily_key=os.getenv('tavily_key')
tavily_client = TavilyClient(api_key=tavily_key)
llm=GeminiModel('gemini-2.0-flash', provider=GoogleGLAProvider(api_key=google_api_key))

google_agent=Google_agent()
deep_research_engine=Deep_research_engine()

@dataclass
class Message_state:
    messages: list[ModelMessage]

def google_agent_tool(query:str):
    """
    Use this tool to interact with the google agent, which can, search for images manage the user's calendar, emails, tasks, contacts, and more.
    Args:
        query (str): The entire query related to the google agent and its capabilities
    Returns:
        str: The response from the google agent
    """
    return google_agent.chat(query)

async def deep_research_tool(query:str):
    """
    Use this tool to do a deep research on a topic, to gather detailed informations and data.
    Args:
        query (str): The query related to the deep research and its capabilities
    Returns:
        str: The response from the deep research engine
    """
    response=await deep_research_engine.chat(query)
    return response

async def quick_research_tool(query:str):
    """
    Use this tool to do a quick research on a topic, to gather basic informations and data.
    Args:
        query (str): the quick research query
    Returns:
        str: The response from the quick research engine
    """
    quick_research_agent=Agent(llm, tools=[tavily_search_tool(tavily_key)], system_prompt="do a websearch based on the query")
    result=await quick_research_agent.run(query)
    return result.data

def get_current_time_tool():
    """
    Use this tool to get the current time.
    Returns:
        str: The current time in a formatted string
    """
   
    return f"The current time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

class Cortana_agent:
    def __init__(self):
        self.agent=Agent(llm, tools=[google_agent_tool, deep_research_tool, quick_research_tool, get_current_time_tool], system_prompt="you are Cortana, a helpful assistant that can help with a wide range of tasks,\
                          you can use the tools provided to you to help the user with their queries")
        self.memory=Message_state(messages=[])
    async def chat(self, query:str):
        result=await self.agent.run(query, message_history=self.memory.messages)
        self.memory.messages=result.all_messages()
        return result.data
