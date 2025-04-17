from google_agent import Google_agent
from deep_research import Deep_research_engine
from pydantic_ai import Agent, RunContext
from pydantic_ai.common_tools.tavily import tavily_search_tool
from pydantic_ai.messages import ModelMessage
from dataclasses import dataclass
from datetime import datetime
from pydantic import Field
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from langchain_google_genai import ChatGoogleGenerativeAI


@dataclass
class Message_state:
    messages: list[ModelMessage]

@dataclass
class Deps:
    deep_research_output: dict


class Cortana_agent:
    def __init__(self, api_keys:dict):
        """
        Args:
            
            api_keys (dict): The API keys to use as a dictionary
        """
        pydantic_llm=GeminiModel('gemini-2.0-flash', provider=GoogleGLAProvider(api_key=api_keys['google_api_key']))
        GEMINI_MODEL='gemini-2.0-flash'
        langchain_llm = ChatGoogleGenerativeAI(google_api_key=api_keys['google_api_key'], model=GEMINI_MODEL, temperature=0.3)
        # tools
        google_agent=Google_agent(langchain_llm,api_keys)
        deep_research_engine=Deep_research_engine(pydantic_llm,api_keys)
        
        def google_agent_tool(query:str):
            """
            Use this tool to interact with the google agent, which can, search for images, manage the user's calendar, emails, google tasks, contacts, and more.
            Args:
                query (str): The entire query related to the google agent and its capabilities
            Returns:
                str: The response from the google agent
            """
            return google_agent.chat(query)

       

        async def search_and_question_answering_tool(ctx: RunContext[Deps], query:str):
            """
            Use this tool to do a deep research on a topic, to gather detailed informations and data, answer_questions from the deep research results or do a quick research if the answer is not related to the deep research.
            Args:
                query (str): The query related to the search_and_question_answering_tool and its capabilities
                

            Returns:
                str: The response from the search_and_question_answering_tool
            """
            @dataclass
            class Route:
                answer: str = Field(default_factory=None,description="the answer to the question if the question is related to the deep research")
                route: str = Field(description="the route, either deep_research or answer_question, or quick_research")
            agent=Agent(pydantic_llm, result_type=Route, system_prompt="you are a router/question answering agent, you are given a query and you need to decide what to do based on the information provided")
            response=await agent.run(f"based on the query: {query}, and the information provided: {ctx.deps.deep_research_output if ctx.deps.deep_research_output else ''} either answer the question or if the answer is not related to the information provided or need more information, return 'quick_research' or 'deep_research'")
            route=response.data.route
            if route=='deep_research':
                response=await deep_research_engine.chat(query)
                ctx.deps.deep_research_output=response
                return response
            elif route=='answer_question':
                
                return response.data.answer
            elif route=='quick_research':
                quick_research_agent=Agent(pydantic_llm, tools=[tavily_search_tool(api_keys['tavily_key'])], system_prompt="do a websearch based on the query")
                result=await quick_research_agent.run(query)
                return result.data

        def get_current_time_tool():
            """
            Use this tool to get the current time.
            Returns:
                str: The current time in a formatted string
            """
        
            return f"The current time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        

        self.agent=Agent(pydantic_llm, tools=[google_agent_tool, search_and_question_answering_tool, get_current_time_tool], system_prompt="you are Cortana, a helpful assistant that can help with a wide range of tasks,\
                          you can use the tools provided to you to help the user with their queries")
        self.memory=Message_state(messages=[])
        self.deps=Deps(deep_research_output={})
        
    async def chat(self, query:str):
        result=await self.agent.run(query, deps=self.deps, message_history=self.memory.messages)
        self.memory.messages=result.all_messages()
        return result.data
