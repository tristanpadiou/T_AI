from pydantic_graph import BaseNode, End, GraphRunContext, Graph
from pydantic_ai import Agent
from pydantic_ai.common_tools.tavily import tavily_search_tool
from dataclasses import dataclass
from pydantic import Field, BaseModel
from typing import  List, Dict, Optional, Any

from tavily import TavilyClient
from IPython.display import Image, display
import time
# import nest_asyncio
# nest_asyncio.apply()
# load_dotenv()
# google_api_key=os.getenv('google_api_key')
# tavily_key=os.getenv('tavily_key')
# tavily_client = TavilyClient(api_key=tavily_key)
# llm=GeminiModel('gemini-2.0-flash', provider=GoogleGLAProvider(api_key=google_api_key))
# pse=os.getenv('pse')

@dataclass
class State:
    query:str
    preliminary_research: str
    research_plan: Dict
    research_results: Dict
    validation : str
    final: Dict
class paragraph_content(BaseModel):
    title: str = Field(description='the title of the paragraph')
    content: str = Field(description='the content of the paragraph')

class paragraph(BaseModel):
    title: str = Field(description='the title of the paragraph')
    should_include: str = Field(description='a description of what the paragraph should include')  
class Paper_layout(BaseModel):
    title: str = Field(description='the title of the paper')
    paragraphs: List[paragraph]= Field(description='the list of paragraphs of the paper')


class Deep_research_engine:
    def __init__(self,llm:any,api_keys:dict):
        """
        Args:
            llm (any): The language model to use using pydantic_ai
            api_keys (dict): The API keys to use
        """
        tavily_client = TavilyClient(api_key=api_keys['tavily_key'])



        paper_layout_agent=Agent(llm, output_type=Paper_layout, instructions="generate a paper layout based on the query, preliminary_search, search_results,include a Title for the paper, for the paragraphs only include the title, no content, no image, no table, start with introduction and end with conclusion")
        paragraph_gen_agent=Agent(llm, output_type=paragraph_content, instructions="generate a paragraph synthesizing the research_results based on the title,what the paragraph should include, and what has already been written to avoid repetition")
        class PaperGen_node(BaseNode[State]):
            async def run(self, ctx: GraphRunContext[State])->End:
                prompt=(f'query:{ctx.state.query}, preliminary_search:{ctx.state.preliminary_research},search_results:{ctx.state.research_results.research_results}')
                result=paper_layout_agent.run_sync(prompt)
                paragraphs=[]
                for i in result.output.paragraphs:
                    time.sleep(2)
                    paragraph_data=paragraph_gen_agent.run_sync(f'title:{i.title}, should_include:{i.should_include}, research_results:{ctx.state.research_results.research_results}, already_written:{paragraphs}')
                    paragraphs.append(paragraph_data.output.model_dump())

                paper={'title':result.output.title,
                        'paragraphs':paragraphs,
                        'references':ctx.state.research_results.references if ctx.state.research_results.references else None}

                ctx.state.final=paper
                        
                return End(ctx.state.final)






        class Research_results(BaseModel):
            research_results: List[str] = Field(default_factory=None,description='the research results')
            references: str = Field(default_factory=None,description='the references (urls) of the research_results')

        class Research_node(BaseNode[State]):
            async def run(self, ctx: GraphRunContext[State])->PaperGen_node:
                research_results=Research_results(research_results=[], references='')
                
                for i in ctx.state.research_plan.search_queries:
                    response = tavily_client.search(i.search_query)
                    data=[]
                    for i in response.get('results'):
                        if i.get('score')>0.50:
                            
                                    
                            data.append(i.get('url'))
                            research_results.research_results.append(i.get('content'))
                research_results.research_results=list(set(research_results.research_results))
                research_results.references=list(set(data))
                research_results.references=', '.join(research_results.references)
                ctx.state.research_results=research_results
                return PaperGen_node()

        class search_query(BaseModel):
            search_query: str = Field(description='the detailed web search query for the research')
            
        class Research_plan(BaseModel):
            search_queries: List[search_query] = Field(description='the detailed web search queries for the research')
        

        research_plan_agent=Agent(llm, output_type=Research_plan, instructions='generate a detailed research plan breaking down the research into smaller parts based on the query and the preliminary search')

        class Research_plan_node(BaseNode[State]):
            async def run(self, ctx: GraphRunContext[State])->Research_node:
                
                prompt=(f'query:{ctx.state.query}, preliminary_search:{ctx.state.preliminary_research}')
                result=research_plan_agent.run_sync(prompt)
                ctx.state.research_plan=result.output
                return Research_node()
            
            
        search_agent=Agent(llm, tools=[tavily_search_tool(api_keys['tavily_key'])], system_prompt="do a websearch based on the query")

        class preliminary_search_node(BaseNode[State]):
            async def run(self, ctx: GraphRunContext[State]) -> Research_plan_node:
                prompt = (' Do a preliminary search to get a global idea of the subject that the user wants to do reseach on as well as the necessary informations to do a search on.\n'
                        f'The subject is based on the query: {ctx.state.query}, return the results of the search.')
                result=search_agent.run_sync(prompt)
                ctx.state.preliminary_research=result.output
                return Research_plan_node()







        self.graph=Graph(nodes=[preliminary_search_node, Research_plan_node, Research_node, PaperGen_node])
        self.state=State(query='', preliminary_research='', research_plan=[], research_results=[], validation='', final='')
        self.preliminary_search_node=preliminary_search_node()

    def chat(self,query:str):
        """Chat with the deep research engine,
        Args:
            query (str): The query to search for
        Returns:
            str: The response from the deep research engine
        """
        self.state.query=query
        response=self.graph.run_sync(self.preliminary_search_node,state=self.state)
        return response.output


    def display_graph(self):
        """Display the graph of the deep research engine
        Returns:
            Image: The image of the graph
        """
        image=self.graph.mermaid_image()
        return display(Image(image))

