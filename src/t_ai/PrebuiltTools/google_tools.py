
import requests
from google import genai
from google.genai import types
from pydantic_ai.tools import Tool
from dataclasses import dataclass


class GoogleImageClient:
    def __init__(self,api_key:str,search_engine_id:str):
        self.api_key=api_key
        self.search_engine_id=search_engine_id
    async def search_images(self,query:str,max_results:int=1):
           # Define the API endpoint for Google Custom Search
        url = "https://www.googleapis.com/customsearch/v1"
        

        params = {
            "q": query,
            "cx": self.search_engine_id,
            "key": self.api_key,
            "searchType": "image",  # Search for images
            "num": max_results  # Number of results to fetch
        }

        # Make the request to the Google Custom Search API
        response = requests.get(url, params=params)
        data = response.json()

        # Check if the response contains image results
        if 'items' in data:
            # Extract the first image result
            if max_results==1:
                image_url = data['items'][0]['link']
                res=f'image url for {query} : {image_url}'
            else:
                res=[]
                for item in data['items']:
                    res.append(item['link'])
            return f'image urls for {query} : {res}'
        else:
            return 'no image found'

@dataclass
class ImageSearchTool:
    client:GoogleImageClient
    async def __call__(self,query:str,max_results:int=1):
        """Search for images using this tool, this tool can search multiple images at a time
        args: query (str): the query to search for
        max_results (int): the maximum number of results to fetch
        return: image url or list of image urls
        """
        result= await self.client.search_images(query,max_results)
        return result


def search_images_tool(api_key:str,search_engine_id:str):
    """
    Search for images using this tool, this tool can search multiple images at a time
    args: 
        api_key (str): the api key for google custom search
        search_engine_id (str): the custom search engine id
    
    """
    return Tool(
        ImageSearchTool(client=GoogleImageClient(api_key,search_engine_id)).__call__,
        name='search_images',
        description='Search for images using this tool, this tool can search multiple images at a time',
    )
 
@dataclass
class CodeExecutionTool:
    """
    Use this tool to answer math questions or any other questions that require code execution, it can handle complex math problems and code execution.
    args: api_key (str): the api key for gemini api
    return: the result of the code execution
    """
    client:genai.Client
    async def __call__(self,query:str):
        """
        Use this tool to answer math questions or any other questions that require code execution, it can handle complex math problems and code execution.
        args: query (str): the detailed query 
        api_key (str): the api key for gemini api
        return: the result of the code execution
        """
        response= self.client.models.generate_content(
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
        return f'the result of the code execution is {res.get("output") if res.get("output") else "no result"}'
    
def code_execution_tool(api_key:str):
    """
    Use this tool to answer math questions or any other questions that require code execution, it can handle complex math problems and code execution.
    args: api_key (str): the api key for gemini api
    return: the result of the code execution
    """
    return Tool(
        CodeExecutionTool(client=genai.Client(api_key=api_key)).__call__,
        name='code_execution',
        description='Use this tool to answer math questions or any other questions that require code execution, it can handle complex math problems and code execution.',
    )




