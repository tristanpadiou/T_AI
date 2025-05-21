

from pydantic import BaseModel, Field
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext


@dataclass
class State:
    """
    A dictionnary representing the state of the agent.
    """
    node_message:str
    query: str
    inbox: dict
    current_draft: dict
    drafts: dict
    inbox_max_results: int
    route:str


