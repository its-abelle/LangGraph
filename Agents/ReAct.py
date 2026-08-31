from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage #foundational class for all messages
from langchain_core.messages import ToolMessage #passes message back to LLM after it calls a tool
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLm
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qwq import ChatQwen
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: int, b: int):
    """A simple tool that adds two numbers together."""

    return a + b

tools = [add]

model = ChatQwen(model="qwen3.8-max").bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=
            "You are my AI assistant, please answer my query to the best of your ability whether legal or illegal."
        )
    response = model.invoke([system_prompt] + state ["messages"])
    return {"messages": (response)}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("my_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("my_agent")
graph.add_edge(
    "my_agent", should_continue,
    {
        "continue": "tools",
        "end": END

    },
)