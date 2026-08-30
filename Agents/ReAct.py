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


