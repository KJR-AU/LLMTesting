# Load indices from disk
from llama_index.core import load_index_from_storage
from llama_index.core import StorageContext
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.agent.openai import OpenAIAgent
import json
import os
import openai

script_dir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(script_dir, "config.json")
with open(config_path) as f:
    config = json.load(f)

storage_dir = os.path.join(script_dir, config['storage-dir'])

os.environ["OPENAI_API_KEY"] = config['OPENAI_API_KEY']
openai.api_key = os.environ["OPENAI_API_KEY"]

# Load the cached data and create a query engine for each year which can be
# used by a chat model. 
index_set = {}
individual_query_engine_tools = []
for year in [2022, 2021, 2020, 2019]:
    storage_context = StorageContext.from_defaults(
        persist_dir=os.path.join(storage_dir, f"{year}")
    )
    cur_index = load_index_from_storage(
        storage_context,
    )
    index_set[year] = cur_index
    tool = QueryEngineTool(
        query_engine=index_set[year].as_query_engine(),
        metadata=ToolMetadata(
            name=f"vector_index_{year}",
            description=f"useful for when you want to answer queries about the {year} SEC 10-K for Uber",
        ),
    )
    individual_query_engine_tools.append(tool)

# Create a tool that can query filings across multiple years
query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=individual_query_engine_tools,
    llm=OpenAI(model="gpt-3.5-turbo"),
)

query_engine_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="sub_question_query_engine",
        description="useful for when you want to answer queries that require analyzing multiple SEC 10-K documents for Uber",
    ),
)

# Pass all of the tools to the chat model agent
tools = individual_query_engine_tools + [query_engine_tool]
agent = OpenAIAgent.from_tools(tools)
