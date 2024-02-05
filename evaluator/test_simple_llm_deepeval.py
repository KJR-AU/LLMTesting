# %% [markdown]
# This is a simple chat model without history

# %%
from dotenv import load_dotenv
import os
from deepeval.tracing import trace, TraceType

load_dotenv()

# Set environment
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["DEEPEVAL_RESULTS_FOLDER"] = "./data"

# %%
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
model = ChatOpenAI(model="gpt-3.5-turbo")

@trace(type=TraceType.EMBEDDING, name="Embedding", model="text-embedding-ada-002")
def get_embedding():
    return OpenAIEmbeddings()

embedding = get_embedding()

# %%
from langchain_community.vectorstores import Chroma
import document_handler

# https://python.langchain.com/docs/integrations/vectorstores/chroma

chroma_collection_name = "LangChainCollection"

print(Chroma)
import chromadb
new_client = chromadb.EphemeralClient()

# collection = new_client.get_or_create_collection = chroma_collection_name
# collection.query()
def setup_vector_store():

    vectorstore_initialize = Chroma.from_documents(
        document_handler.processed_texts,
        embedding=embedding,
        collection_name=chroma_collection_name,
        client=new_client,
    )

    vectorstore = Chroma(
        client=new_client,
        collection_name=chroma_collection_name,
        embedding_function=embedding,
    )
    
    retriever = set_retriever()

@trace(type=TraceType.RETRIEVER, name="Retriever")
def set_retriever():
    return vectorstore.as_retriever()

retriever = set_retriever()

# %%
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, tool
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# %% [markdown]
# Feedback function trulens

# %%
# Prompt

system_message_template = (
    "You are a helpful assistant who helps answer questions. Answer only the facts based on the context. "
    "Your goal is to provide accurate and relevant answers based on the facts in the provided context. "
    "Make sure to reference the above source documents appropriately and avoid making assumptions or adding personal opinions. "
    "Emphasize the use of facts from the provided source documents. "
    "Instruct the model to use source name for each fact used in the response. "
    "Avoid generating speculative or generalized information. "
    "Use square brackets to reference the source, e.g. [info1.txt]. "
    "Do not combine sources, list each source separately, e.g. [info1.txt][info2.pdf].\n"
    "Here is how you should answer every question:\n"
        "-Look for relevant information in the above source documents to answer the question.\n"
        "-If the source document does not include the exact answer, please respond with relevant information from the data in the response along with citation. You must include a citation to each document referenced.\n"
        "-If you cannot find answer in below sources, respond with I am not sure. Do not provide personal opinions or assumptions and do not include citations.\n"
        "-If you use any information in the context, include the index(starts at 1) of the statement as citation in your answer\n"
    "At the end of your response:\n" 
    "1. Add key words from the paragraphs. \n"
    "2. Suggest a further question that can be answered by the paragraphs provided. \n"
    "3. Create a source list of source name, author name, and a link for each document you cited.\n"
    "{context}"
)

MEMORY_KEY = "chat_history"

@trace(type=TraceType.TOOL, name="Format")
def format_prompt():
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message_template),
            # MessagesPlaceholder(variable_name=MEMORY_KEY),
            ("human", "{question}"),
            # MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    return final_prompt


final_prompt = format_prompt()


# %%
chat_history = []

# %%
functions = [
    {
        "name": "response_with_source",
        "description": "The llm's what's needed to provide response",
        "parameters": {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "The question"},
                "context": {
                    "type": "string",
                    "description": "The context used to provide response",
                },
                "output": {"type": "string", "description": "The response"}
            },
            "required": ["input", "output", "context"],
        },
    }
]

# %%
# Set llm chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser, JsonOutputFunctionsParser
from langchain_core.output_parsers import JsonOutputParser

from deepeval.tracing import trace, TraceType
import openai

@trace(type=TraceType.LLM, name="OpenAI", model="gpt-3.5-turbo")
def define_chain():
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | final_prompt
        | model.bind(function_call={"name": "response_with_source"}, functions=functions)
        | JsonOutputFunctionsParser()
        )
    return rag_chain

rag_chain = define_chain()
# %%
# query = "What is chocolate?"
# result1 = rag_chain.invoke(query)
# print(result1)
# print (result1.input)

# %%
# print (result1["input"])

# %%
queries = []
# from langchain_community.document_loaders.csv_loader import CSVLoader
import pandas as pd

df = pd.read_csv('./test_data/Questions.csv', delimiter=',')
tuples = [tuple(x) for x in df.values]
dicts = df.to_dict('records')

print(dicts)

questions = list(map(lambda x : x['Question'], dicts))
print(questions)

# %%
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import HallucinationMetric
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric, AnswerRelevancyMetric
from deepeval import assert_test
from deepeval import run_test
from deepeval import evaluate

# %%
import deepeval
@deepeval.set_hyperparameters
def hyperparameters():
    return {
        "temperature": 0.7,
        "chunck_size": 1000,
        "model": "gpt-3.5-turbo",
        "prompt_template": final_prompt,
    }

# %%
# One question
# query = "What is chocolate?"
# result = rag_chain.invoke(query)
# print(result)

# test_case = LLMTestCase(
#     input=final_prompt.format(question=result["input"], context=result["context"]),
#     actual_output=result["output"],
#     retrieval_context = result["context"]
# )

# %%
# The hallucination metric determines whether your LLM generates factually correct information by comparing the actual_output to the provided context.

# input
# actual_output
# context
metric = HallucinationMetric(threshold=0.5)

# %%
# metric.measure(test_case)
# print(metric.score)

# %%
# from deepeval.metrics import ToxicityMetric
# from deepeval.test_case import LLMTestCase

# metric = ToxicityMetric(threshold=0.5)
# test_case = LLMTestCase(
#     input="What if these shoes don't fit?",
#     # Replace this with the actual output from your LLM application
#     actual_output = "We offer a 30-day full refund at no extra cost."
# )

# metric.measure(test_case)
# print(metric.score)

# %%
# How relevant the answer is to the input 
# QA Relevance - Response relevant to input? - TruLens matching
metricQR = AnswerRelevancyMetric(threshold=0.5, model="gpt-3.5-turbo", include_reason=True)
# Whether the output is factually aligning with contents of retrieved contexts
# Groundedness - is response relevant to context? - TruLens matching
metricF = FaithfulnessMetric(threshold=0.5, model="gpt-3.5-turbo", include_reason=True)
# Relevancy of the retrieved context to the input
# Context Relevance - Is context relevant to input? - TruLens matching
metricCR = ContextualRelevancyMetric(threshold=0.5, model="gpt-3.5-turbo", include_reason=True)


# %%
# metric.measure(test_case)
# print(metric.score)
# print(metric.reason)

# %%
from deepeval import evaluate
from deepeval import assert_test
from deepeval import track

metrics = [metricQR, metricF, metricCR]

test_cases = []

questions = [questions[0]]

for question in questions:
    result = rag_chain.invoke(question)
    print(result)

    question=result.get("input", None)
    actual_output=result.get("output", None)
    retrieval_context = [result.get("context", None)]
    input = final_prompt.format(question=result["input"], context=result["context"])

    print(input)
    print(actual_output)
    print(retrieval_context)

    test_case = LLMTestCase(
        input=final_prompt.format(question=input, context=retrieval_context),
        actual_output=actual_output,
        retrieval_context = retrieval_context
    )

    # print(assert_test(test_case, metrics))
    test_cases.append(test_case)

print(evaluate(test_cases, metrics))


