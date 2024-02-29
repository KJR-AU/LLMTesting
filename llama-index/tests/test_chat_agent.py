import json
import sys
sys.path.append("../")
from agent import agent
from deepeval.test_case import LLMTestCase
from deepeval import evaluate, assert_test
from metrics import metrics

def load_inputs(dataset_path: str):
    with open(dataset_path) as f:
        dataset = json.load(f)
    return [data['input'] for data in dataset]

inputs = load_inputs("dataset/dataset.json")
test_cases = []
for input in inputs:
    response = agent.query(input)
    
    response_string = response.response
    retrieval_context = [node.get_content() for node in response.source_nodes]
    
    test_case = LLMTestCase(
        input=input,
        actual_output=response_string,
        retrieval_context=retrieval_context
    )
    assert_test(test_case, metrics)