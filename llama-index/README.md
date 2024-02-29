# LlamaIndex RAG Application
This is an example of how to build a RAG chatbot application using Llama Index.

1. Install requirements

    ```pip install -r requirements.txt```

2. Set your open AI api key in config.json

3. Run the setup_data script to download and unzip the dataset.

    ```bash setup_data.sh```

4. Load the data into a vector store index

    ```python build_vector_index.py```

5. Run the chatbot

    ```python chatbot.py```

### Tests
A simple testing suite has been set up using [deepeval](https://docs.confident-ai.com/docs/getting-started). The tests are executed through pytest.

```
cd tests
deepeval test run test_chat_agent.py
```

#### Approach
A list of inputs are defined in a JSON file in the `tests/dataset` directory. These inputs are passed into the RAG agent, then used alongside the response and context to form test cases. The test cases are then evaluated against metrics defined in the `tests/metrics` directory. 