# LLMTesting
A selection of tools and example projects for testing Large Language Models


### Set Up
Watch this YouTube https://www.youtube.com/watch?v=Cq08yTa8dQU
1. OPENAI_API_KEY: To get this key, you need to create an account and create API key from https://platform.openai.com/api-keys and define it in the .env file
2. set up interpreter
3. python -m pip install -r requirements.txt
4. Start Exploring

### Information
1. OpenAI (default "gpt-3.5-turbo" model is used, "text-embedding-ada-002" embedding ) is used
2. Chroma vectorstore is used by default (https://docs.trychroma.com/reference/Collection)
3. Langchain is used as RAG/LLM model framework


### evaluation
This repository has several evaluation framework
1. TruLens
    - pip install trulens-eval
    - guide: https://www.trulens.org/trulens_eval/langchain_quickstart/
2. Deepeval
    - pip install -U deepeval
    - guide: https://docs.confident-ai.com/docs/getting-started
3. Langchain
    - https://python.langchain.com/docs/guides/evaluation/

