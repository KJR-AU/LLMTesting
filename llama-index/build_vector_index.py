import os
import openai
import nest_asyncio
import json
from llama_index.readers.file import UnstructuredReader
from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import Settings

def build_vector_index(years):
    loader = UnstructuredReader()
    Settings.chunk_size = 512
    for year in years:
        year_docs = loader.load_data(
            file=Path(f"./data/UBER/UBER_{year}.html"), split_documents=False
        )
        # insert year metadata into each year
        for d in year_docs:
            d.metadata = {"year": year}

        storage_context = StorageContext.from_defaults()
        cur_index = VectorStoreIndex.from_documents(
            year_docs,
            storage_context=storage_context,
        )
        storage_context.persist(persist_dir=f"./storage/{year}")


if __name__ == "__main__":
    with open("config.json") as f:
        config = json.load(f)

    os.environ["OPENAI_API_KEY"] = config['OPENAI_API_KEY']
    openai.api_key = config['OPENAI_API_KEY']
    nest_asyncio.apply()

    years = config['years']
    build_vector_index(years)