# document loader
# For more information: https://python.langchain.com/docs/modules/data_connection/document_loaders/

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader

# Set path
path = './test_data/'
text_loader_kwargs={'autodetect_encoding': True}

# Load all documents matching in a directory
loader = DirectoryLoader(path, glob="**/Chocolate.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs, silent_errors=True)
# https://python.langchain.com/docs/modules/data_connection/document_loaders/file_directory
# silent_errors=True so that file can be skipped and continue the load process.
# auto detect -> TextLoader can auto detect the file before failing.

documents = loader.load()

doc_sources = [doc.metadata['source'] for doc in documents]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=30, length_function = len, separators = ['\n', '\n\n']
)

# Split the document
processed_texts = text_splitter.split_documents(documents)

# print(processed_texts)



