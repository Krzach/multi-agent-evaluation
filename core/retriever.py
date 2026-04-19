from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import FakeEmbeddings
from langchain_text_splitters import CharacterTextSplitter

class SimpleRetriever:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.vector_store = self._build_index()

    def _build_index(self):
        try:
            with open(self.data_path, "r") as f:
                text = f.read()
        except FileNotFoundError:
            text = "Default knowledge base."

        # Split text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        docs = text_splitter.create_documents([text])
        
        # Use FakeEmbeddings for testing without an API key. 
        # Replace with OpenAIEmbeddings() or similar for real usage.
        embeddings = FakeEmbeddings(size=128)
        
        # Create an in-memory FAISS vector store
        return FAISS.from_documents(docs, embeddings)
        
    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": 2})
