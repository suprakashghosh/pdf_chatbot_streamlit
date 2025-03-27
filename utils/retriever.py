from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5") #Setting the embedding model to be BAAI/bge-small-en-v1.5. Default is OpenAIEmbedding which requires an OpenAI API Key



def create_vectorstore_retriever(directory= "./press_releases"):
    """
    Creates a VectorIndexRetriever for retrieving documents from a specified directory.

    This function reads documents from the given directory, splits the text into chunks,
    builds a vector store index, and configures a retriever to find the most similar documents.

    Args:
        directory (str): The path to the directory containing documents to be indexed. 
                         Defaults to "./press_releases".

    Returns:
        VectorIndexRetriever: A configured retriever for finding similar documents.
    """
    # Load documents from the specified directory
    documents = SimpleDirectoryReader(input_dir=directory).load_data()
    
    # Initialize a text splitter to divide documents into chunks
    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)
    
    # Set the text splitter in the global settings
    Settings.text_splitter = text_splitter

    # Build a vector store index from the documents using the text splitter
    index = VectorStoreIndex.from_documents(documents=documents, transformations=[text_splitter])

    # Configure the retriever to find the top 10 most similar documents
    retriever = VectorIndexRetriever(index=index, similarity_top_k=10)

    # Return the configured retriever
    return retriever


def extract_sources_and_text(most_similar_documents):
    
    """
    Extracts source information and concatenates text from the list of most similar documents extracted from the vectorstore retriever.
    This function iterates over a list of documents, extracting metadata such as file name and page number, and concatenates the text content of each document. 
    The extracted source information and concatenated text are returned as a tuple.

    Args:
        most_similar_documents (list): A list of document objects, each potentially containing metadata and text attributes.

    Returns:
        tuple: A tuple containing two elements:
            - sources_as_text (str): A string with source information including file names and page numbers.
            - concatenated_text (str): A string with the concatenated text content of all documents.
    """
    sources = []
    concatenated_text = ""
    
    if len(most_similar_documents) >0:
        for document in most_similar_documents:
            if hasattr(document, 'metadata') and 'file_name' in document.metadata:
                sources.append(f"File name: {document.metadata['file_name']}, page number: {document.metadata['page_label']}. Text Snippet- {document.text[:50]}...")
            if hasattr(document, 'text'):
                concatenated_text += document.text + "\n"

    sources_as_text= "\n\n".join(sources)
    return sources_as_text, concatenated_text
    
        