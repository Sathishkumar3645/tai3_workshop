import json
import logging
from app.utils import embedding
from langchain_community.vectorstores import Chroma
from app.core.config import settings

logger = logging.getLogger(__name__)


def retrieveDocument(query: str) -> str:
    """Retrieve relevant documents from Vector DB based on user query.
    
    Args:
        query: User's search query
        
    Returns:
        JSON string with search results or error
    """
    scope = "general"
    function_description = "Retrieve product information from the vector database based on user query."
    query_description = "Search query to find relevant products"
    
    try:
        logger.info(f"Retrieving documents for query: {query}")
        db = Chroma(persist_directory=settings.vectorDBPath, embedding_function=embedding)
        results = db.similarity_search_with_score(query, k=5)
        return str(results)
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        return json.dumps({"error": str(e)})