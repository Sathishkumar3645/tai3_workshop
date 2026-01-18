import logging
from fastapi import APIRouter
from app.utils.vectordb_gen import VectorDBGenerator
from app.schemas.chat_schema import ChatRequest
from app.utils.flow_controller import run_bot

logger = logging.getLogger(__name__)

router = APIRouter()

conversation_history = []

@router.get("/", tags=["Root"])
def root():
    """Root endpoint to verify API is accessible."""
    return {
        "message": "AI Chatbot API",
        "status": "running",
        "version": "1.0.0"
    }

@router.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint to verify API is running."""
    return {"status": "ok"}


@router.post("/create_vectorDB", tags=["VectorDB"])
def create_vectorDB():
    """Create and persist vector database from product catalog."""
    try:
        response = VectorDBGenerator().generate_vector_db()
        return {"status": response}
    except Exception as e:
        logger.error(f"Error creating vector DB: {str(e)}")
        return {"status": f"Error: {str(e)}"}


@router.post("/chat", tags=["Chat"])
def chat(request: ChatRequest):
    """Chat endpoint with multi-turn conversation support and tool calling.
    
    Maintains conversation history and calls LLM with tool definitions.
    """
    global conversation_history
    try:
        provider = "openai"
        user_type = "general"
        response = run_bot(provider, None, request.user_query, user_type, conversation_history)
        logger.info(f"Chat response generated for query: {request.user_query[:50]}...")
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return {"response": f"Error: {str(e)}"}