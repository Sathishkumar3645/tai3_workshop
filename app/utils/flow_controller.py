import logging
from app.utils.jinja_prompt import render_chat_prompt
from app.utils.llm_call import LLMTrigger
from app.utils.tool_constructor import LLMToolConstructor

logger = logging.getLogger(__name__)


def format_conversation_history(history: list) -> str:
    """Format conversation history into a readable string format.
    
    Args:
        history: List of dicts with 'role' and 'content' keys
        
    Returns:
        Formatted conversation history as string
    """
    return '\n'.join([f"{item['role']}: {item['content']}" for item in history])


def run_bot(provider: str, tools, user_query: str, user_type: str, conversation_history: list) -> str:
    """Execute chatbot flow: append query, generate prompt, call LLM, append response.
    
    Args:
        provider: LLM provider ('groq' or 'openai')
        tools: List of tool definitions for the LLM
        user_query: User's input query
        user_type: Type of user (e.g., 'general')
        conversation_history: List maintaining conversation history
        
    Returns:
        LLM response string
    """
    conversation_history.append({"role": "user", "content": user_query})
    
    formatted_history = format_conversation_history(conversation_history)
    prompt = render_chat_prompt(user_query, formatted_history)
    print("Generated Prompt:\n", prompt)
    tool_constructor = LLMToolConstructor(provider, user_type)
    tools = tool_constructor.main()
    
    llm = LLMTrigger(provider, tools, user_query, user_type, formatted_history, prompt)
    response = llm.main()
    
    conversation_history.append({"role": "assistant", "content": response})
    
    return response