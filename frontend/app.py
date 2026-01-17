import streamlit as st
import requests
import time
import urllib3

# Disable SSL warnings for corporate proxy/Zscaler
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Page configuration
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for chat interface
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .main {
        padding: 0;
        max-width: 100%;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 8rem;
        max-width: 900px;
        margin: 0 auto;
    }
    
    /* Chat messages container */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    /* User message (right side) */
    .user-message {
        display: flex;
        justify-content: flex-end;
        margin: 0.5rem 0;
    }
    
    .user-message-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.25rem;
        border-radius: 18px 18px 4px 18px;
        max-width: 70%;
        word-wrap: break-word;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    /* AI message (left side) */
    .ai-message {
        display: flex;
        justify-content: flex-start;
        margin: 0.5rem 0;
    }
    
    .ai-message-content {
        background: #f7f7f8;
        color: #1f2937;
        padding: 1rem 1.25rem;
        border-radius: 18px 18px 18px 4px;
        max-width: 70%;
        word-wrap: break-word;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        border: 1px solid #e5e7eb;
    }
    
    /* Loader animation */
    .loader {
        display: flex;
        gap: 0.5rem;
        padding: 1rem;
    }
    
    .loader-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #9ca3af;
        animation: bounce 1.4s infinite ease-in-out both;
    }
    
    .loader-dot:nth-child(1) {
        animation-delay: -0.32s;
    }
    
    .loader-dot:nth-child(2) {
        animation-delay: -0.16s;
    }
    
    @keyframes bounce {
        0%, 80%, 100% {
            transform: scale(0);
        }
        40% {
            transform: scale(1);
        }
    }
    
    /* Input area */
    .stChatInputContainer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 1rem;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
        z-index: 1000;
    }
    
    /* Header */
    .chat-header {
        text-align: center;
        padding: 1.5rem 0;
        border-bottom: 2px solid #f0f0f0;
        margin-bottom: 2rem;
    }
    
    .chat-header h1 {
        font-size: 2rem;
        margin: 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "is_loading" not in st.session_state:
    st.session_state.is_loading = False

# Backend API URL
BACKEND_URL = "http://localhost:8000/chat"

def get_bot_response(user_message):
    """Send message to backend and get response"""
    try:
        response = requests.post(
            BACKEND_URL,
            json={"user_query": user_message},
            timeout=30,
            verify=False  # Disable SSL verification for corporate proxy/Zscaler
        )
        
        if response.status_code == 200:
            return response.json().get("response", "Sorry, I couldn't generate a response.")
        else:
            return f"Error: Unable to get response from server (Status: {response.status_code})"
    
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to the backend server. Please make sure it's running on http://localhost:8000"
    except requests.exceptions.Timeout:
        return "Error: Request timed out. Please try again."
    except Exception as e:
        return f"Error: {str(e)}"

# Header
st.markdown("""
<div class="chat-header">
    <h1>🤖 AI Chatbot</h1>
    <p style="color: #6b7280; margin-top: 0.5rem;">Ask me anything!</p>
</div>
""", unsafe_allow_html=True)

# Display chat messages
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <div class="user-message-content">
                    {message["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="ai-message">
                <div class="ai-message-content">
                    {message["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Show loader if waiting for response
    if st.session_state.is_loading:
        st.markdown("""
        <div class="ai-message">
            <div class="ai-message-content">
                <div class="loader">
                    <div class="loader-dot"></div>
                    <div class="loader-dot"></div>
                    <div class="loader-dot"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Chat input at the bottom
user_input = st.chat_input("Type your message here...", key="chat_input")

if user_input and not st.session_state.is_loading:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Set loading state
    st.session_state.is_loading = True
    
    # Rerun to show user message and loader
    st.rerun()

# Process bot response if loading
if st.session_state.is_loading:
    # Get the last user message
    last_user_message = st.session_state.messages[-1]["content"]
    
    # Get bot response from backend
    bot_response = get_bot_response(last_user_message)
    
    # Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    
    # Reset loading state
    st.session_state.is_loading = False
    
    # Rerun to show bot response
    st.rerun()

# Clear chat button in sidebar
with st.sidebar:
    st.title("Options")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.is_loading = False
        st.rerun()
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This is an AI-powered chatbot interface built with Streamlit and FastAPI.")
    st.markdown(f"**Total Messages:** {len(st.session_state.messages)}")
    
    st.markdown("---")
    st.markdown("### Backend Status")
    try:
        response = requests.get("http://localhost:8000/", timeout=2, verify=False)
        if response.status_code == 200:
            st.success("✅ Connected")
        else:
            st.error("❌ Not responding")
    except:
        st.error("❌ Disconnected")