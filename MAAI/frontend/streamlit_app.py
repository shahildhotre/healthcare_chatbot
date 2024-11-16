import streamlit as st
import requests
from datetime import datetime

BACKEND_URL = "http://localhost:8000"

def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []

def display_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def get_bot_response(messages):
    try:
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json={"messages": messages}
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        return f"Error communicating with backend: {str(e)}"

def main():
    st.title("👨‍⚕️ HealthCare Chatbot")
    
    # Initialize session state
    init_session_state()
    
    # Display chat messages
    display_messages()
    
    # Chat input
    if user_input := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get bot response from backend
        bot_response = get_bot_response(st.session_state.messages)
        
        # Add bot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        
        # Rerun to display new messages
        st.rerun()

if __name__ == "__main__":
    main() 