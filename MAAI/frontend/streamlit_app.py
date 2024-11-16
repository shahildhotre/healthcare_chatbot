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
        # Get the latest user message
        latest_message = messages[-1]
        
        # Create the request payload with just the message content
        payload = {
            "message": latest_message["content"]  # Send just the string content
        }
        
        print("Sending payload:", payload)  # Debug print
        
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json=payload
        )
        response.raise_for_status()
        response_data = response.json()
        
        bot_message = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "content": response_data["response"]
        }
        return bot_message
    except requests.exceptions.RequestException as e:
        print(f"Error details: {str(e)}")  # Debug print
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "content": f"Error communicating with backend: {str(e)}"
        }

def main():
    st.title("👨‍⚕️ HealthCare Chatbot")
    
    # Initialize session state
    init_session_state()
    
    # Display chat messages
    display_messages()
    
    # Chat input
    if user_input := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Get bot response from backend
        bot_response = get_bot_response(st.session_state.messages)
        print("Bot Response:", bot_response)
        
        # Add bot response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": bot_response["content"],
            "timestamp": bot_response["timestamp"]
        })
        
        # Rerun to display new messages
        st.rerun()

if __name__ == "__main__":
    main() 