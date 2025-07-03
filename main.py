import streamlit as st
from custom_chatbot import CustomLLMChatbot

# Initialize chatbot
@st.cache_resource
def get_chatbot():
    return CustomLLMChatbot(knowledge_base_path="data/your_docs.txt")

chatbot = get_chatbot()

# App title
st.set_page_config(page_title="Custom LLM Chatbot", page_icon="🤖")
st.title("🤖 Custom LLM Chatbot")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Ask me anything...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate response
    with st.chat_message("assistant"):
        response = chatbot.chat(user_input)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})