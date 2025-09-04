import streamlit as st
from openai import OpenAI


st.set_page_config(page_title="GPT-5-mini Chatbot", page_icon="🤖", layout="wide")


# Sidebar for API key input
st.sidebar.title("🔑 API Settings")
api_key = st.sidebar.text_input(
"Enter your OpenAI API key", type="password", placeholder="sk-..."
)


st.title("🤖 GPT-5-mini Chatbot")
st.write("Chat with OpenAI's GPT-5-mini. Enter your API key in the left panel to begin.")


# Initialize chat history if not exists
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# User input
if prompt := st.chat_input("Type your message..."):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)


if not api_key:
    st.warning("Please enter your API key in the sidebar.")
else:
    try:
        client = OpenAI(api_key=api_key)


        # Call GPT-5-mini
        response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=st.session_state.messages,
        )


        reply = response.choices[0].message.content


        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": reply})


        with st.chat_message("assistant"):
            st.markdown(reply)
    except Exception as e:
        st.stop()