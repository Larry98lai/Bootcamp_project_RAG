import sys

# Force Python to use pysqlite3 instead of the system sqlite3
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

import streamlit as st
import random
import time
import sys, os, re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logics.user_query_handler import response_generator_from_crewai

st.title("AI-Powered Mental Wellness Companion")

st.markdown("""
Welcome to the **StressLess Bot 🧘🏻‍♂️😌🍃 - Your Work Wellness Chatbot**!  
You can use this chatbot if you are feeling down at work, or if you need guidance on how to approach mental health in the workplace! 
""")

# Section titles for structured display
section_titles = [
    "Emotional Support",
    "Actionable Plan",
    "Mentoring Tips",
    "Recommended Resources"
]

def display_structured_response(response_text):
    sections = {}
    pattern = "|".join([re.escape(title) for title in section_titles])
    section_pattern = re.compile(rf"(?P<title>{pattern})(?P<content>.*?)(?=(?:{pattern})|$)", re.DOTALL)

    for match in section_pattern.finditer(response_text):
        title = match.group("title").strip()
        content = match.group("content").strip()
        if content:
            sections[title] = content

    if sections:
        for title in section_titles:
            if title in sections:
                st.subheader(title)
                st.markdown(sections[title])
    else:
        st.markdown(response_text)
        
#initialise chat history - session_state helps to store data persistently across reruns of the app (like chat history)
if "messages" not in st.session_state:
    st.session_state.messages = []

#display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            display_structured_response(message["content"])
        else:
            st.markdown(message["content"])

#accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    #display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    full_response = "".join(response_generator_from_crewai(prompt))

    #add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    with st.chat_message("assistant"):

        display_structured_response(full_response)
