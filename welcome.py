
import streamlit as st

st.title("About this App")
st.write("This is a streamlit app that act as a bot companion to help users relieve/ manage stress.")

with st.expander("How to use this App properly?"):
    st.markdown("""
    - Use the sidebar to navigate between pages.
    - On the **Main Page**, enter what you want to share with the bot.
    """)