import streamlit as st
import LangChainHelper as lch
import textwrap

st.title("YouTube Assistant")
with st.sidebar:
    with st.form(key="my-form"):
        youtube_url = st.sidebar.text_area(label="What is the YouTube video url?", max_chars=50)
        query = st.sidebar.text_area(label="Ask me about the video..", max_chars=50, key="query")
        submit_btn = st.form_submit_button(label="Submit")

if query and youtube_url:
    db = lch.createVectorDBFromYouTubeURL(youtube_url)
    response = lch.get_resp_from_query(db,query)
    st.subheader("Answer:")
    st.text(textwrap.fill(response,width=80))