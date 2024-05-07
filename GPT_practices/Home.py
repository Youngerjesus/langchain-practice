import streamlit as st
from langchain.prompts import PromptTemplate

st.set_page_config(
    page_title="FullstackGPT Home",
    page_icon="📚"
)

st.markdown("""
    # Hello
    
    Welcome to my FullstackGPT Portfolio! 
    
    Here are the apps I made: 
    
    - [DocumentGPT](/DocumentGPT)
    - [PrivateGPT](/PrivateGPT) 
    - [QuizGPT](/QuizGPT) 
    - [SiteGPT](/SiteGPT) 
    - [MeetingGPT](/MeetingGPT)
    - [InvestorGPT](/InvestorGPT)
""")

