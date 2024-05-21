import json
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser, output_parser

# StreamingStdOutCallbackHandler는 스트리밍 콜백을 표준 출력에 출력하기 위한 핸들러입니다.
# 이 핸들러는 모델의 출력을 스트리밍 방식으로 처리할 때, 그 출력을 표준 출력(stdout)에 실시간으로 출력합니다. 이를 통해 모델이 생성하는 응답을 실시간으로 모니터링할 수 있습니다.
from langchain.callbacks import StreamingStdOutCallbackHandler

st.set_page_config(
    page_title="QuizGPT",
    page_icon="1"
)

st.title("QuizGPT")

@st.cache_resource(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    loader = PyPDFLoader(file_path)
    spliter = RecursiveCharacterTextSplitter()
    docs = spliter.split_documents(loader.load())
    return docs

# Streamlit의 cache_data 기능은 함수의 결과를 캐싱하여 성능을 최적화하는 데 사용됩니다
# 캐싱은 동일한 입력에 대해 함수를 여러 번 호출할 때, 첫 번째 호출 시 계산된 결과를 저장하고 이후 동일한 입력에 대해 동일한 결과를 반환하여 불필요한 재계산을 방지합니다.
@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs):
    chain = {"context": questions_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)

@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


# streaming=True: 스트리밍 모드를 활성화합니다.
# callbacks=[StreamingStdOutCallbackHandler()]: 출력 스트림을 처리하기 위해 콜백 핸들러를 추가합니다.
# ChatOpenAI 클래스의 인스턴스를 생성할 때, 여러 가지 인자를 전달할 수 있습니다.
# 그중 callbacks는 모델이 출력을 생성할 때 호출되는 콜백 함수나 핸들러의 목록을 지정하는 인자입니다.
# 콜백은 모델의 각 출력 토큰을 생성할 때마다 실행되며, 이를 통해 출력을 실시간으로 처리하거나 로그를 남기는 등의 작업을 수행할 수 있습니다.
# ChatOpenAI 인스턴스를 생성할 때 callbacks 인자로 StreamingStdOutCallbackHandler의 인스턴스를 전달하여 모델의 출력을 실시간으로 표준 출력에 출력하도록 설정합니다. 이는 모델이 응답을 생성하는 과정에서 각 출력 토큰을 실시간으로 화면에 보여주는 역할을 합니다.
llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4o-2024-05-13",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

question_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", """
                You are a helpful assistant that is role playing as a teacher.
         
                Based ONLY on the following context make 10 (TEN) questions minimum to test the user's knowledge about the text.
    
                Each question should have 4 answers, three of them must be incorrect and one should be correct.
         
                Use (o) to signal the correct answer.
         
                Question examples:
                     
                Question: What is the color of the ocean?
                Answers: Red|Yellow|Green|Blue(o)
                     
                Question: What is the capital or Georgia?
                Answers: Baku|Tbilisi(o)|Manila|Beirut
                     
                Question: When was Avatar released?
                Answers: 2007|2001|2009(o)|1998
                     
                Question: Who was Julius Caesar?
                Answers: A Roman Emperor(o)|Painter|Actor|Model
                     
                Your turn!
                     
                Context: {context}
            """
        )
    ]
)

questions_chain = { "context": format_docs } | question_prompt | llm

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a powerful formatting algorithm.

    You format exam questions into JSON format.
    Answers with (o) are the correct ones.

    Example Input:
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)

    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut

    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998

    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model


    Example Output:

    ```json
    {{ "questions": [
            {{
                "question": "What is the color of the ocean?",
                "answers": [
                        {{
                            "answer": "Red",
                            "correct": false
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": false
                        }},
                        {{
                            "answer": "Green",
                            "correct": false
                        }},
                        {{
                            "answer": "Blue",
                            "correct": true
                        }},
                ]
            }},
                        {{
                "question": "What is the capital or Georgia?",
                "answers": [
                        {{
                            "answer": "Baku",
                            "correct": false
                        }},
                        {{
                            "answer": "Tbilisi",
                            "correct": true
                        }},
                        {{
                            "answer": "Manila",
                            "correct": false
                        }},
                        {{
                            "answer": "Beirut",
                            "correct": false
                        }},
                ]
            }},
                        {{
                "question": "When was Avatar released?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }},
                ]
            }},
            {{
                "question": "Who was Julius Caesar?",
                "answers": [
                        {{
                            "answer": "A Roman Emperor",
                            "correct": true
                        }},
                        {{
                            "answer": "Painter",
                            "correct": false
                        }},
                        {{
                            "answer": "Actor",
                            "correct": false
                        }},
                        {{
                            "answer": "Model",
                            "correct": false
                        }},
                ]
            }}
        ]
     }}
    ```
    Your turn!
    Questions: {context}
""",
        )
    ]
)

class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()

formatting_chain = formatting_prompt | llm

docs = None

with st.sidebar:
    choice = st.selectbox("Choose what you want to use", (
        "File",
        "Wikipedia Article"
    ))

    if choice == "File":
        file = st.file_uploader("Upload a .docx, .txt or .pdf file", type=["pdf", "txt", "docs"])

        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia")
        if topic:
            with st.status("Searching Wikipedia..."):
                docs = wiki_search(topic)


if not docs:
    st.markdown("""
        Welcome to QuizGPT. 
        
        I will make a quize from Wikipedia articles or files you upload to text your knowledge and help you study. 
        
        Get started by uploading a file or searching on Wikipedia in the sidebar. 
    
    """)

else:
    response = run_quiz_chain(docs)

    with st.form("questions_form"):
        for question in response["questions"]:
            st.write(question["question"])
            value = st.radio(
                "Select an option.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
            )
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
            elif value is not None:
                st.error("Wrong!")

        button = st.form_submit_button()