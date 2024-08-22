# streamlit run streamlit_project.py

import streamlit as st
from langchain_core.messages.chat import ChatMessage
# from LangChain_InMemory_ver2 import *

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_teddynote import logging
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from operator import itemgetter
from langchain_core.runnables.history import RunnableWithMessageHistory
import pymupdf4llm
from langchain_text_splitters import MarkdownTextSplitter
from langchain_teddynote.prompts import load_prompt
from langchain import hub
import os
st.title("Hello World")

def get_retriever(): 
    # 단계 1: 문서 로드(Load Documents)
    
    loader = PyMuPDFLoader("../../docs/SPRi_AI_Brief_6월호_산업동향_최종.pdf")
    # loader = PyMuPDFLoader("../../docs/REN_r01uh0495ej0100_rx634_MAH_20150225.pdf")
    docs = loader.load()
    print(f"문서의 페이지수: {len(docs)}")
    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)
    print(f"분할된 청크의수: {len(split_documents)}")
    
    
    # md_docs = pymupdf4llm.to_markdown("../../docs/REN_r01uh0495ej0100_rx634_MAH_20150225.pdf")
    # md_docs = pymupdf4llm.to_markdown("../../docs/SPRi_AI_Brief_6월호_산업동향_최종.pdf")
    # print("-- MD_DOCS --", md_docs[1:10])    
    # splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=20)
    # split_documents = splitter.create_documents([md_docs])
    
    # from langchain_community.embeddings import HuggingFaceEmbeddings
    
    # Path to your local model
    local_model_path = "h:/RAG/rag_ollama/embedding_model/multilingual-e5-large-instruct"
    # Instantiate the HuggingFaceEmbeddings class
    embeddings = HuggingFaceEmbeddings(model_name=local_model_path)

    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore_index = "rx634_datasheet_vectorstore"
    if not os.path.exists(vectorstore_index) :    
        vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
        vectorstore.save_local(vectorstore_index)
    else :
        vectorstore = FAISS.load_local(vectorstore_index, embeddings,
                                allow_dangerous_deserialization=True)
        
    print("end of vectorstore...")
    
    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever()
    print("end of retriever...")
    return retriever

def get_llm():
    # llm = ChatOllama(
    #     model="EEVE-Korean-10.8B-Q5:latest",
    #     temperature=0.0,
    #     num_gpu=1,
    # )
    
    llm = ChatOllama(
        model="llama3.1:8b",
        temperature=0.1,
        num_gpu=1,
    )
    return llm

def get_promptTemplate(prompt_option):
    # 기본 템플릿
    prompt = PromptTemplate.from_template(
        """
            You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question or ask for more information. 
            If you don't know the answer, just say that you don't know or show me the context that might be relevant. 
            If you think you have a guess that might be close to the answer, feel free to share it as well.
            Answer in Korean and detail.\n
            
            #Previous Chat History:{chat_history}
            
            #Question:{question}
            
            #Retrieved Context:{context}
            
            #Answer:
        """
    )
    
    if prompt_option == "SNS 게시글" :
        prompt = load_prompt("prompts/sns.yaml", encoding="utf-8")
        
    elif prompt_option == "요약" :
        prompt = load_prompt("prompts/summary.yaml", encoding="utf-8")
        
    print(prompt)
    return prompt

def get_session_history(session_ids):
    print(f"[대화 세션ID]: {session_ids}")
    if session_ids not in st.session_state["store"]:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]  # 해당 세션 ID에 대한 세션 기록 반환


def create_chain():
    prompt = get_promptTemplate(prompt_option_box)
    llm = get_llm()
    retriever = get_retriever()
    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# -- LOAD .env --#
load_dotenv()
logging.langsmith("LangChain_Streamlit")
# -------------- #

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "RAG_WITH_HISTORY" not in st.session_state:
    st.session_state["RAG_WITH_HISTORY"] = []
    
if "store" not in st.session_state:
    st.session_state["store"] = {}

## side bar
with st.sidebar:
    clear_btn = st.button("대화 초기화")

    ### select box
    prompt_option_box = st.selectbox(
        "Select prompt Template...",
        ("기본", "SNS 게시글", "요약"), index = 0
    )

# -------------- #
# 새로운 메시지 추가
def add_message(role, message):
    st.session_state["messages"].append(
        ChatMessage(role=role, content=message)
    )

# 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)
        print()
        

# 처음 한번만 수행되게

def create_rag_with_history():
    chain = create_chain()
    rag_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,  # 세션 기록을 가져오는 함수
        input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
        history_messages_key="chat_history",  # 기록 메시지의 키
    )
    return rag_with_history

if clear_btn : 
    # ALL CLEAR!
    st.session_state["message"] = []
    st.session_state["store"] = {}
    st.session_state["RAG_WITH_HISTORY"] = []

print_messages() # 이전 대화 출력

user_input = st.chat_input("Say something")
if user_input: 
    st.chat_message("user").write(user_input)
    print_messages()
    rag_with_history = create_rag_with_history()
    # ai_answer= rag_with_history.invoke(
    #     # 질문 입력
    #     {"question": user_input},
    #     # 세션 ID 기준으로 대화를 기록합니다.
    #     config={"configurable": {"session_id": "rag123"}},
    # )
    
    response = rag_with_history.stream(
        # 질문 입력
        {"question": user_input},
        # 세션 ID 기준으로 대화를 기록합니다.
        config={"configurable": {"session_id": "rag123"}},
    )
    
    with st.chat_message("assistant") :
        # 빈 컨테이너를 만들어서, 여기에 토큰을 스트리밍 출력 
        container = st.empty()
        
        ai_answer = ""
        for token in response :
            ai_answer += token
            container.markdown(ai_answer)
        
    # st.chat_message("assistant").write(ai_answer)
    add_message("user", user_input)
    add_message("assistant", ai_answer)
 
