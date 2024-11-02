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
from langchain_huggingface import HuggingFacePipeline  # for huggingface local model

# 앙상블리트리버
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# CHROMA
from langchain_chroma import Chroma

import glob
import os
st.title("PDF 문서 QA")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    
    # 단계 3: 임베딩 생성(Create Embeddings)s
    # Path to your local model
    local_model_path = "../embedding_model/multilingual-e5-large-instruct"
    # Instantiate the HuggingFaceEmbeddings class
    # embeddings = HuggingFaceEmbeddings(
    #     model_name=local_model_path,
    #     model_kwargs={"device":"mps"}
    # )
    
    # 허깅 페이스에서 다운로드 받아서 임베딩 진행
    embedding_model_name = "intfloat/multilingual-e5-large-instruct"
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device":"mps"},
        encode_kwargs={"normalize_embeddings":True},
    )
    
    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore_index = "vectorstore_index"
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

def get_retriever_for_file_upload(upload_file_path=None): 
    # 단계 1: 문서 로드(Load Documents)
    loader = PyMuPDFLoader(upload_file_path)
    docs = loader.load()
    print(f"문서의 페이지수: {len(docs)}")
    
    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)
    print(f"분할된 청크의수: {len(split_documents)}")
    print(f"문서의 페이지수: {len(split_documents)}")
    print(f"문서의 첫번째 페이지: {split_documents[0]}")
    
    local_model_path = "../embedding_model/multilingual-e5-large-instruct"

    # 허깅 페이스에서 다운로드 받아서 임베딩 진행
    embedding_model_name = "intfloat/multilingual-e5-large-instruct"
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device":"mps"},
        encode_kwargs={"normalize_embeddings":True},
    )
    
    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore_index = os.path.basename(upload_file_path).replace(".pdf", "-db").replace(" ", "_")
    if not os.path.exists(vectorstore_index) :    
        vectorstore = FAISS.from_documents(
            documents = split_documents,
            embedding = embeddings
        )
        vectorstore.save_local(
            folder_path = os.path.join("./cache/embeddings", vectorstore_index),
            index_name = vectorstore_index,
        )
    else :
        vectorstore = FAISS.load_local(
            folder_path = os.path.join("./cache/embeddings", vectorstore_index),
            index_name = vectorstore_index,
            embeddings = embeddings,                # 문서를 저장할 때 썼던 임베딩을 사용해야함
            allow_dangerous_deserialization = True, # 
        )
    
    # Chroma - s
    # 벡터 저장소 생성할 때는 대부분 from_documents를 사용
    
    # 1. DB 생성
    # 1-1 Embedding을 넣을 때, 리트리버와 동일한 임베딩을 써야함
        # Collection_name : 저장소의 이름, 문서마다 각자 사용하기 위해 파일의 이름으로 DB이름을 지정
        # FAISS가 더 나을 듯 ㅎㅎ..
    # collection_name_by_filename = os.path.basename(upload_file_path).replace(".pdf", "_collection").replace(" ", "_")
    # persist_directory = "./cache/embeddings"
    # collection_path = os.path.join(persist_directory, collection_name_by_filename)
    
    # if not os.path.exists(collection_path) :
    #     Chroma_db = Chroma.from_documents(
    #         documents=split_documents,
    #         embedding=embeddings,
    #         collection_name=collection_name_by_filename,    # 폴더의 개념, 안하면 임시로 생성
    #         persist_directory=persist_directory, # 임베딩을 저장할 디렉토리
    #     )
    # else :
    #     # 이미 같은 폴더명이 존재할 경우, 로드를 한다.
    #     Chroma_db = Chroma(
    #         persist_directory=persist_directory,
    #         collection_name=collection_name_by_filename,    # 해당 콜렉션을 가져온다.
    #         embedding=embeddings,
    #     )
    # # Chroma - e
    
    
    # 단계 5 : 검색기(Retriever) 생성
    # BM25Retriever
    bm25_retriever = BM25Retriever.from_documents(documents=split_documents)
    bm25_retriever.k = 5
    
    # faissRetriever
    faiss_retriever = vectorstore.as_retriever(
        search_type = "mmr",    # 검색 방법 (mmr, bm25)
        search_kwargs={"k": 10, "lambda_mult": 0.5, "fetch_k": 20}, 
    )
    
    # EnsembleRetriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.7, 0.3],
    )

    return ensemble_retriever

def get_llm():
    # llm = ChatOllama(
    #     model="EEVE-Korean-10.8B-Q5:latest",
    #     temperature=0.0,
    #     num_gpu=1,
    # )
    
    # llm = ChatOllama(
    #     model="llama3.1:8b",
    #     temperature=0.1,
    # )
    
    
    # llm = HuggingFacePipeline.from_model_id(
    #     model_id="sh2orc/Llama-3.1-Korean-8B-Instruct",
    #     task="text-generation",
    #     pipeline_kwargs={
    #         "max_new_tokens": 256,
    #         "top_k": 35,
    #         "temperature": 0.1,
    #     }
    # )
    
    llm = ChatOllama(
        # model="llama3.1-Korean-8B-Q8",
        model="llama-3.2-Korean-Bllossom-3B.Q8",
        temperature=0.1,
    )
    
    return llm

def get_promptTemplate(prompt_option_path, task=""):
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
    
    prompt = load_prompt(prompt_option_path, encoding="utf-8")
    if task : 
        prompt = prompt.partial(task=task)
    
    print(prompt)
    return prompt

def get_session_history(session_ids):
    print(f"[대화 세션ID]: {session_ids}")
    if session_ids not in st.session_state["store"]:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]  # 해당 세션 ID에 대한 세션 기록 반환


def create_chain(retriever):
    prompt = get_promptTemplate(prompt_option_box, task_input)
    llm = get_llm()
    
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

def create_rag_with_history(retriever):
    chain = create_chain(retriever)
    rag_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,  # 세션 기록을 가져오는 함수
        input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
        history_messages_key="chat_history",  # 기록 메시지의 키
    )
    return rag_with_history

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

if "chain" not in st.session_state:
    st.session_state["chain"] = None

# 캐시 디렉토리 생성
# 파일 업로드 때문에
if not os.path.exists("./cache"):
    os.makedirs("./cache")

if not os.path.exists("./cache/files"):
    os.makedirs("./cache/files")
    
if not os.path.exists("./cache/embeddings"):
    os.makedirs("./cache/embeddings")

## side bar
with st.sidebar:
    clear_btn = st.button("대화 초기화")
    
    ## 파일 업로드 기능 start
    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])
    ## 파일 업로드 기능 end
    
    ## 프롬프트 옵션 선택 start

    prompt_files = glob.glob("prompts/*.yaml")
    ### select box
    prompt_option_box = st.selectbox(
        "프롬프트 옵션 선택",
        prompt_files, index = 0
    )
    ## 프롬프트 옵션 선택 end
    task_input = st.text_input("Task 입력", "")

# 파일이 업로드 되었을 때
# 파일을 캐시 저장(시간이 오래걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="file uploading...") # 파일이 업로드가 되면 데코레이터로 캐싱을 해줌, 파일 관련할 때 cache_resource를 많이 사용
def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장함.
    file_content = file.read()
    file_path = f"./cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
        
    # 문서를 임베딩하기 위한 객체 생성
    retriever = get_retriever_for_file_upload(file_path)
    return retriever
    
    
# 파일을 업로드 하고 처리하는 것
if uploaded_file:
    # 파일 업로드 후 retriever 객체 생성
    retriever = embed_file(uploaded_file)
    chain = create_rag_with_history(retriever)
    st.session_state["chain"] = chain
    
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
        

# 처음 한번만 수행되게

if clear_btn : 
    # ALL CLEAR!
    st.session_state["message"] = []
    st.session_state["store"] = {}
    st.session_state["RAG_WITH_HISTORY"] = []
    st.session_state["chain"] = None

print_messages() # 이전 대화 출력

user_input = st.chat_input("Ask me something")

# 경고 메시지를 띄우기 위한 영역
warn_msg = st.empty()

if user_input: 
    chain = st.session_state["chain"]
    if chain is not None :
        st.chat_message("user").write(user_input)
        response = chain.stream(
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
    else :
        warn_msg.warning("파일을 업로드 해주세요.")
        
        
