# !pip install langchain-teddynote
from langchain_teddynote import logging
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_teddynote.messages import stream_response

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import (
    RunnableLambda,
    ConfigurableFieldSpec,
    RunnablePassthrough,
)
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from operator import itemgetter
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings # pip install -U langchain-huggingface
#### 표준 출력의 인코딩을 UTF-8로 설정
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')
#### 

from dotenv import load_dotenv
import os
load_dotenv()
# 프로젝트 이름을 입력합니다.
logging.langsmith("LangChain_InMemory")

def get_retriever(): 
    # 단계 1: 문서 로드(Load Documents)
    # loader = PyMuPDFLoader("../docs/SPRi_AI_Brief_6월호_산업동향_최종.pdf")
    loader = PyMuPDFLoader("../../docs/SPRi_AI_Brief_6월호_산업동향_최종.pdf")
    docs = loader.load()
    print(f"문서의 페이지수: {len(docs)}")

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)
    print(f"분할된 청크의수: {len(split_documents)}")

    # from langchain_community.embeddings import HuggingFaceEmbeddings
    
    # Path to your local model
    local_model_path = "h:/RAG/rag_ollama/embedding_model/multilingual-e5-large-instruct"
    # Instantiate the HuggingFaceEmbeddings class
    embeddings = HuggingFaceEmbeddings(model_name=local_model_path)

    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever()
    
    return retriever

def get_llm():
    llm = ChatOllama(
        model="EEVE-Korean-10.8B-Q5:latest",
        temperature=0.0
    )
    return llm

def get_promptTemplate():
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
    
    return prompt

store = {}  # 세션 기록을 저장할 딕셔너리

# 세션 ID를 기반으로 세션 기록을 가져오는 함수

def get_session_history(session_ids):
    print(f"[대화 세션ID]: {session_ids}")
    if session_ids not in store:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환

print(store)

def create_chain():
    prompt = get_promptTemplate()
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



chain = create_chain()
# 대화를 기록하는 RAG 체인 생성
rag_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,  # 세션 기록을 가져오는 함수
    input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
    history_messages_key="chat_history",  # 기록 메시지의 키
)


result1 = rag_with_history.invoke(
    # 질문 입력
    {"question": "AI안전보안이사회에 대해서 알려줘"},
    # 세션 ID 기준으로 대화를 기록합니다.
    config={"configurable": {"session_id": "rag123"}},
)

print(result1)

result2 = rag_with_history.invoke(
    # 질문 입력
    {"question": "이전 답변을 짧게 더 요약해줘"},
    # 세션 ID 기준으로 대화를 기록합니다.
    config={"configurable": {"session_id": "rag123"}},
)