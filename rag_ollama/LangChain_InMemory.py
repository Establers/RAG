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
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import Runnable
from operator import itemgetter
from langchain_core.runnables import (
    RunnableLambda,
    ConfigurableFieldSpec,
    RunnablePassthrough,
)
from langchain_core.messages import BaseMessage, AIMessage
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

# 단계 1: 문서 로드(Load Documents)
loader = PyMuPDFLoader("docs/SPRi_AI_Brief_6월호_산업동향_최종.pdf")
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
retriever.invoke("AI안전보안이사회에 대해서 알려줘")

llm = ChatOllama(
    model="EEVE-Korean-10.8B-Q5:latest",
    temperature=0.0
)

store = {}  # 세션 기록을 저장할 딕셔너리
def get_session_history(user_id: str, conversation_id: str):
    print("user_id: ", user_id, "conversation_id: ", conversation_id)
    if (user_id, conversation_id) not in store:
        store[(user_id, conversation_id)] = ChatMessageHistory()
    return store[(user_id, conversation_id)]


history = get_session_history("1", "1")
history.add_message(AIMessage(content="hello"))
print(store)

# ChatPromptTemplate
    # 대화 목록을 프롬포트로 주입하고자 할 때 사용
    # 메시지는 튜플형식으로 구성하며, (role, message)로 구성하여 리스트로 생성
    # role
        # "system": 시스템 메시지
            # 주로 전역설정과 관련된 프롬포트 -> 보통 Ai의 역할 및 페르소나 지정, 임무 등 (대화 전체에 적용)
        # "human": 사용자 메시지
        # "ai": AI 메시지

chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question or ask for more information. "
                "If you don't know the answer, just say that you don't know or show me the context that might be relevant. "
                "If you think you have a guess that might be close to the answer, feel free to share it as well."
                "Answer in Korean and detail.\n"
                
                "#Retrieved Context:\n{context}\n\n"
            ),
        ),
        # 대화 기록을 변수로 사용, history 가 MessageHistory 의 key 가 됨
        # MessagePlaceholder
            # 확정된 메시지가 아니지만 나중에 메시지 목록을 삽입하려는 경우에 사용
            # 대화 내용을 기록하고자 할 때 사용 즉, 위치만 잡아두는 것 + 키값으로
        MessagesPlaceholder(variable_name="history"),
        (
            "human", 
            (
                "#Question:\n{question}\n"
                "#Answer:"
            )
        ),  # 사용자 입력을 변수로 사용
    ]
)

context = itemgetter("question") | retriever
first_step = RunnablePassthrough.assign(context=context)

runnable: Runnable = itemgetter("question") | retriever | chat_prompt | llm | StrOutputParser()



# 세션 ID를 기반으로 세션 기록을 가져오는 함수

with_message_history = RunnableWithMessageHistory (  # RunnableWithMessageHistory 객체 생성
    runnable,  # 실행할 Runnable 객체
    get_session_history,  # 세션 기록을 가져오는 함수
    input_messages_key="question",  # 입력 메시지의 키
    history_messages_key="history",  # 기록 메시지의 키
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="Unique identifier for the user.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="Unique identifier for the conversation.",
            default="",
            is_shared=True,
        ),
    ],
)

print(first_step.invoke({"question": "hello", "history": "boom"}))
result = with_message_history.invoke(
    {"question" : "AI안전보안이사회에 대해서 알려줘"},
    config = {"configurable" : {"user_id" : "test2", "conversation_id" : "2"}}
)

print(result)