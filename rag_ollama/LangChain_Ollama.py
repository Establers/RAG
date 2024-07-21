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
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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
logging.langsmith("LangChain_Ollama")

# 단계 1: 문서 로드(Load Documents)
loader = PyMuPDFLoader("docs/SPRi_AI_Brief_6월호_산업동향_최종.pdf")
docs = loader.load()
print(f"문서의 페이지수: {len(docs)}")

# 단계 2: 문서 분할(Split Documents)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)
print(f"분할된 청크의수: {len(split_documents)}")

# # 단계 3: 임베딩(Embedding) 생성
# # embeddings = OpenAIEmbeddings()

# # 허깅 페이스 로컬 임베딩 사용
# from langchain_community.embeddings import HuggingFaceHubEmbeddings
# embeddings = HuggingFaceHubEmbeddings()
# text = (
#     "임베딩 테스트를 하기 위한 샘플 문장입니다."  # 테스트용 문서 텍스트를 정의합니다.
# )
# query_result = embeddings.embed_query(text)
# len(query_result)
# query_result[:3]

###
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings # pip install -U langchain-huggingface
# Path to your local model
local_model_path = "h:/RAG/rag_ollama/embedding_model/multilingual-e5-large-instruct"
# Instantiate the HuggingFaceEmbeddings class
embeddings = HuggingFaceEmbeddings(model_name=local_model_path)

# text = (
#     "임베딩 테스트를 하기 위한 샘플 문장입니다."  # 테스트용 문서 텍스트를 정의합니다.
# )
# query_result = embeddings.embed_query(text)

# print(len(query_result))
# print(query_result[:3])

# 단계 4: DB 생성(Create DB) 및 저장
# 벡터스토어를 생성합니다.
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

# 단계 5: 검색기(Retriever) 생성
# 문서에 포함되어 있는 정보를 검색하고 생성합니다.
retriever = vectorstore.as_retriever()

# # 검색기에 쿼리를 날려 검색된 chunk 결과를 확인합니다.
# retriever.invoke("AI안전보안이사회에 대해서 알려줘")

###

# Ollama 모델을 불러옵니다.
# llm = ChatOllama(
#     model="EEVE-Korean-10.8B-Q5:latest",
#     temperature=0.2
# )

llm = ChatOllama(
    model="EEVE-Korean-10.8B-Q8:latest",
    temperature=0.0
)


# llm = ChatOllama(model="Llama-3-Open-Ko-8B-Q8:latest")
# from langchain.memory import ConversationBufferMemory
# ConvMemory = ConversationBufferMemory(return_messages=True)

chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question or ask for more information. "
                "If you don't know the answer, just say that you don't know or show me the context that might be relevant. "
                "If you think you have a guess that might be close to the answer, feel free to share it as well."
                "Answer in Korean and detail."
            ),
        ),
        # 대화 기록을 변수로 사용, history 가 MessageHistory 의 key 가 됨
        MessagesPlaceholder(variable_name="history"),
        (
            "human", 
            (
                "#Question:\n{question}\n"
                "#Context:\n{context}\n\n"
                "#Answer:"
            )
        ),  # 사용자 입력을 변수로 사용
    ]
)

runnable = chat_prompt | llm | StrOutputParser()

store = {}  # 세션 기록을 저장할 딕셔너리

# 세션 ID를 기반으로 세션 기록을 가져오는 함수
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

def get_session_history(session_ids: str) -> BaseChatMessageHistory :
    print(session_ids)
    if session_ids not in store:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환

from langchain_core.runnables.history import RunnableWithMessageHistory

with_message_history = (
    RunnableWithMessageHistory(  # RunnableWithMessageHistory 객체 생성
        runnable,  # 실행할 Runnable 객체
        get_session_history,  # 세션 기록을 가져오는 함수
        input_messages_key="qusetion",  # 입력 메시지의 키
        history_messages_key="history",  # 기록 메시지의 키
    )
)

#
# prompt = PromptTemplate.from_template(
#     """You are an assistant for question-answering tasks. 
#     Use the following pieces of retrieved context to answer the question or ask for more information. 
#     If you don't know the answer, just say that you don't know or show me the context that might be relevant.
#     If you think you have a guess that might be close to the answer, feel free to share it as well.
#     Answer in Korean and detail.

#     #Question: 
#     {question} 
#     #Context: 
#     {context} 

#     #Answer:"""
# )


# ChatPromptTemplate
    # 대화 목록을 프롬포트로 주입하고자 할 때 사용
    # 메시지는 튜플형식으로 구성하며, (role, message)로 구성하여 리스트로 생성
    # role
        # "system": 시스템 메시지
            # 주로 전역설정과 관련된 프롬포트 -> 보통 Ai의 역할 및 페르소나 지정, 임무 등 (대화 전체에 적용)
        # "human": 사용자 메시지
        # "ai": AI 메시지



# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "최대한 자세하게 답변하세요.",
#         ),
#         # 대화 기록을 변수로 사용, history 가 MessageHistory 의 key 가 됨
#         MessagesPlaceholder(variable_name="history"),
#         ("human", "{input}"),  # 사용자 입력을 변수로 사용
#     ]
# )

# MessagePlaceholder
    # 확정된 메시지가 아니지만 나중에 메시지 목록을 삽입하려는 경우에 사용
    # 대화 내용을 기록하고자 할 때 사용 즉, 위치만 잡아두는 것 + 키값으로
    


# 체인 생성
chain = (
    {
        "context": retriever, 
        "question": RunnablePassthrough(),
        # "history": 
        #     [
        #         ("human", "Foo"),
        #         ("ai", "Bar"),    
        #     ],
    }   # 프롬프트에 들어가는 변수에 해당하는 값 # chain.
    | chat_prompt
    | llm
    | StrOutputParser() # 답변을 할 때 사용하는 출력 파서, 최종 문자열로 나온다?
)

# from langchain_core.runnables.history import RunnableWithMessageHistory
# chain_with_history  = RunnableWithMessageHistory(
#     chain,
#     input_messages_key   = "question",
#     history_messages_key = "history",
# )

# 간결성을 위해 응답은 터미널에 출력됩니다.
answer = chain.stream("AI안전보안이사회에 대해서 알려줘")

# 스트리밍 출력
stream_response(answer)