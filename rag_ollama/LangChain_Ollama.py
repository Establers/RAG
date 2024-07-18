# !pip install langchain-teddynote
from langchain_teddynote import logging
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
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
loader = PyMuPDFLoader("docs/REN_r01uh0495ej0100_rx634_MAH_20150225.pdf")
docs = loader.load()
print(f"문서의 페이지수: {len(docs)}")

# 단계 2: 문서 분할(Split Documents)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)
print(f"분할된 청크의수: {len(split_documents)}")

# 단계 3: 임베딩(Embedding) 생성
# embeddings = OpenAIEmbeddings()

# 허깅 페이스 로컬 임베딩 사용
from langchain_community.embeddings import HuggingFaceHubEmbeddings
embeddings = HuggingFaceHubEmbeddings()
text = (
    "임베딩 테스트를 하기 위한 샘플 문장입니다."  # 테스트용 문서 텍스트를 정의합니다.
)
query_result = embeddings.embed_query(text)
len(query_result)
query_result[:3]

# 단계 4: DB 생성(Create DB) 및 저장
# 벡터스토어를 생성합니다.
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

# 단계 5: 검색기(Retriever) 생성
# 문서에 포함되어 있는 정보를 검색하고 생성합니다.
retriever = vectorstore.as_retriever()

# 검색기에 쿼리를 날려 검색된 chunk 결과를 확인합니다.
retriever.invoke("SCI 통신 레지스터를 설정하려고 하는데 주의해야할 점은 뭐야? 자세하게 알려줘")

# Ollama 모델을 불러옵니다.
llm = ChatOllama(model="EEVE-Korean-10.8B:latest")

prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question or ask for more information. 
    If you don't know the answer, just say that you don't know or show me the context that might be relevant.
    If you think you have a guess that might be close to the answer, feel free to share it as well.
    Answer in Korean and detail.

    #Question: 
    {question} 
    #Context: 
    {context} 

    #Answer:"""
)

# 체인 생성
chain = (
    {"context": retriever, "question": RunnablePassthrough()}   # 프롬프트에 들어가는 변수에 해당하는 값
    | prompt
    | llm
    | StrOutputParser() # 답변을 할 때 사용하는 출력 파서, 최종 문자열로 나온다?
)

# 간결성을 위해 응답은 터미널에 출력됩니다.
answer = chain.stream("SCI 통신 레지스터를 설정하려고 하는데 주의해야할 점은 뭐야? 자세하게 알려줘")

# 스트리밍 출력
stream_response(answer)