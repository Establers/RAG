from dotenv import load_dotenv
import os

import sys
import io
# 표준 출력의 인코딩을 UTF-8로 설정
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

load_dotenv()

print(f"[API KEY]\n{os.environ['OPENAI_API_KEY']}")

# LangSmith 추적을 설정합니다. https://smith.langchain.com
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("CH12-Basic")

from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 객체 생성
# llm = ChatOpenAI(
#     temperature=0.1,  # 창의성 (0.0 ~ 2.0)
#     model_name="gpt-4o",  # 모델명
# )


# 단계 1: 문서 로드(Load Documents)
loader = PyMuPDFLoader("docs/REN_r01uh0495ej0100_rx634_MAH_20150225.pdf")
docs = loader.load()
print(f"문서의 페이지수: {len(docs)}")

# 단계 2: 문서 분할(Split Documents)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)
print(f"분할된 청크의수: {len(split_documents)}")

# 단계 3: 임베딩(Embedding) 생성
embeddings = OpenAIEmbeddings()

# 단계 4: DB 생성(Create DB) 및 저장
# 벡터스토어를 생성합니다.
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

# 단계 5: 검색기(Retriever) 생성
# 문서에 포함되어 있는 정보를 검색하고 생성합니다.
retriever = vectorstore.as_retriever()

# 검색기에 쿼리를 날려 검색된 chunk 결과를 확인합니다.
retriever.invoke("SCI 통신 레지스터를 설정하려고 하는데 주의해야할 점은 뭐야? 자세하게 알려줘")

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

# 단계 7: 언어모델(LLM) 생성
# 모델(LLM) 을 생성합니다.
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.1)

# 단계 8: 체인(Chain) 생성
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 체인 실행(Run Chain)
# 문서에 대한 질의를 입력하고, 답변을 출력합니다.
question = "SCI 통신 레지스터를 설정하려고 하는데 주의해야할 점은 뭐야? 자세하게 알려줘"
response = chain.invoke(question)
print(response)