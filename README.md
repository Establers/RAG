## llama3 
https://ollama.com/library/llama3

## LM Studio
https://lmstudio.ai/

## Dependency (LangChain)
```
pip install -r https://raw.githubusercontent.com/teddylee777/langchain-kr/main/requirements.txt
```
> 3.11 <= python --version < 3.12

## LangChine(LangSmith)
https://docs.smith.langchain.com/
https://smith.langchain.com/

## reference
https://wikidocs.net/book/14314
https://www.youtube.com/watch?v=1scMJH93v0M&t=71s

--- 

## Todos
- SRS 상에 있는 이미지는 어떻게 처리?
  - 1. 해당 라인 근처 사진을 가져와서 ollama 멀티 모달로 학습한 내용을 같이 프롬포트에?

- 메모리 기능
  - 어떤 메모리 기능을 사용할 지?.. 
    - LLM으로 요약? -> 시간이 많이 걸릴지도.. + 정보 손실
    - 입력은 요약하고, 출력은 있는 그대로 저장?
    - 저장을 한다면 얼만큼?... Token? 개수? 아니면 무제한?
  - DB에 저장?
    - 레디스

### else
- 문서를 load 하는 기능
- LangServe, Streamlit

### Deployment
- 배포는 Docker?
- 서버를 내 컴에 두면 너무 랙 걸릴 듯...
- 각자 챗봇 형식으로 각자의 로컬에서 할 수 있도록 하는게 베스트일 듯

### sourcetree test...