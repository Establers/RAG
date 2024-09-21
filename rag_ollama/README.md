## Model Download
https://huggingface.co/teddylee777/EEVE-Korean-Instruct-10.8B-v1.0-gguf/tree/main
- 집에서 `EEVE-Korean-Instruct-10.8B-v1.0-Q5_K_M.gguf` 로 모델 다운 받았음.

### Modelfile 설정
Modelfile 해당 내용 파일 생성 후, 복사 붙여넣기 

### ollama create
```
ollama create EEVE-Korean-10.8B -f Modelfile
// Modelfile 이 있는 폴더에서 수행
```

### ollama list
```
ollama list
```
```
NAME                    ID              SIZE    MODIFIED      
EEVE-Korean-10.8B:latest        4d957747ec33    7.7 GB  5 seconds ago  
```
- 모델이 추가 된 것을 확인할 수 있다.

### HuggingFace Open Embedding model Download
```
MAC : `brew install git-lfs`
git lfs clone https://huggingface.co/intfloat/multilingual-e5-large-instruct
```


