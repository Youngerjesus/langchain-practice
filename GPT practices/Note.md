
#### UnstructuredFileLoader: 
- 주로 비구조화된 파일에서 데이터를 읽어오기 위해 사용되는 클래스임. 
- 클래스는 텍스트 파일, PDF, HTML 등 다양한 형태의 비구조화된 파일을 처리할 때 사용함. 


#### Q) UnstructuredFileLoader 를 사용할 때는 특정 파일의 타입을 모르는 경우에 사용하겠네? 

맞다. 


RecursiveCharacterTextSplitter: 
- 문장이나 문단의 끝을 기준으로 잘러준다. 문장이나 문단이 중간에 끊기는 경우를 막아줌.

#### Embedding: 
- Openai 에서 Embedding 은 1000개의 벡터 차원으로 나눠서 표현한다고 한다. 
- 이 차원이라는게 무슨 말인지 잘 이해가 안될거임. 예시로 나타내서 3개의 차원만 있다고 가정해보자. (남성스러움, 여성스러움, 왕족스러움) 이렇게 차원이 있다고 가정했을 때 King 이라는 단엉는 남성스러움과 왕족스러움 차원을 가질거임. 이런식으로 표현을 하는거다. 이런식으로 표현하면 각 단어마다 유사한 의미를 가졌지 표현할 수 있을거임. 두 단어의 벡터 차가 아주 작다면 두 단어는 비슷한 의미를 가지는거니까. 

### CacheBackedEmbeddings
- 텍스트를 임베딩한 결과를 캐싱할 수 있는 기능을 제공해줌.
- 오해할 수 있는데 임베딩한 벡터 값은 Vector Store 에 저장한다.
- CacheBackedEmbeddings 는 문서를 여러번 임베딩하지 않도록 만드는 것. (문서를 임베딩할 때 OpenAI 의 LLM 을 사용하게 되는데 비용이 발생하니까.)


#### CacheBackedEmbeddings 사용 가이드:
```python
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores import Chroma


cache_dir = LocalFileStore("./.cache/")

embedder = OpenAIEmbeddings()

cached_embedding = CacheBackedEmbeddings(embedder, cache_dir)

vectorstore = Chroma.from_documents(docs, cached_embedding)
```

#### langhsmith: 
- 이 설정을 해두면 langchian 내부에서 일어나는 작업에 대한 디버깅을 할 수 있음. 


#### Lagnchain 에서는 파이프 연산자를 이용해서 chain 을 구성할 수 있다: 
- prompt 로 들어갈 땐 `prompt.format_messages(context=context, question=question)` 이런식으로 호출됨. 


```python
from langchain.chains import RetrievalQA 
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

llm = ChatOpenAI(
    temperature=0.1
)

retriever = vectorstore.as_retriever()

chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever= retriever
)


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assitant. answer questions uisng only the following context. If you don't know the answer just say you don't know don't make it up: \n\n{context}"),
    ("human", "{question}")
])
 
 
chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm 

chain.invoke("카프카에서 브로커를 신뢰성있게 쓰려면 어떻게 해야하는가?")

```

LangChain 에서 Chain 을 이용하는 이유:
- LLM 에게 질문하고 답변을 가져오는 과정은 템플릿 구조를 따른다: 
  - 1. Vector Store 에서 Document 가져오고
  - 2. 가져온 Document 를 가지고 Context 만들어주고
  - 3. 만든 Context 와 질문 그리고 프롬포트 템플릿을 가지고 LLM 에게 던져줄 프롬포트를 작성해주고
  - 4. LLM 에게 프롬포트를 던져서 답변을 가져오는 것. 

```python
docs = retriever.invoke(message)
docs = "\n\n".join(document.page_content for document in docs)

prompt = template.format_messages(context=docs, question=message)

llm.predit_messages(prompt)
```

LangChain 의 Callback Handler 를 이용하면 LLM 이 에러를 던졌을 때, 시작했을 때 등의 이벤트를 받아낼 수 있다.  

