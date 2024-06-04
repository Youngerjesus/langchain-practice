# 벡터 DB 검색하기

앞의 포스트에서 pinecone 벡터데이터베이스에 임베딩된 chunk를 저장하였으면, 이제 이 chunk를 검색하는 방법을 살펴보자.

아래 예제는 langchain을 이용하지 않고, pinecone의 search API를 직접 사용해서 검색하는 방법이다.

```python

import pinecone
import openai
import os
from langchain.embeddings.openai import OpenAIEmbeddings

pinecone_api_key = os.getenv("PINECONE_API_KEY")
#Connect database
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

#pinecone.create_index("terry-wiki",dimension=1536,metric="cosine")
vectordb = pc.Index("ethan-wiki")

embedding = OpenAIEmbeddings()
question = ["Where is the Cuba?"]

embedded_question = embedding.embed_documents(question)


query_result=vectordb.query(
    vector=embedded_question,
    top_k=3,
    include_values=False,
    include_metadata=True
)

result_ids = [ result.id for result in query_result.matches]

for result in query_result.matches:
    id = result.id
    text = result.metadata['text'].replace('\n','')[:500]
    title = result.metadata['title']
    score = result.score
    print(id,score,title)
    print(text,"....")
    print('\n')
```

“Where is the Cuba?”가 질의 내용이고 이와 유사한 문서를 검색하도록 한다.

이 텍스트를 검색하려면 먼저 텍스트를 임베딩 벡터로 변환해야 한다. “embedding.embed_documents(question)” 를 이용하여 질문을 임베딩 한후에, index.query를 이용해서 검색을 한다. 

이때 메타 데이터를 같이 리턴하도록 input_metadata 플래그를 True로 해주고, 검색 결과는 3개의 유사한 문장을 찾도록 하였다.

다음 검색 결과를 출력하였다. 다음은 실행 결과이다.

```text
5f46e8ec-d0f1-4b00-8774-8f5fc1fde4c4 0.878542185 Cuba
Cuba is an island country in the Caribbean Sea. The country is made up of the big island of Cuba, the Isla de la Juventud island (Isle of Youth), and many smaller islands. Havana is the capital of Cuba. It is the largest city. The second largest city is Santiago de Cuba. In Spanish, the capital is called "La Habana". Cuba is near the United States, Mexico, Haiti, Jamaica and the Bahamas. People from Cuba are called Cubans (cubanos in Spanish). The official language is Spanish. Cuba is warm all y ....


82ce5a99-c772-4bfa-8281-34249dfd9cfc 0.859715343 Cuba
Cuba is an island country in the Caribbean Sea. The country is made up of the big island of Cuba, the Isla de la Juventud island (Isle of Youth), and many smaller islands. Havana is the capital of Cuba. It is the largest city. The second largest city is Santiago de Cuba. In Spanish, the capital is called "La Habana". Cuba is near the United States, Mexico, Haiti, Jamaica and the Bahamas. People from Cuba are called Cubans (cubanos in Spanish). The official language is Spanish. Cuba is warm all y ....


418545a7-f358-4667-b941-9ecd904eab5d 0.840268135 Cuba
Cuba is an island country in the Caribbean Sea. The country is made up of the big island of Cuba, the Isla de la Juventud island (Isle of Youth), and many smaller islands. Havana is the capital of Cuba. It is the largest city. The second largest city is Santiago de Cuba. In Spanish, the capital is called "La Habana". Cuba is near the United States, Mexico, Haiti, Jamaica and the Bahamas. People from Cuba are called Cubans (cubanos in Spanish). The official language is Spanish. Cuba is warm all y ....
```

3개의 chunk가 연관되어 검색되었고, 3 문서 모두 Cuba에 대한 같은 글을 포인팅하고 있다. 이때 주목할만한점은 Score 필드인데, 이 필드는 검색된 결과가 얼마나 유사도가 높은지를 나타낸다.

Langchain에서 이렇게 벡터 데이터베이스를 검색하는 방식을 추상화해놓고, 유사도 검색뿐만 아니라 다양한 방식의 검색 방식을 지원하며 이 기능을 Retriever라고 한다. 

## Similarity (유사도 기반 검색)

앞에서 사용한 유사도 기반의 검색을 langchain asmilarity 기능을 이용하여 검색해보면 다음과 같다. 

단순하게 similarity_search 를 이용하면 되고, pinecone native api를 사용하는 것과 다르게, 별도로 임베딩하지 않고 일반 Text를 입력해도, 자동으로 임베딩하여 검색을 수행한다.


## MultiQuery (멀티 쿼리)

만약에 질문에 답변하기 위해서 필요한 정보가 하나가 아니라 여러개라면? 질문을 하나의 벡터로 바꿔서 검색하게 되면 전체 질문과 유사한 정보만 가지고 오지 개개별 필요 정보에 대한 정보는 가지고 오지 않는다.


예를 들어 “쿠바가 있는 위치는? 그리고 쿠바와 가장 가까운 나라는? 이 질문에 답하기 위해서는 몇가지 정보가 필요하다.
- 쿠바의 지리적 위치
- 쿠바가 위치한 곳의 인접국가
- 인접국가 중에서 쿠바와 가장 가까운 나라


이렇게 하나의 질문에 여러개의 정보가 필요한 경우 사용할 수 있는 Retriever가 langchain을 MultiQuery Retriever이다.

MultiQuery Retriever는 LLM (ChatGPT와 같은)을 이용하여 먼저 질문을 분석하여, 하나의 질문을 여러개의 질문으로 쪼게낸 후에, 이 분리된 질문을 기반으로 임베딩을 검색한다.


아래는 “쿠바가 있는 위치와 가장 가까운 나라"를 질의 하는 쿼리를 MultiQuery Retriever를 이용하여 구현한 예제이다.


```python
import openai
import logging
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.vectorstores import Pinecone  

load_dotenv(override=True)

embedding = OpenAIEmbeddings()
llm = ChatOpenAI()

#Connect database
index = pc.Index("ethan-wiki")
text_field = "text"
pinecone_api_key = os.getenv("PINECONE_API_KEY")

vectordb = Pinecone(
    index=index,
    embedding=embedding,
    text_key=text_field
)

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

query = "Where is the cuba? Where is other country near the cuba?"

retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(), llm=llm
)

docs = retriever_from_llm.get_relevant_documents(query=query)
for doc in docs:
    print(doc.metadata)
```

MultiQueryRetriever가 어떻게 질문을 이해해서 작동하는지를 알아보기 위해서 multi_query에 대한 logger를 INFO 레벨로 조정하였다.

실행하면 아래와 같은 디버깅 메시지를 확인할 수 있다. 질문에 대한 답을 하기 위해서 아래와 같이 3개의 질문으로 분리된것을 볼 수 있다. 쿠바의 위치, 쿠바와 인접한 국가 그리고 쿠바와 가까운 국가.

```text
INFO:langchain.retrievers.multi_query:Generated queries: ['1. What is the location of Cuba?', '2. Can you provide information on countries that are geographically close to Cuba?', '3. Which neighboring countries are near Cuba?']
```




