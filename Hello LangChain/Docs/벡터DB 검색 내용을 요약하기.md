# 벡터DB 검색 내용을 요약하기

벡터 데이터 베이스에서 관련된 문서를 찾아온 후에, 이 문서의 내용을 프롬프트에 컨텍스트로 삽입하여 LLM에 전달해야 한다. 그런데 LLM은 입력 사이즈에 대한 한계가 있기 때문에, 검색해온 문서의 크기가 클 경우에는 입력사이즈 제한에 걸려서 프롬프트에 삽입하지 못할 수 있다

프롬프트에 넣을 수 있는 사이즈라 하더라도, 원본 문서는 질문에 대한 답변을 줄 수 있는 정보뿐만 아니라 관련없는 텍스트가 많이 포함되어 있을 수 있다. 이런 문제를 해결하는 방법중의 하나는 LLM을 이용하여 검색된 문서를 질의와 상관있는 정보 위주로 요약해서 프롬프트에 삽입하면 된다.

이런 일련의 작업을 Langchain에서는 LLM을 이용한 Contextual Compression Retriever라는 기능으로 제공한다. 벡터 데이터 베이스에서 검색해온 문서를 LLM을 이용하여 자동 요약하여 리턴해준다.

사용법은 의외로 간단하다.

```python
import pinecone
import openai
import logging
import os
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone

embedding = OpenAIEmbeddings()
llm = OpenAI()

index = pc.Index("ethan-wiki")
text_field = "text"
pinecone_api_key = os.getenv("PINECONE_API_KEY")

vectordb = Pinecone(
    index=index,
    embedding=embedding,
    text_key=text_field
)

compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(base_compressor = compressor,base_retriever = vectordb.as_retriever())

query = "Where is the best locaction for vacation?"
docs = compression_retriever.get_relevant_documents(query,k=5)
for doc in docs:
    print(doc)
    print("\n")
```

앞의 예제와 마찬가지로, pinecone 데이터 베이스 연결을 생성하고, embedding API로 사용할 OpenAIEmbedding 객체를 생성한다. 추가로, 검색된 텍스트에 대한 요약을 LLM을 사용하기 때문에 OpenAI LLM 객체를 생성하여  llm에 저장하였다. 


위키피디아는 수백줄에 해당하는데 이 결과는 수줄 내로 내용을 요약한 것. 
```text
page_content='- The Cathars believed that the world had been made by a bad god.  \n- They believed this cycle of coming back to life could be escaped by a ritual cleansing.  \n- Women were prominent in the faith.  \n- They were pacifists.  \n- They didn\'t eat anything that was made from other animals, including meat and cows milk.  \n- They preached tolerance of other faiths.  \n- They rejected the usual Christian rules of marriage and only believed in the New Testament.  \n- In the South of France, the Languedoc nobles protected it, and many noble women became "Perfects".  \n- Parish clergy had low morale, or confidence.  \n- The Catholic Church was against Catharism, seeing it as a heresy.  \n- The Albigensian Crusade  \n- The Pope ordered a crusade against the Cathars in southern France.  \n- Arnauld Amaury made the famous quote "Kill them all, god knows his own"  on being asked how to tell who were Cathars during the assault.  \n- It is interesting to note that at the siege of Montsegur when the fires were lit the Cathars ran down the hill and threw themselves on, as their beliefs were very' metadata={'chunk': 13.0, 'source': 'https://simple.wikipedia.org/wiki/Catharism', 'title': 'Catharism', 'wiki-id': '135'}


page_content='Tokyo, Japan - 37+ million\n Mexico City, Mexico - 21 million\n Mumbai, India - 20 million\n São Paulo, Brazil - 18 million\n Lagos, Nigeria - 13 million\n Calcutta, India - 13 million\n Buenos Aires, Argentina - 12 million\n Seoul, South Korea - 12 million\n Beijing, China - 12 million\n Karachi, Pakistan - 12 million\n Dhaka, Bangladesh - 11 million\n Manila, Philippines - 11 million\n Cairo, Egypt - 11 million\n Osaka, Japan - 11 million\n Rio de Janeiro, Brazil - 11 million\n Tianjin, China - 10 million\n Moscow, Russia - 10 million\n Lahore, Pakistan - 10 million' metadata={'chunk': 15.0, 'source': 'https://simple.wikipedia.org/wiki/City', 'title': 'City', 'wiki-id': '144'}


page_content='A continent is a large area of the land on Earth that is joined together.\n\nThere are no strict rules for what land is considered a continent, but in general it is agreed there are six or seven continents in the world, including Africa, Antarctica, Asia, Europe, North America, Oceania(or Australasia), and South America.' metadata={'chunk': 4.0, 'source': 'https://simple.wikipedia.org/wiki/Continent', 'title': 'Continent', 'wiki-id': '117'}
```


## LLMFilter

Retriever를 이용해서 검색한 내용은 전체문서를 chunk 단위로 나눈 텍스트에 대한 임베딩 데이터로 검색하였기 때문에, 전체 문서를 대표하는 사실 어렵다. 예를 들어 휴가지에 대한 질의에 대한 검색 결과로 어떤 문서가 리턴되었을때, 그 문서에 휴가지에 대한 내용이 한줄이고 나머지 99줄이 다른 내용이라 하더라도 휴가지에 대한 한줄 문서의 임베딩 값에 의해서 그 문서가 검색될 수 있다.

그래서 검색된 문서가 실제로 질의와 많은 연관성이 있는지 다시 확인해서 연관성이 낮다면 그 문서를 제거하고 다른 문서를 사용하는 것이 더 좋은 결과를 얻을 수 있는데, 이러한 기능을 지원하는 것이 LLMFilter이다. LLMFilter는 ContextualCompressionRetriever와 같이 사용될 수 있으며, 검색된 결과문서가 질의와 연관성이 얼마나 높은지를 LLM을 이용하여 판단하고, 연관성이 높지 않다면 그 문서를 제거하는 기능을 한다

사용법은 매우 간단하다. ContextualCompressionRetriever 부분에서 LLMExtract대신 LLMChainFilter를 사용하도록하면된다.

```python
from langchain.retrievers.document_compressors import LLMChainFilter

filter = LLMChainFilter.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
base_compressor = filter,base_retriever = vectordb.as_retriever(),k=2)

query = "Where is the best locaction for vacation?"
docs = compression_retriever.get_relevant_documents(query)
for doc in docs:
    print(doc)
    print("\n")
```

LLMChainFilter를 사용하면 LLM을 이용하여 의미상 질문과 관련 없는 정보를 걸러낼 수 있지만, LLM 모델을 호출해야 하기 때문에, 속도가 느리고 추가적인 비용이 든다. 이를 보완하는 방법으로 LLMChainFiltere 대신에 EmbeddingsFilter를 사용하는 방법이 있다.

이 방식은 검색된 문서와 질의의 임베딩 벡터간의 유사도를 측정하여 검색된 문서가 얼마나 연관성이 있는지를 판단한다.

데이터베이스에서 질문을 임베딩 벡터로 이미 검색했기 때문에 같은 내용이라고 착각할 수 도 있지만, 벡터데이터베이스에서의 검색은 질문과 임베딩된 문장간의 검색이고, EmbeddingFilter는 검색된 질문과 검색된 문서간의 유사도 비교이기 때문에 다르다고 볼 수 있다.


## Filter와 Extractor를 함께 사용하기

앞서 Contextual Compressor에서 LLMChainExtractor,LLMChainFilter 등을 살펴봤는데, 이를 같이 사용할 수는 없을까? 예를 들어 관련 없는 문서들을 LLMChainFilter를 통해서 제거하고, 관련된 문서들만 LLM을 통해서 요약하는 유스케이스를 구현할 수 없을까?

ContextualCompressorRetriever는 Extractor 부분에 여러 필터를 파이프라인식으로 연결함으로써 이 기능들을 같이 적용할 수 있다.

아래 코드는 파이프라인을 적용한 코드로 먼저 EmbeddingsRedundantFilter를 적용하였다.

앞에서는 소개하지 않은 필터인데, 벡터 데이터베이스를 검색하면 많은 비율로 같은 문서가 검색결과로 나오는 경우가 있다. 이유는 하나의 문서가 여러개의 chunk로 분할되어 벡터 데이터베이스에 저장되기 때문에, 검색이 chunk 단위로 이루어지게 되고 그래서 한문서에 유사한 내용이 있는 chunk 가 많기 때문에 같은 문서가 검색되게 된다.

이는 참조 정보에 대한 다양성을 저해할 수 있기 때문에, top-k값을 늘려서 검색 결과의 수를 늘리고, 그중에서 중복된 문서를 제거하면 다양한 검색 결과를 얻을 수 있다.

중복을 제거한 후에는 위에서 살펴보았던 Extractor를 통해서 검색 결과를 요약하고, LLM Filter를 이용하여 관계 없는 문서를 제거하도록 하였다.

```python
from langchain.llms import OpenAI
import pinecone
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter

embedding = OpenAIEmbeddings()
llm = OpenAI()

index = pc.Index("ethan-wiki")
text_field = "text"
pinecone_api_key = os.getenv("PINECONE_API_KEY")

vectordb = Pinecone(
    index=index,
    embedding=embedding,
    text_key=text_field
)


llm_filter = LLMChainFilter.from_llm(llm)
llm_extractor = LLMChainExtractor.from_llm(llm)
redundant_filter = EmbeddingsRedundantFilter(embeddings=embedding)

pipeline_compressor = DocumentCompressorPipeline(transformers=[redundant_filter,llm_extractor,llm_filter])

compression_retriever = ContextualCompressionRetriever(base_compressor = pipeline_compressor,base_retriever = vectordb.as_retriever(),k=10)


#query = "Where is the best place for summer vacation?"
query ="Where is the cuba? and nearest country by the Cuba?"
docs = compression_retriever.get_relevant_documents(query)
for doc in docs:
    print(doc)
    print("\n")
```

References: 
- https://bcho.tistory.com/1417