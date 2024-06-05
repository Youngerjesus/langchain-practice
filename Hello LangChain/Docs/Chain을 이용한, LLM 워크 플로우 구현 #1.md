# Chain을 이용한, LLM 워크 플로우 구현 #1

LLM 기반 애플리케이션을 개발할때, 한번의 LLM 호출로 결과를 낼 수 도 있지만, 복잡한 LLM 애플리케이션의 경우, LLM의 출력을 다시 다음 LLM의 입력으로 넣어서 LLM을 여러개를 연결해서 답변을 낼 수 도 있고, 입력 프롬프트에 따라서 알맞은 LLM이나 프롬프트를 선택하도록 분기 할 수 있다.

예를 들어 Python Coding을 해주는 LLM에서 API 파이썬 코드를 생성한 후에, 이 코드에 맞는 Unit Test 코드를 생성하는 LLM을 호출하거나,

![](../images/순차%20llm%20chain.png)

아래 그림과 같이 학교 학생의 공부를 도와주는 챗봇에서 질문의 종류에 따라서, 영어,과학,수학 LLM을 선택적으로 호출하는 구조를 예로 들 수 있다.

![](../images/분기%20llm%20chain.png)


이렇게 여러개의 LLM을 연결하여 LLM 애플리케이션을 개발할 수 있는 기능이 Langchain에서 Chain이라는 컴포넌트이다. 물론 직접 chatgpt api등을 사용해서 이런 복잡한 흐름을 개발할 수 있지만 Langchain의 chain은 이를 쉽게 개발 할 수 있도록 추상화된 계층을 제공한다.

이번 장에서는 Chain에 대해서 알아보기로 한다.

## LLMChain

먼저 LLMChain의 개념을 이해해야 하는데, LLM Chain은 프롬프트 템플릿을 LLM을 합쳐서 컴포넌트화 한 것이다.

즉 입력값으로 문자열을 넣으면 프롬프트 템플릿에 의해서 프롬프트가 자동으로 완성되고, LLM 모델을 호출하여 텍스트 출력을 내주는 기능을 한다. 

아래 예제를 보자.

이 예제는 도시 이름을 입력하면 출력으로 그 도시의 유명한 관광지를 리턴하는 LLM Chain을 구현한 예이다.

```python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI


llm = ChatOpenAI()

prompt = PromptTemplate.from_template("what is the famous tour place in {city}?")
chain = LLMChain(llm=llm, prompt=prompt)

city = "Seoul"
chain.invoke(city)
```

chain = LLMChain(llm=model,prompt)에서, LLMChain을 생성하였다. 이때, 사용할 프롬프트 템플릿과 LLM 모델을 지정하였다. 이 LLMChain을 호출할때는 chain.run(입력변수)를 이용해서 호출할 수 있다. 아래 출력결과는 “seoul”을 입력값으로 넣었을때 결과이다.

```json
{'city': 'Seoul',
 'text': "One of the most famous tour places in Seoul is Gyeongbokgung Palace. This grand palace complex is a symbol of Korea's royal heritage and is a popular destination for tourists seeking to explore the country's history and culture. Other popular tour places in Seoul include Bukchon Hanok Village, N Seoul Tower, Myeongdong shopping district, and Changdeokgung Palace."}
```

## Sequential Chain

LLMChain 컴포넌트를 만들었으면, 이제 LLMChain들을 서로 연결하는 방법에 대해서 알아보자.

아래 예제는 먼저 도시 이름 {city}을 입력 받은 후에, 첫번째 chain에서 그 도시의 유명한 관광지 이름을 {place}로 출력하도록 한다. 

다음 두번째 chain에서는 관광지 이름 {place}를 앞의 chain에서 입력 받고, 추가적으로 교통편 {transport}를 입력받아서, 그 관광지까지 가기 위한 교통편 정보를 최종 출력으로 제공한다.

이렇게 여러 Chain을 순차적으로 연결하게 해주는 컴포넌트가 SequentialChain이다. 아래 코드를 살펴보자.

```python
from langchain.chains import SequentialChain

model = ChatOpenAI()

prompt1 = PromptTemplate.from_template("what is the famous tour place in {city}? Tell me the name of the place only without additional comments.")
prompt2 = PromptTemplate.from_template("How can I get {place} by {transport}?")
chain1 = LLMChain(llm=model,prompt=prompt1,output_key="place",verbose=True)
chain2 = LLMChain(llm=model,prompt=prompt2,verbose=True)

chain = SequentialChain(chains=[chain1,chain2],input_variables=["city","transport"],verbose=True)

chain.invoke({'city':'Seoul','transport':'subway'})
```

출력 결과: 

```json
{'city': 'Seoul',
 'transport': 'subway',
 'text': 'You can get to Gyeongbokgung Palace by taking the subway to Gyeongbokgung Station (Line 3). From there, take exit 5 and walk straight for about 5-10 minutes until you reach the entrance of the palace.'}ı
```








References: 
- https://bcho.tistory.com/1420