# LCEL (LangChain Expression Language)

앞에서 소개한 Chain 은 개념적으로는 훌륭하지만, 코드 양이 다소 많아지고, 병렬처리나 비동기 처리, 스트리밍 같은 고급 기능을 구현하기 어렵다.

이런 한계를 극복하기 위해서 2023년 8월에 LangChain Expression Language (이하 LCEL이 개발되었다.)

Chain의 기능을 대처하는 컴포넌트로, 병렬,비동기,스트리밍 같은 고급 워크플로우 처리에서 부터 FallBack이나 Retry 와 같은 장애 처리 기능을 지원하며, 추후에 소개할 Langchain 모니터링/평가 솔루션인 LangSmith와 쉽게 연동이 된다.

이번장에서는 앞에서 구현한 LLMChain, Sequential Chain, Advanced Sequential Chain 그리고 Router Chain을 이 LCEL로 구현하여 LCEL에 대해서 알아보고 기존 Chain과의 차이점을 이해한다.

2024년 1월 현재, 앞에서 소개한 Chain은 아직 그대로 지원이 되고 있다. LCEL은 소개가 된지 조금 되었지만 세부 컴포넌트들에 대한 기능이 아직 Chain에 비해서 약하기 때문에 계속 지원되고 있고, 또한 Chain의 경우 코딩형식이 LCEL에 비해서 function을 이용하는 전통적인 방식으로, 개발자의 취향에 따라서 이해하기 편리할 수 있다.


## LLMChain

앞에서 구현한 LLMChain을 LCEL로 포팅해보면 다음과 같다.

```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

model = ChatOpenAI()

prompt = PromptTemplate.from_template("what is the famous tour place in {city}?")
chain = prompt | model
city = "Seoul"

chain.invoke({"city":city})
```

기존의 체인 코드인 `chain = LLMChain(llm=model, prompt=prompt)` 를 이렇게 변경함. `chain = prompt | model`

Input 값으로 Prompt 를 온전히 만들고 이걸 Model 에게 넘겨준다는 의미임. 


## Sequential Chain

```python
# Sequential Chain with LCEL
from operator import itemgetter

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser

prompt1 = PromptTemplate.from_template("what is the famous tour place in {city}? Tell me the name of the place only without additional comments.")
prompt2 = PromptTemplate.from_template("How can I get {place} by {transport}?")

chain1 = prompt1 | model
chain2 = prompt2 | model
chain = {"place":chain1,"transport":itemgetter("transport")} | chain2

output = chain.invoke({"city": "Seoul", "transport": "subway"})

print(output)
```

체인은 이렇게 만들어진다. `chain = {"place":chain1,"transport":itemgetter("transport")} | chain2`

chain2의 입력으로 `{"place":chain1,"transport":itemgetter("transport")}` 를 사용했는데, place 변수는 chain1의 출력값을 사용한것이고, transport 값을 itemgetter를 이용하여 애플리케이션으로 부터 받아왔다.

앞에서 부터 순차적으로 실행되기 때문에, chain2의 입력전에 “place”:chain1부분에서 chain1이 실행되게 되고, 그 결과와 함께, place와 transport가 chain2의 입력으로 전달되어 chain2가 실행되게 된다.

## Advanced Sequential Chain

병렬 실행을 포함하는 조금더 복잡한 흐름을 구현해 보면 다음과 같다. 아래 코드는 앞의 Chain을 이용하여 Advanced Sequential Chain 예제를 LCEL로 포팅한 예제이다.

호출 흐름이 복잡하기 때문에, chain간의 호출 구조를 다시 도식화 해보면 다음과 같다.

![](../images/복잡한%20llm%20chain.png)

```python
from operator import itemgetter
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.schema import StrOutputParser

prompt1 = PromptTemplate.from_template("what is the famous tour place in {city}? Tell me the name of the place only without additional comments.")
prompt2 = PromptTemplate.from_template("What is the top 5 restaurant in the {place} in city {city} without additional comments?") #output : restaurants
prompt3 = PromptTemplate.from_template("What is the best one restaurant and food for family dinner among {restaurants} ?") #output : restaurant_information
prompt4 = PromptTemplate.from_template("How can I get the {place} by using {transport}?") #output : transport_information

final_prompt = PromptTemplate.from_template("""
Please summarize the tour information with reastaurant information and transportation by using the this information.
Restaurant informations : {restaurant_information}
Transport information : {transport_information}
""")

chain1 = {"city": itemgetter("city")} | prompt1 | model | StrOutputParser()

chain2 = {"place":chain1,"city":itemgetter("city")} | prompt2 | model | StrOutputParser()
chain3 = {"restaurants":chain2} | prompt3 | model |StrOutputParser()
chain4 = {"place":chain1,"transport":itemgetter("transport")} | prompt4 | model | StrOutputParser()
final_chain = { "restaurant_information":chain3 , "transport_information":chain4 } | final_prompt | model | StrOutputParser()
output = final_chain.invoke({"city": "Seoul", "transport": "subway"})

print(output)
```





References: 
- https://bcho.tistory.com/1423