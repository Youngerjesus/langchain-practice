# Chain을 이용한 복잡한 워크 플로우의 구현

## Advanced Sequential Chain 

앞의 예제는 순차적으로 LLMChain을 간단한 워크 플로우를 구현해봤다. SequentialChain은 순차적인 실행뿐만 아니라, 병렬로 LLM 호출을 하는 흐름등을 구현이 가능하다. 이번 예제에서는 조금 더 발전된 Chain의 구조를 살펴보자.

아래 예제는 도시명{city}과 교통편{transport}를 입력하면, 유명 관광지를 추천해서 그곳까지 도착하기 위한 교통편과 식당에 대한 정보를 출력하는 Chain의 구조이다.

![](../images/복잡한%20llm%20chain.png)

예제 코드를 살펴보기전에, 먼저 흐름을 보자.

애플리케이션에서 도시명{city}와 교통편{transport)를 입력받는다.
- chain1에서는 도시에서 유명한 관광지를 추천 받아 {place}로 리턴한다.
- chain2에서는 chain1의 출력값인 관광지{place}를 기반으로 근처에 레스토랑 5개를 추천받는다.
- chain3에서는 chain2에서 추천받은 5개의 레스토랑 중에서 패밀리 디너로 좋은 음식을 추천받는다.
- chain4에서는 chain1의 관광지 장소로 가기 위한 경로를 애플리케이션에서 입력받은 교통편{transport} 기반으로 추천 받는다.
- 마지막으로 final_prompt에서는 chain3과 chain4의 출력값을 합쳐서 관광지 주변의 레스토랑의 추천 음식과 교통편 정보를 함께 출력한다.


아래 예제코드를 보자. 

```python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

model = ChatOpenAI()

prompt1 = PromptTemplate.from_template("what is the famous tour place in {city}? Tell me the name of the place only without additional comments.")

prompt2 = PromptTemplate.from_template("What is the top 5 restaurant in the {place} in city {city} without additional comments?") #output : restaurants

prompt3 = PromptTemplate.from_template("What is the best one restaurant and food for family dinner among {restaurants} ?") #output : restaurant_information

prompt4 = PromptTemplate.from_template("How can I get the {place} by using {transport}?") #output : transport_information

final_prompt = PromptTemplate.from_template("""
Please summarize the tour information with reastaurant information and transportation by using the this information.
Restaurant informations : {restaurant_information}
Transport information : {transport_information}
""")

chain1 = LLMChain(llm=model,prompt=prompt1,output_key="place",verbose=True)
chain2 = LLMChain(llm=model,prompt=prompt2,output_key="restaurants",verbose=True)
chain3 = LLMChain(llm=model,prompt=prompt3,output_key="restaurant_information",verbose=True)
chain4 = LLMChain(llm=model,prompt=prompt4,output_key="transport_information",verbose=True)
final_chain = LLMChain(llm=model,prompt=final_prompt,output_key="tour_summary",verbose=True)


chain = SequentialChain(chains=[chain1,chain2,chain3,chain4,final_chain],input_variables=["city","transport"],verbose=True)
chain.run({'city':'Seoul','transport':'subway'})
```

앞의 예제에 비하면 LLMChain의 수만 늘어난것을 확인할 수 있다. 

그런데 chains에서 입력값은 chains=[chain1,chain2,chain3,chain4,final_chain] 와 같은데, chain1에서 chain2,4로 분기하고 chain3,4의 출력값을 final_chain으로 모으는 흐름은 어떻게 표현했을까? 

답은 output_key와 템플릿에 있다.

chain1의 출력값 키는 chain1 = LLMChain(llm=model,prompt=prompt1,output_key="place",verbose=True) 와 같이 {place}가 된다. 

이 {place}는 chain2와4의 프롬프트에서 다음과 같이 입력값으로 사용된다.

## Router Chain

지금까지 순차적 LLMChain을 실행하는 방법에 대해서 알아보았다. 다른 방법으로는 입력값에 따라서 Chain을 선택해서 분기 하는 RouterChain에 대해서 알아보겠다.

여행 챗봇에서 레스토랑 정보, 교통편정보, 여행지 정보에 대한 LLM 모델 3개를 LLMChain으로 만들어놓고, 질문의 종류에 따라서 적절한 LLMChain으로 라우팅 하는 시나리오이다.

![](../images/Router%20llm%20chain.png)

만약에 적절한 LLMChain이 없는 경우, 예를 들어 여행정보가 아니라 전혀 관계 없는 질문이 들어올 경우에는 앞에 언급한 3가지 LLMChain이 아니라 Default Chain을 사용하도록 구현한다.

이제 예제 코드를 보자. 

```python
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

model = ChatOpenAI()


restaurant_template = """
You are a tourist guide. You know many good restaurants around the tourist destination.
You can recommend good foods and restaurants.
Here is a question:
{input}
"""

transport_template = """
You are a tourist guide. You have a lot of knowledge in public transportation.
You can provide information about public transportation to help tourists get to the tourist destination.
Here is a question:
{input}
"""

destination_template = """
You are a tourist guide. You know many good tourist places.
You can recommend good tourist places to the tourists.
Here is a question:
{input}
"""


prompt_infos = [
    {
        "name":"restaurants",
        "description":"Good for recommending restaurants around the tourist destinations",
        "prompt_template": restaurant_template
    },
    {
        "name":"transport",
        "description":"Good for guiding the transport to get the place",
        "prompt_template": transport_template
    },
    {
        "name":"destination",
        "description":"Good for recommending place to tour",
        "prompt_template": destination_template
    }
]

destination_chains = {}

for prompt_info in prompt_infos:
    name = prompt_info["name"]
    prompt = PromptTemplate.from_template(prompt_info["prompt_template"])
    chain = LLMChain(llm = model, prompt=prompt, verbose=True)
    destination_chains[name] = chain

default_prompt = PromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=model, prompt=default_prompt)

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
)

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)

router_chain = LLMRouterChain.from_llm(model, router_prompt)


chain = MultiPromptChain(router_chain=router_chain, destination_chains=destination_chains, default_chain=default_chain, verbose=True)

chain.run("What is best restaurant in Seoul?")
```




References: 
- https://bcho.tistory.com/1421