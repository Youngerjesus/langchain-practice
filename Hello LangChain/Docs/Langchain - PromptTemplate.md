# Langchain - PromptTemplate


LLM 기반 애플리케이션 개발에서 가장 중요한 것중의 하나가 프롬프트 엔지니어링이다

프롬프트를 잘 만들어서 원하는 답변을 얻어야 하는데, 프롬프트 템플릿은 프롬프트를 재 사용할 수 있도록 해주고, 여러 프롬프트를 구조화하여, 적절한 프롬프트를 생성할 수 있도록 한다

프롬프트 템플릿은 개념적으로 이해하기 쉬운 스트링(문자열) 연산이지만, 잘 사용하면 강력한 기능이 될 수 있기 때문에 숙지하기 바란다.

프롬프트 템플릿은 프롬프트를 생성하기 위한 템플릿이다.

예를 들어 “Tell me about {city_name} city” 라는 템플릿이 있으면, {city_name}은 가변 변수가 되고, 프롬프트를 생성할때 이 값을 지정해서, 프롬프트를 생성할 수 있다. 

만약 이 템플릿에서 city_name을 “Seoul”로 지정하고 싶다면, template.format(city_name=”Seoul”) 이라는 식으로 정의하면 템플릿에 값을 채워 넣어서 프롬프트를 생설할 수 있다.

아래 예제는 프롬프트 템플릿 예제로 "Tell me a {adjective} {topic} in {city} in 300 characters. 템플릿을 생성한 후에, 각각 다른 값을 채워 넣어서 두번 호출하는 예제이다.

```python
from langchain import PromptTemplate
from langchain.llms import OpenAI

template = PromptTemplate.from_template(
    "Tell me a {adjective} {topic} in {city} in 300 characters."
)

prompt= template.format(adjective="famous", topic="place", city="seoul")

print(prompt)
print(llm.invoke(prompt))

prompt = template.format(adjective="popular", topic="reastaurant", city="San francisco")

print(prompt)
print(llm.invoke(prompt))
```

출력 결과: 
```text
Tell me a famous place in seoul in 300 characters.
content="Gyeongbokgung Palace is a famous historical site in Seoul, South Korea. Built in 1395, it served as the main royal palace of the Joseon dynasty. Visitors can explore the beautiful architecture, traditional Korean gardens, and learn about the country's rich history." response_metadata={'token_usage': {'completion_tokens': 58, 'prompt_tokens': 20, 'total_tokens': 78}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-df65c33e-f4b0-41cc-be27-7824810ec148-0'
Tell me a popular reastaurant in San francisco in 300 characters.
content='One of the most popular restaurants in San Francisco is Gary Danko, known for its upscale dining experience and innovative American cuisine. With a Michelin-starred chef at the helm, guests can enjoy a multi-course tasting menu or choose from a variety of a la carte options. Reservations are highly recommended.' response_metadata={'token_usage': {'completion_tokens': 61, 'prompt_tokens': 23, 'total_tokens': 84}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-01c2fa2f-2313-4bf6-a21d-e837f7884f39-0'
```

## 직렬화를 이용한 프롬프트 저장과 로딩

LLM 애플리케이션을 개발하면, 많은 작업중의 하나가 프롬프트 튜닝이다. 그래서 프롬프트를 수시로 변경해야 하는 경우도 있고, 국제화된 애플리케이션을 개발하기 위해서는 같은 프롬프트도 다국어로 개발해야 할 경우가 있다. 이런 경우 애플리케이션 코드내에서 하드코딩된 프롬프트보다. 프롬프트 템플릿을 별도의 파일로 저장하여, 애플리케이션에서 로드하여 사용할 수 있다. 이를 Serialization (직렬화)라고 한다. 

프롬프트 템플릿을 저장하는 방법은 간단하다. 아래 코드에서 처럼 PrompteTemplate객체를 생성한후에 save(“{JSON 파일명}”)을 해주면 해당 템플릿이 파일로 저장된다.

```python
from langchain import PromptTemplate

template = PromptTemplate.from_template(
    "Tell me a {adjective} {topic} in {city} in 300 characters."
)

template.save("template.json")
```

저장 결과:

```json
{
    "name": null,
    "input_variables": [
        "adjective",
        "city",
        "topic"
    ],
    "input_types": {},
    "output_parser": null,
    "partial_variables": {},
    "metadata": null,
    "tags": null,
    "template": "Tell me a {adjective} {topic} in {city} in 300 characters.",
    "template_format": "f-string",
    "validate_template": false,
    "_type": "prompt"
}
```

이렇게 저장된 템플릿은 langchain.prompts 패키지의 load_prompt 함수를 이용하여 다시 부를 수 있다. 

아래는 template.json에 저장된 템플릿을 로딩한후에, adjective, topic,city의 값을 채워서 프롬프트로 화면에 출력하는 예제이다.

```python
from langchain.prompts import load_prompt

loaded_template = load_prompt("template.json")
prompt = loaded_template.format(adjective="popular", topic="cafe", city="San francisco")
print(prompt)
```

## 채팅 프롬프트 템플릿

일반적인 텍스트 모델과 같이 채팅 모델 역시 프롬프트를 지원한다.

단 채팅의 경우, SystemMessage, HumanMessage,AIMessage 들이 순차 리스트 형식으로 저장이 되기 때문에, 템플릿도 역시 같은 형태로 템플릿을 정의한다.

아래는 채팅 프롬프트 템플릿 예제이다. 

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a tour guide."),
        ("human","I'm planning to visit {Country}."),
        ("ai","I'm tour guide for {Country}."),
        ("human","{user_input}")
    ]
)

prompt = chat_template.format_messages(Country="Korea",user_input="What is top5 best place to go there?")

print("Prompt :",prompt)
print("-"*30)

aiMessage=llm.invoke(prompt)
print(aiMessage)
```

chat_template에 system,human,ai message 히스토리를 정의하였고, Country와 user_input을 변수로 지정하였다. 

PromptTemplate과 마찬가지로 format_messages 메서드를 이용하여, 템플릿에 값을 채워서 프롬프트를 생성한다.

아래 실행결과로 생성된 프롬프트를 보면, Country에 Korea가 채워져 있고, user_input에는 “What is top 5 best place to go there?”라는 질문이 채워진것을 확인할 수 있다.

## 프롬프트 조합

앞서 간단한 단일 프롬프트만 살펴보았는데,상황에 따라서 여러개의 프롬프트를 조합하거나 또는 프롬프트내에 대한 프롬프트를 포함하여 새로운 프롬프트를 만들어낼 수 있다.

이러한 기법은 프롬프트 조합 (Prompt Composition)이라고 한다. 

다음 예제는 role_prompt와 question_prompt 두개를 합해서 새로운 full_prompt를 만들어내는 예제이다.

```python
from langchain.prompts import PromptTemplate

role_prompt = PromptTemplate.from_template("You are tour guide for {country}")
question_prompt = PromptTemplate.from_template("Please tell me about {interest} in {country}")

full_prompt=role_prompt + question_prompt
full_prompt.format(country="Korea",interest="famous place to visit")
```

단순하게 role_prompt + question_prompt 로 새로운 프롬프트를 조합할 수 있음을 확인할 수 있다.

단순한 조합이외에도, 프롬프트가 다른 프롬프트들을 포함할 수 있는 기법이 있는데, 이를 프롬프트 파이프라이닝 (prompt pipelining)이라고 한다.

아래 예제는 full_prompt 안에 role_prompt와 question_prompt를 포함하는 예제이다.

```python
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts import PromptTemplate

full_template ="""
    {role}
    {question}
    Please do not reply with anything other than information related to travel to {country} and reply 'I cannot answer.'
"""

full_prompt = PromptTemplate.from_template(full_template)
role_prompt = PromptTemplate.from_template("You are tour guide for {country}")
question_prompt = PromptTemplate.from_template("Please tell me about {interest} in {country}")



# composition
input_prompts = [
    ("role",role_prompt),
    ("question",question_prompt)
]

pipeline_prompt = PipelinePromptTemplate(final_prompt=full_prompt,pipeline_prompts=input_prompts)

prompt_text = pipeline_prompt.format(
    country="Korea",
    interest="famous place to visit"
)


print(prompt_text)
```

## 부분 프롬프트 템플릿

프롬프트 템플릿의 변수에 값을 채워넣는 방법중에서 Partial Prompt Template이라는 방식이 있다.

이 방식은 프롬프트의 변수 값을 한번에 채워 넣는 것이 아니라, 이 중 일부만 먼저 채워 넣고 나머지는 나중에 채워 넣는 방식이다. 

예를 들어 템플릿 변수 A,B,C가 있을때, 템플릿을 생성할때 A를 먼저 채워 넣고 나중에 B,C를 채워 넣는 방식이다.

애플리케이션 코드내에서 템플릿 생성시 이미 알고 있는 값이 있을때 사용하기 편리한 방식이다.

아래 코드를 보면, topic,city 두개의 템플릿 변수가 있는데, 템플릿을 생성할때 city에 대한 값을 prompt.partial(city=”Seoul”)을 통해서 먼저 채워 놓았고, 그 다음줄에서 format 메서드를 이용하여 topic을 채워 놓았다.

```python
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(template="What is famous {topic} in {city}?",input_variables=["topic","city"])
partial_prompt = prompt.partial(city="Seoul")
print(partial_prompt.format(topic="food"))
```

또는 아래 예제와 같이 PromptTemplate을 생성할때, partial_variables를 인자로, 이미 알고 있는 변수의 값을 넘겨줄 수 있다.

```python
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(template="What is famous {topic} in {city}?",input_variables=["topic"],partial_variables={"city":"seoul"})

print(prompt.format(topic="food"))
```


References: 
- https://bcho.tistory.com/1413

