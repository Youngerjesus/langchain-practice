# ChatGPT에서 대화 히스토리 유지하기

LLM 기반의 챗봇 에서는 질문에 대한 답변을 기존의 대화의 내용이나 컨텍스트(문맥)을 참고하는 경우가 많다. 

예를 들면, “서울에서 유명한 여행지는 어디야?” 라는 질문 후에, “그 근처에 맛있는 식당이 어디있어?” 라고 질문을 하면 챗봇은 서울의 유명한 여행지를 추천한 내용을 기반으로 해서, 그 근처의 맛있는 식당을 추천한다.

이렇게 기존 대화 내용을 참고하려면 챗봇이 기존 대화 내용을 알고 있어야 하는데, LLM 모델은 미리 학습이 된 모델로, 대화 내용을 기억할 수 있는 기능이 없고, Stateless 형태로 질문에 대한 답변만을 제공하는데, 최적화가 되어 있다.

그렇다면 LLM 기반의 애플리케이션들은 어떻게 기존의 컨택스트를 기억할 수 있을까? 이렇게 기존의 컨택스트를 기억하는 기능이 langchain에서 Memory라는 컴포넌트이다.


기본적인 개념은 다음과 같다. 

채팅 애플리케이션에 질문(Question 1)을 하면 애플리케이션에서 미리 정의되어 있는 프롬프트 템플릿에 질문을 추가하여 LLM에 질문한다. 답변이 나오면 질문 (Question 1)과 답변 (Answer 1)을 메모리에 저장한다.

![](../images/LangChain%20Memory.png)

다음 대화에서 질문이 추가로 들어오면, 메모리에 저장된 기존의 대화 내용 (Question 1, Answer 1)을 불러서, 프롬프트 템플릿에 컨텍스트 정보로 추가하고, 여기에 더해서 새로운 질문 (Question 2)를 추가하여 LLM에 질의 한다.

![](../images/LangChain%20Memory%202.png)

## Conversational Buffer Memory

메모리를 지원하는 컴포넌트는 여러가지가 있는데, 그중에서 가장 기본적인 ConversationalBufferMemory를 먼저 살펴보자. 

아래 예제는 ConversationalBufferMemory를 테스트하는 코드이다.

ConversationalBufferMemory는 대화 내용을 그대로 저장하는 메모리 형태이다.

이 메모리에 대화내용 (컨택스트)를 저장하기 위해서는 save_context를 이용하여, 사람의 질문은 input이라는 키로 전달하고,  챗봇의 답변은 output 키에 서술한다.

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
memory.clear()

memory.save_context({"input":"Hello chatbot!"},{"output":"Hello. How can I help you?"})
memory.save_context({"input":"My name is Terry"},{"output":"Nice to meet you Terry"})
memory.save_context({"input":"Where is Seoul?"},{"output":"Seoul is in Korea"})

memory.load_memory_variables({})
```

이때 두 가지 옵션을 줄 수 있는 데, 첫번째는 memory_key, 두번째는 return_message 이다.

ConversationBufferMemory에 저장된 기존의 대화 내용은 결과적으로 프롬프트에 삽입되게 되는데, 프롬프트에 삽입되는 위치를 템플릿 변수로 지정한다.

이때 이 템플릿 변수의 이름이 “memory_key”의 값이 된다.

return_messages는 메모리에서 대화 내용을 꺼낼 때 어떤 형식으로 꺼낼것인가인데, return_messages=True로 되어 있으면 아래 출력 결과를 HumanMessage, AIMessage 식의 리스트형태로 리턴을 해준다. 이 메세지 포맷은 Langchain에서 ChatModel을 사용할때 사용해야 하는 포맷이다.


```text
{'chat_history': [HumanMessage(content='Hello chatbot!'),
    AIMessage(content='Hello. How can I help you?'),
    HumanMessage(content='My name is Terry'),
    AIMessage(content='Nice to meet you Terry'),
    HumanMessage(content='Where is Seoul?'),
    AIMessage(content='Seoul is in Korea')]}
```

반대로 return_messages=False로 할 경우, 채팅 히스토리를 리스트 형식이 아니라, 아래와 같이 문자열로 리턴한다.

```text
{'chat_history': 'Human: Hello chatbot!\nAI: Hello. How can I help you?\nHuman: My name is Terry\nAI: Nice to meet you Terry\nHuman: Where is Seoul?\nAI: Seoul is in Korea'}
```

## 챗봇에서 Conversational Buffer Memory

그러면, CoversationalBufferMemory를 이용해서, 챗봇 서비스를 제공하는 코드를 살펴보자.

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import os

llm = ChatOpenAI(temperature=0.1)

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a chatbot having a conversation with a human."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)



conversation.invoke({"question": "hi my name is Terry"})
conversation.invoke({"question": "Can you recommend fun activities for me?"})
conversation.invoke({"question": "What is my name?"})
memory.load_memory_variables({})
```

채팅 애플리케이션이기 때문에, ChatOpenAI로 채팅 모델을 만들고, 채팅에서 사용할 프롬프트 템플릿을 정의한다. 아래와 같이 템플릿에는 SystemMessage에 LLM 모델의 역할을 챗봇이라고 정의했는데, 필요하다면 추가적인 프롬프트를 넣을 수 있다.

MessagePlaceholder에는 외부로 부터 받은 컨택스트를 프롬프트에 포함하기 위한 위치인데, “chat_history”를 키로 해서, 이 부분에는 메모리에 저장된 기존의 채팅 히스토리 내용을 삽입한다

이 키 값은 이후에 선언하는 ConversationalBufferMemory의 memory_key의 값과 일치해야 한다.

마지막으로 HumanMessagePrompt에는 {question}으로 들어온 내용을 삽입하는데, 이는 사용자가 챗봇에게 질의한 내용이 된다.

프롬프트가 준비되었으면 ConversationalBufferMemory를 생성하고, return_messages=True 로 하여, 메모리의 히스토리를 리턴할때 챗봇 형태의 리스트 데이터형으로 리턴하도록 한다. ConverationalBufferMemory에서 memory_key를 “chat_history”로 하여, 프롬프트에 “chat_history” 프롬프트 변수가 있는 곳에 채팅 히스토리를 삽입하도록 연결한다.


References: 
- https://bcho.tistory.com/1429