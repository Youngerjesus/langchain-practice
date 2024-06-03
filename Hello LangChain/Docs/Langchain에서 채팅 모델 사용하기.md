# Langchain에서 채팅 모델 사용하기

텍스트 LLM 모델이 단일 입력에 대한 단일 출력을 지원하는 모델이라면, Chat 모델은 기존의 대화 히스토리를 기반으로 해서 질문에 대한 답변을 출력한다. 

이를 위해서 LangChain은 4가지 메시지 타입을 지원하는데, SystemMessage, HumanMessage, AIMessage가 주로 사용된다.
- SystemMessage: 
  - SystemMessage는 챗봇 에게 개발자가 명령을 내리기 위해서 사용하는 메시지이다. 
  - 예를 들어 쳇봇이 “여행가이드 역할을 하며, 여행에 관련되지 않은 질문은 답변하지 말아라" 라는 등의 역할에 대한 명령이나 대화에 대한 가이드라인이나 제약 사항을 설정할 수 있다.
- HumanMessage: 
  - HumanMessage는 쳇봇 사용자가 쳇봇에게 질의 하는 대화 이다. 예를 들어 여행가이드 쳇봇에게 “서울에서 가장인기있는 관광지 5개를 추천해줘.” 와 같은 대화가 될 수 있다.
- AIMessage: 
  - AIMessage는 쳇봇이 답변한 대화 내용이다.

아래 예제를 보자. 

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage

messages = [
    SystemMessage(content="You are the travel agent. You can provide travel itinery to the user"),
    HumanMessage(content="Where is the top 3 popular space for tourist in Seoul?"),
]


aiMessage=llm.invoke(messages)
print(aiMessage.content)
```

먼저 ChatOpenAI 객체로 ChatGPT 모델을 이용하여 채팅을 지원한 모델을 생성한다.

앞에서도 언급하였지만, 모델 제공자에 따라서 다양한 모델이 지원하고 있으며, 특정 모델 제공자는 채팅 전용 모델을 지원하는 경우도 있으니, (예 구글 PaLM2 모델은 채팅 전용 모델로 Bison-Chat 모델을 별도로 제공한다. ) 반드시 모델 사용전에 적절한 모델을 확인하고 선택하기 바란다.

messages 리스트를 보면 처음 SystemMessage로, 이 쳇봇은 여행 에이전트라는 역할을 정의하였다. 다음에 HumanMessage를 통하여, 서울에서 가장 인기있는 장소 3군데에 대한 추천을 요청하였다.

그 후에, chat.invoke를 통하여, 모델을 호출하였다.

그러면 다음 대화를 호출하기 위해서는 어떻게 해야 할까? 다음 코드를 추가해보자

```python

# Append AIMessages into Chat History
messages.append(aiMessage)
print("-"*30)

# Add new conversation
messages.append(HumanMessage(content="Which transport can I use to visit the places?"))
aiMessage=chat.invoke(messages)
print(aiMessage.content)
print("-"*30)


# Add new conversation
messages.append(HumanMessage(content="Where is the good restaurant for family near the placee?"))
aiMessage=chat.invoke(messages)
print(aiMessage.content)
```

위의 코드를 보면 message.append(aiMessage)를 이용하여, 쳇봇의 답변을 messages 리스트에 저장을 하였고, 다음 질문도 messages.append를 이용하여 리스트에 추가한 후에 모델을 호출 하였다. 즉, 모델을 호출할때 지금의 대화문장만으로 호출하는 것이 아니라, 기존의 대화 내용을 모두 포함해서 다시 호출을 하는 구조이다.

아래 개념과 같이 처음에 messages에는 SystemMessage와 HumanMessage만 있었지만, HumanMessage에 대한 결과를 합해서 messages에 저장한후, 이를 Conversation History로 사용하여, 쳇봇이 지금까지 사용자와 어떤 대화를 했는지에 대한 컨택스트를 유지할 수 있도록 해준다. 

이렇게 직접 리스트 자료구조를 사용해도 되지만, LangChain에서는 대화 기록을 좀더 쉽게 유지할 수 있도록 ChatMessageHistory라는 클래스를 제공한다. 다음은 ChatMessageHistory를 이용하여 작성한 코드이다. 

```python
#ChatMessageHistory is used if you are managing memory outside of a chian directly
from langchain.memory import ChatMessageHistory

history = ChatMessageHistory()

history.add_user_message("Where is the top 3 popular space for tourist in Seoul?")
aiMessage = llm.invoke(history.messages)

history.add_ai_message(aiMessage.content)
print(aiMessage.content)
print("-"*20)

history.add_user_message("Which transport can I use to visit the places?")
aiMessage = llm.invoke(history.messages)
history.add_ai_message(aiMessage.content)
print(aiMessage.content)
```

이런식으로, 메세지의 히스토리를 저장할 수 있지만, LLM 모델에서 입력으로 받아들일 수 있는 문장의 길이는 제약이 있다.

예를 들어 chatgpt-3.5-turbo 의 경우 최대 4K 토큰만 지원을 하기 때문에, 지난 대화의 길이가 4K를 넘어서게 되면 지난 대화 내용을 잃어버릴 수 있다.

이를 위해서 지난 대화 내용을 요약한다던지 등의 기법을 사용할 수 있는데, 이는 나중에 Memory 부분에서 다시 설명하도록 한다. 


References: 
- https://bcho.tistory.com/1412