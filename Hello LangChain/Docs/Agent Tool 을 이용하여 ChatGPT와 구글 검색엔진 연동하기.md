# Agent Tool 을 이용하여 ChatGPT와 구글 검색엔진 연동하기

LLM 모델은 기본적으로 학습 당시에 저장된 데이터만을 기반으로 답변을 한다.

RAG를 이용하여 외부의 벡터 데이터 베이스에 있는 내용을 참고하여 지식 데이터를 확장할 수 있지만, 이 역시 저장한 문서에만 해당된다.

LLM 애플리케이션을 데이터를 확장하고 싶다면? 예를 들어 LLM에 저장되지 않은 데이터를 구글 서치 엔진을 통해서 검색해서 참고하거나 유투브의 비디오 스크립트를 참고할 수 있다면? 아니면 회사내의 데이터베이스의 정보를 참고해서 답변을 할 수 있다면?

이러한 요구사항에 부합하여 LLM이 외부 정보를 참고하여 답변을 할 수 있도록 기능을 제공하는 컴포넌트가 langchain의 agent와 tool 이다.

기본 LLM에는 없는 기능으로 외부 도구를 통합 연결함으로써 LLM의 기능을 확장 시키는 구조로 langchain에서 가장 유용한 기능중에 하나이다.

Agent와 tool의 동작 개념을 살펴보면 다음과 같다.

![](../images/Agent%20Tool%20Concept.png)

- 1) 질문을 Agent가 받으면, 이 질문을 답변할 수 있는 방법을 생각한다. 이 과정에서 LLM을 사용한다. LLM으로 답변이 가능하다면 그냥 답변을 한다.
- 2) 만약에 LLM으로 답변이 불가능하다면 등록되어 있는 외부 tool들을 참고한다. 각 tool들은, tool 들이 할 수 있는 기능들을 description이라는 필드에 텍스트로 서술해놓았다. 예를 들어 2023년 골프 PGA 우승자 정보를 알고 싶을때 이 정보가 없다면, LLM은 “이 정보를 모르니 인터넷에서 검색해야 겠다.” 라고 판단하고 Google search tool을 이용하여 정보를 검색해 온후, 검색한 정보에서 우승자 정보를 추출하는 방식으로 사용된다.


즉 Agent는 어떤 정보가 필요한지를 판단을 해서 질문을 다시 정의하고, 이 질문에 맞는 tool을 호출하여 정보를 추출하고, 추출한 정보를 분석하여 답변을 낼 수 있는지 판단한후, 만약에 답변에 추가적인 정보가 필요하다면 다시 질문을 하고, 질문에 맞는 tool을 선택하는 반복적인 과정을 통해서 답변에 도달한다.

이러한 패턴을 ReAct 패턴이라고 하는데, Reasoning + Action 의 합성어인데, 한글로 번역하자면 추리와 행동 정도로 볼 수 있다.

예를 들어 “서울 유명 관광지 주변의 음식점과 그 음식점의 유명한 음식을 알고 싶다”는 질문이 있다고 하자.

Agent는 아래와 같은 과정을 통해서 답변을 찾기 위해서 tool을 이용하여 정보를 수집하고, 필요한 정보를 질문을 재정의 함으로써 tool을 통해서 모아서 답변에 도달하게 된다.


1) Round 1 - Thought: 먼저 서울 유명관광지 정보가 필요하다. 구글 검색을 통해서 검색해야겠다.

2) Round 1 - Action: Google Search Tool 을 이용해서 서울 유명 관광지 정보를 검색한다.

3) Round 1 - Observation: 서울 유명 관광지는 경복궁이다. 

4) Round 2 - Thought: 이제 서울 유명 관광지가 “경복궁" 인것을 알았다. 경복궁 근처의 유명 식당 정보가 필요하다. 이를 구글 검색을 통해서 검색해야 겠다.

5) Round 2 - Action: 경복궁 주변의 유명 식당 검색.

6) Round 2 - Observation: 홍길동 레스토랑

7) Round 3 - Thought: 홍길동 레스토랑이 경복궁 주변의 유명한 식당인것을 알았다. 홍길동 식당의 유명한 메뉴 정보가 필요하다. 구글 검색 도구를 이용하여 검색해야 겠다.

8) Round 3 - Action: 홍길동 식당의 유명한 메뉴

9) Round 3 - Observation: 삼계탕

10) Final Round- Thought: 모든 필요한 정보를 얻었으니 답변을 조합하여 답해야겠다.

11) Final Round - Action: 서울의 유명한 관광지는 경복궁이며, 그 근처에 유명한 식당은 삼계탕으로 유명한 홍길동 식당입니다.


## Google Search Tool

그러면 Agent/Tool의 사용법을 실제 예제를 통해서 알아보도록 하자.

이 예제는 구글 검색 엔진을 통해서 필요한 정보를 검색해서 LLM 애플리케이션이 이 정보를 참고해서 답변을 생성하는 애플리케이션으로 Google Search API를 서비스로 제공하는 https://serper.dev/ 서비스를 사용한다. 이 서비스를 사용하기 위해서는 https://serper.dev/ 에 접속하여 가입을 한다. 

사이트에 가입을 한후 검색 API를 사용하기 위해서 API키를 발급 받아야 한다. 로그인을 한후 대쉬보드 왼쪽 메뉴에서 API Key를 선택하면 API Key를 생성하여 얻을 수 있다.

Serper API Key를 발급했으면 코드를 작성해보자.

```python
from langchain.llms.openai import OpenAI
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

load_dotenv(override=True)

model = ChatOpenAI(temperature=0.1) 

google_search = GoogleSerperAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer",
        func=google_search.run,
        description="useful for when you need to ask with search"
    )
]

agent = initialize_agent(tools = tools,
    llm = model,
    agent=AgentType.SELF_ASK_WITH_SEARCH,
    verbose=True
)

agent.run("What is the hometown of the 2001 US PGA champion?")
```

OPENAI API 키와, SERPER API 키를 os.environ 을 이용하여 환경 변수로 지정한후에, GoogleSerperAPIWrapper를 이용하여 Google Search 툴을 생성한다

이 생성한 툴을 등록해야 하는데, tools 리스트에 이를 등록한다. 이때 툴 객체는 func 변수로 넘기고, name에 툴의 이름 그리고 description에 툴을 언제 사용해야 할지 설명한다. 여기서는 “검색이 필요할 경우 유용하다"라고 설명하였다.

이 예제에서는 ReAct Agent가 아니라, 검색 엔진과 연동에 최적화된 SELF_ASK_WITH_SEARCH라는 에이전트를 사용하였는데, 이 에이전트는 검색 엔진 툴의 이름을 “Intermediate Answer”로 지정해야 하는 규약이 있기 때문에, tools에서 구글 검색 엔진의 툴 이름을 “Intermediate Answer”로 지정하였다

그리고 마지막으로 agent가 어떻게 판단하여 답변을 도출하는지를 모니터링 해보기 위해서 initialize_agent에서 verbose=True로 지정하였다.

다음은 실행 결과 이다. 

```text
Follow up: Who is the 2001 US PGA champion?
Intermediate answer: David Toms won his only major championship, one stroke ahead of runner-up Phil Mickelson. 2001 PGA Championship. Tournament information. Dates, August 16–19, ... The 2001 PGA Championship was the 83rd PGA Championship, held August 16–19 at the Atlanta Athletic Club in Duluth, Georgia, a suburb northeast of Atlanta. Gene Sarazen, three-time PGA Championship champion (1922, 1923, and 1933). ... 2001 · United States · David Toms · Atlanta Athletic ... ^ "Jason Day wins US PGA ... 16 - 19 Aug 2001. US PGA Championship. Atlanta Athletic Club, John's Creek, Georgia, USA. Flag of USA. Feed · Results · Leaderboard · Tee Times · Entry List ... Walter Hagen, Jack Nicklaus (5); Tiger Woods (4); Gene Sarazen, Sam Snead, Brooks Koepka (3); Jim Barnes, Leo Diegel, Denny Shute, Paul Runyan, ... THE PLAYERS Championship ; 1, United States T. Woods ; 2, Fiji V. Singh ; 3, Germany B. Langer ; 4, United States J. Kelly ... OnThisDay in 2001 at Atlanta Athletic Club in Georgia, David Toms won his only Major ... Duration: 0:28. Posted: Aug 19, 2021. David Toms won the 2001 PGA Championship at Atlanta Athletic Club after an historic hole-in ... Duration: 0:31. Posted: Jul 15, 2015. PGA Champions: Stroke Play Era ; 2001. David Toms. Atlanta Athletic Club. 66-65-65-69. -15 ; 2000. Tiger Woods. Valhalla C.C.. 66-67-70-67. -18.
Follow up: Where is David Toms from?
Intermediate answer: Monroe, LA
So the final answer is: Monroe, Louisiana

> Finished chain.
'Monroe, Louisiana'
```

질문은 “2001년 USA PGA 챔피언의 고향”을 질의하는 질문인데

첫번째로 Follow up: Who is the 2001 US PGA champion? 질문을 agent가 생성하여 누가 2001 US PGA 에서 우승했는지를 질의 한후, 두번째 질문에서 Followup: Where is David Toms from? 우승자인 David toms의 고향을 질의하는 두번째 질문을 하여 답변을 생성하였다. 


References: 
- https://bcho.tistory.com/1426