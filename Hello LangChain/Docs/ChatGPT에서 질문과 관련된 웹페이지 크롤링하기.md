# ChatGPT에서 질문과 관련된 웹페이지 크롤링하기

Langchain 에서는 Agent가 사용하는 Tool을 사용자가 쉽게 개발해서 추가할 수 있다.

이번 예제에서는 DuckDuckSearch Tool을 이용하여, 질문에 관련된 웹사이트를 검색한후, 그 중 한 웹사이트의 내용을 크롤링해서 웹페이지 내용을 읽어온후에, 이를 요약하는 예제를 만들어 본다.

이를 위해서 웹페이지를 크롤링하는 툴을 BeautifulSoup 을 이용해서 만들어 본다.

커스텀 툴을 정의하는 방법은 몇가지가 있는데, 이 예제에서는 데코레이터를 사용하는 방법과 StructuredTool을 사용하는 방법 두가지를 살펴보자.

먼저 decorator를 사용하는 방법이다. 

```python
import requests
from bs4 import BeautifulSoup

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:90.0) Gecko/20100101 Firefox/90.0'
}

def parse_html(content) -> str:
    soup = BeautifulSoup(content, 'html.parser')
    text_content_with_links = soup.get_text()[:3000]
    return text_content_with_links

@tool
def web_fetch_tool(url:str) -> str:
    """Useful to fetches the contents of a web page"""
    if isinstance(url,list):
        url = url[0]
    print("Fetch_web_page URL :",url)
    response = requests.get(url, headers=HEADERS)
    return parse_html(response.content)

```

web_fetch_tool 라는 이름으로 툴을 만들었는데, 툴을 만들기 위해서는 함수에 @tool 이라는 데코레이터를 선언해주면 된다. 

이때 input 과 return 에 대한 데이터 타입을 반드시 지정해줘야 한다. 입출력 인자는 함수 선언시에 정의된 입출력의 변수명과 변수 타입을 tool의 입출력 정보로 사용하기 때문이다.

그리고 함수 첫줄에 “”” 으로 주석을 달아주면, 주석이 툴에 대한 description이 된다.

즉 위의 예제에서는 툴에 대한 정보는 아래와 같이 정의 된다.

```python
Tool name : web_fetch_tool
Tool description : web_fetch_tool(url: str) -> str - Useful to fetches the contents of a web page
Tool argument : {'url': {'title': 'Url', 'type': 'string'}}
```

web_fetch_tool은 url을 인자로 받은 후에, request.get(url)을 통해서 url에 있는 웹페이지를 크롤링한다.

크롤링을 위해서 HTTP Header의 내용을 HEADERS 변수에 저장하여 전달하였다.

이렇게 크롤링 된 HTML은 HTML 태그 부분을 제외하고, 텍스트 부분만 추출하기 위해서 parse_html에서 BeautifulSoup 의 HTML Parser를 이용해서, 텍스트 부분만 추출하여 리턴한다.

decorator를 사용하는 방법 이외에도 StructuredTool 을 이용하는 방법이 있다. 아래는 StructuredTool을 이용하여 fetch_web_page 함수를 툴로 등록하는 코드이다.

func에 툴로 등록할 함수 이름을 지정하고, name에 툴의 이름, 그리고 마지막으로 description에 툴에 대한 설명을 추가한다.

```python
def fetch_web_page(url:str) -> str:
    if isinstance(url,list):
        url = url[0]
    print("Fetch_web_page URL :",url)
    response = requests.get(url, headers=HEADERS)
    return parse_html(response.content)

web_fetch_tool = StructuredTool.from_function(
    func=fetch_web_page,
    name="WebFetcher",
    description="Useful to fetches the contents of a web page"
)
```

지정된 웹 페이지 URL을 크롤링 하는 툴을 만들었으면, 이제 전체 애플리케이션을 만들어보자.

```python
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import Tool, DuckDuckGoSearchResults
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import BaseTool, StructuredTool, tool
import os

from langchain.chat_models import ChatOpenAI

load_dotenv(override=True)

model = ChatOpenAI(temperature=0.1) 

ddg_search = DuckDuckGoSearchResults()
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:90.0) Gecko/20100101 Firefox/90.0'
}


def parse_html(content) -> str:
    soup = BeautifulSoup(content, 'html.parser')
    text_content_with_links = soup.get_text()[:3000]
    return text_content_with_links

def fetch_web_page(url:str) -> str: 
    if isinstance(url,list):
        url = url[0]
    print("Fetch_web_page URL :",url)
    response = requests.get(url, headers=HEADERS)
    return parse_html(response.content)

web_fetch_tool = StructuredTool.from_function(
    func=fetch_web_page,
    name="WebFetcher",
    description="Useful to fetches the contents of a web page"
)

summarization_chain = LLMChain(
    llm=model,
    prompt=PromptTemplate.from_template("Summarize the following content: {content}")
)

summarize_tool = Tool.from_function(
    func=summarization_chain.run,
    name="Summarizer",
    description="Useful to summarizes a web page"
)

tools = [ddg_search, web_fetch_tool, summarize_tool]

template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)

agent = create_react_agent(model,tools,prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
    handle_parsing_errors=True,
)

question = "Tell me about best Korean reastaurant in Seoul.\
Use search tool to find the information.\
To get the details, please fetch the contents from the web sites.\
Summarize the details in 1000 words."

print(agent_executor.invoke({"input":question}))
```

이 예제는 DuckDuckGo 서치를 이용하여, 필요한 정보를 검색하도록 하고, DuckDuckGo 서치에서 검색된 페이지의 URL을 필요한 경우 web_fetch_tool로 전달하여, URL에서 부터 본문을 추출한 후, summarize_tool을 이용해서 요약한 정보를 출력하도록 하는 예제이다.

먼저 duckduckgo Search 툴을 등록한다. https://duckduckgo.com/ 는 구글과 같은 검색엔진으로, 사용자 정보를 수집하지 않고, 개인 정보를 보호하는 기능이 강화된 검색 엔진이다

파이썬의 DuckDuckGoSearchResult() 는 검색 결과에 검색 결과 텍스트 뿐만 아니라, URL 까지 같이 리턴하기 때문, 특정 페이지의 내용을 모두 크롤링하는 이 예제의 시나리오에 적절하기 때문에 사용하였다. 







References:
- https://bcho.tistory.com/1428