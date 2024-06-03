# LLM 의 비용 컨트롤 및 토큰 카운트

LLM 개발은 단순한 API 서버 개발과 다르게, 외부의 LLM API 서비스를 호출하는 형태이므로 이 API는 토큰 (단어)단위로 비용을 카운트하기 때문에 개발과 서비스 과정에서 비용이 발생한다.

그래서, 개발과 운영 과정에서 발생하는 API 호출 비용을 모니터링 하고 비용을 관리해야 하는 필요성이 있다.
- 참고: https://openai.com/api/pricing/
- 참고: https://cloud.google.com/vertex-ai/pricing?hl=ko#generative_ai_models

가격 체계는 모델 서비스 회사의 홈페이지에서 확인이 가능한데, 위의 그림과 같이, 모델의 종류나 버전 그리고 Input,Output 토큰인지, 서빙 형태가 온라인인지 배치인지 등에 따라서 다를 수 있기 때문에, 개발하고자하는 시나리오에 맞춰서 가격 예측하고 관리하는 것이 좋다.

## 비용 모니터링 및 컨트롤

API의 사용량을 확인하려면 보통 콘솔을 사용하면 되는데, OpenAI의 ChatGPT의 경우에는 https://platform.openai.com/usage 에서 API 호출수, 토큰 그리고 가격등을 모니터링할 수 있다.

API 호출에는 비용이 들어가는데, 개발을 하면서 비용 모니터링을 신경쓰지 않으면 잘못하면 대규모로 호출이 되면서 모르는 사이에 비용이 과다 청구될 수 있다. 그래서 대부분의 서비스들은 사용량에 대한 제한을 줄 수 있도록 하고 있다. 아래는 OpenAI chatGPT에 대한 사용량 제한 기능으로, “Set Monthly budget”을 통해서 한달에 11$ 이상을 사용할 수 없도록 하였고, 만약에 10$ 이상을 사용할 경우 이메일로 알리도록 하였다. 

어렵지 않지만 요금 폭탄을 방지할 수 있는 수단이니 반드시 적용하기를 권장한다.

이렇게 모니터링을 하더라도, 내가 개발중인 LLM 애플리케이션에서 호출하는 LLM API의 토큰 수가 궁금할 수 있다. 그래서 Langchain 에서는 LLM 제공자가 제공하는 콜백함수를 통해서 LLM 호출건별 토큰 수를 카운트 할 수 있다. 아래는 openai에서 제공하는 callback 함수를 이용하여 토큰수를 출력하는 코드이다

```python
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

with get_openai_callback() as callback:
    prompt = "What is famous street foods in Seoul Korea in 200 characters"
    llm.invoke(prompt)
    print(callback)
    print("Total Tokens:",callback.total_tokens)
```

“with get_openai_callback() as callback:” 코드 블럭내에 LLM 호출 코드를 작성하게 되면, LLM 을 호출한후에, callback에 호출에 대한 메타 정보를 담아서 리턴한다. 이 정보에는 아래 출력되는 내용과 같이 input/output 토큰수, 그리고 호출 금액이 포함된다.

```text

Tokens Used: 134
Prompt Tokens: 11
Completion Tokens: 123
Successful Requests: 1
Total Cost (USD): $0.00268
Total Tokens: 134
```

뒤에서 설명하겠지만 Chain이나 Agent를 사용하더라도, 위의 callback 블록을 그대로 사용할 수 있다.

References: 
- https://bcho.tistory.com/1410