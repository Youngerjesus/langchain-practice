# 자연어로 SQL 생성하기

지금까지 살펴본 Chain 은 모두 LLMChain으로, 입력값을 프롬프트에 삽입하여 모델에 입력해서 결과를 리턴하는 형태였다.

Chain 기능을 통해서 연결될 수 있는 체인은 LLMChain 뿐만 아니라 단순하게 출력값을 포맷팅 하는 체인이나, 아니면 문서 파일을 읽어드리는 체인등 여러가지 용도의 체인이 있을 수 있다.

Chain 기능을 통해서 연결될 수 있는 체인은 LLMChain 뿐만 아니라 단순하게 출력값을 포맷팅 하는 체인이나, 아니면 문서 파일을 읽어드리는 체인등 여러가지 용도의 체인이 있을 수 있다.

유틸리티 체인중에서 대표적인 체인인 create_sql_query_chain을 알아보자. 이 체인은 데이터베이스의 스키마를 기반으로 입력된 질문을 SQL로 변환해주는 역할을 한다.

이 예제는 미국의 영화/TV 프로에 대한 랭킹 정보 사이트인 imdb.com의 데이터를 기반으로, 자연어 질의를 통해서 SQL 쿼리를 생성하는 코드를 작성하고자 한다.

예를 들어 “영화중에서 평점이 8.0이상이고, 2008년 이후 상영된 영화들을 알려줘" 라는 입력을 하면 “SELECT "primaryTitle" FROM my_table WHERE "titleType" = 'movie' AND "startYear" >= 2008 AND "averageRating" >= 8.0 ORDER BY "averageRating"” 와 같은 SQL 을 생성해주는 시나리오이다.

## 데이터셋 다운로드

예제를 구현하기 위해서는 데이타 셋을 로드해야 한다. 데이타셋은 아래와 같이 https://datasets.imdbws.com/ 에서 다운로드 한다.

title.basics.tsv.gz 와 title.rating.tsv.gz 를 다운받으면 된다. 

다운로드 된 데이터셋은 압춧을 풀어서, Pandas 데이터 프레임에 저장한다.

두개의 데이터 셋 (영화 기본 정보 파일과, 영화 평점)을 받았기 때문에 영화 ID tconst를 이용하여 병합하고, 100개의 레코드만 랜덤으로 샘플링하여 Pandas 데이터 프레임에 저장한다

```python
import gzip, shutil
import pandas as pd

with gzip.open("./dataset/title.basics.tsv.gz", 'rb') as f_in: 
    with open("./dataset/title.basics.tsv", 'wb') as f_out: 
        shutil.copyfileobj(f_in, f_out)

with gzip.open("./dataset/title.ratings.tsv.gz", 'rb') as f_in: 
    with open("./dataset/title.ratings.tsv", 'wb') as f_out: 
        shutil.copyfileobj(f_in, f_out)
        

basics = pd.read_csv('./dataset/title.basics.tsv', sep='\t', low_memory=False, na_values=['\\N'])
ratings = pd.read_csv('./dataset/title.ratings.tsv', sep='\t', low_memory=False, na_values=['\\N'])

full_data = pd.merge(basics, ratings, on="tconst")
samples = full_data.sample(n=100,random_state=42)
samples.head()
```

## 데이터베이스 생성

데이터 프레임이 만들어졌으면, 이 데이터를 데이터 베이스에 로드한다.

간단한 예제이기 때문에, sqlite3를 사용하였다. 데이터 프레임 내용을 그대로 로드하고, sqlite3 데이터 베이스 파일 example.db에 저장하였다.

```python
import pandas as pd
import sqlite3

conn = sqlite3.connect('example.db')
cursor = conn.cursor()
samples.to_sql('my_table', conn, index=False, if_exists='replace')
query = "SELECT * FROM my_table"
result = pd.read_sql_query(query, conn)
print(result)
conn.close()
```

## SQL 쿼리 생성하기

데이터베이스가 준비되었으면, 체인을 생성하고 호출해보자.

create_sql_query_chain(model, db,k=20)에 question 입력변수에 질문을 입력하면 된다. 아래는 “영화중에서 평점이 8점 이상이고, 2008년 이후 상영된 영화 이름”을 조회하는 질의이다. k=20은 20개의 결과만 리턴하도록 하는 설정이다.

```python
from langchain.llms import OpenAI
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain.schema import StrOutputParser
from langchain.chat_models import ChatOpenAI

model = ChatOpenAI()

db = SQLDatabase.from_uri("sqlite:///example.db")
chain = create_sql_query_chain(model, db,k=20) | StrOutputParser()

result = chain.invoke({"question": """Please provide a list of movies that have an averageRating of 8.0 or higher and have been commercially available since 2008."""})

print(result)
```

References: 
- https://bcho.tistory.com/1424
