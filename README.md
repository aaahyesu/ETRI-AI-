제2회 ETRI 휴먼이해 인공지능 논문경진대회
==========
[제2회 ETRI 휴먼이해 인공지능 논문경진대회](https://aifactory.space/competition/detail/2234)   

## 라이프로그와 수면의 정보 연관성을 활용한 수면의 질 예측 알고리즘

### Setup
```
install 형태로 다쓸지 
환경 세팅처럼 할지 고민
```
### 데이터셋
* [ETRI 라이프로그 데이터셋 (2020-2018)](https://nanum.etri.re.kr/share/schung1/ETRILifelogDataset2020?lang=ko_KR) 에서 아래 파일들 다운로드
  - user01-06 data (user01-06.7z)
  - user07-10 data (user07-10.7z)
  - user11-12 data (user11-12.7z)
  - user21-25 data (user21-25.7z)
  - user26-30 data (user26-30.7z)
  - 2020 실험자별 정보 (user_info_2020.csv)
  - 2020 수면 측정 데이터 (user_sleep_2020.csv)
 
### 데이터 전처리
  1. user01 - user30 데이터 전처리   
    - 각 user파일의 timestamp_label.csv에서 userId, ts, actionOption, date를 추출한다.   
    - actionOption컬럼의 데이터를 신체활동 분류표에 따른 강도로 분류한다.   
    - 강도 분류에 따라 좌식행동-1, 저강도-2, 중강도-3, 고강도-4 로 변환하였다. 
  
  
### 데이터 분석 결과

### Model Architecture

### 실행 방법
### 디렉토리 설명

### 결과

<br>
<br>
<br>

2.1 환경설정
여러분의 PC나 서버에 GPU가 있고 cuda setting이 되어있어야합니다.
여러분의 환경에 이 repo를 clone합니다 : git clone <this_repo>
requirements libraries를 확인합니다 : pip install -r requirements.txt

2.2 데이터셋 다운로드
[ETRI 라이프로그](https://nanum.etri.re.kr/share/schung1/ETRILifelogDataset2020?lang=ko_KR) dataset을 다운로드하세요.
로그인 후 협약서 승인이 필요할 수 있습니다.

데이터 파일
- user01-06 data
- user07-10 data
- user11-12 data
- user21-25 data
- user26-30 data
- 2020 실험자별 정보
- 2020 수면 측정 데이터



 'ETRI_2022_AI_Competition/data' 폴더에 넣으세요.

 <2023_ETRI_AI_Competition>
                    ├ <2020>
                        └ <user01-30>
                            ├ <user01>
                            ├ ...
                            └ <user30>  
                                ├ <1598827200>
                                ├ ...
                                └ <1601165700>
                                    ├ ...
                                    └ 1601165700_label.csv
                        ├ user_info_2020.csv
                        └ user_sleep_2020.csv

                    ├ prerprocessing.ipynb
                    ├ constants.py
                    ├ model.py
                    ├ utils.py
                    ├ requirements.txt
                    └ README.md
