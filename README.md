# Cognitive
제2회 ETRI 휴먼이해 인공지능 논문경진대회 / 라이프로그와 수면의 정보 연관성을 활용한 수면의 질 예측 알고리즘


ETRI - 2022 휴먼이해 인공지능 경진대회
본 대회는 한국전자통신연구원(ETRI)이 주최하고 과학기술정보통신부와 국가과학기술연구회(NST)가 후원합니다

2. How To Use?
이 코드를 사용하는 방법을 다룹니다
순서와 지시를 그대로 따라 사용해주세요

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
