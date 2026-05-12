# [iM DiGital Banker] DL Competition

## Credit Score Classifiaction (신용점수 분류)

본 프로젝트는 신용 데이터에서 **신용점수**에 영향을 미치는 주요인을 분석하고, 딥러닝 분류 모델을 활용하여 모델 성능을 75% 이상으로 출력하는 것을 목표로 한다.

### 기간

2026년 5월 12일 (화)

### 데이터

data/train2.csv  
출처: [Credit score classification | Kaggle](https://www.kaggle.com/datasets/parisrohan/credit-score-classification)

> _Problem Statement_  
> You are working as a data scientist in a global finance company. Over the years, the company has collected basic bank details and gathered a lot of credit-related information. The management wants to build an intelligent system to segregate the people into credit score brackets to reduce the manual efforts.

### 디렉토리 구조

```
imbk_dl_comp/
├── data/
│ └── train2.csv
├── notebooks/
│ └── dl_comp.ipynb
├── .gitignore
├── datadictionary.txt
├── README.md
└── requirements.txt
```

### 라이브러리 리스트

#### Data Manipulation

![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

#### Deep Learning

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

#### Visualization

![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logoColor=white)

### 수행 과정

1. EDA 수행 및 데이터 전처리
   1. 결측치: **의사 결측치 존재 확인**  
       데이터의 성격상 결측 자체가 하나의 정보가 될 수도 있음  
       -> **"unknown"**으로 별도의 카테고리화
      - **Type_of_Loan**  
        No Data 11408  
        not specified 1408  
        컬럼 특성상 유니크 값이 많으므로 결측치를 다른 값으로 대체하지 않음. 대체값 넣어버리면 데이터 왜곡 생길 가능성 있기 때문.

      - **Payment_of_Min_Amount**  
        NM 12007  
        컬럼이 바이너리 값으로 이루어져므로 결측치를 다른 값으로 대체하지 않음.

   2. 타겟 분포 확인
      Standard에 고객이 몰려있고, Good 비율이 가장 낮음  
       -> 클래스 불균형 문제  
       -> 따라서 이후 모델 학습 시 다음을 고려하였음.
      - stratify split 적용
      - class weight 실험

       <img width="589" height="433" alt="image" src="https://github.com/user-attachments/assets/fb2e8b83-6d9e-46ee-9760-28c8a5a7e737" />

   3. 상관관계 히트맵상 Credit*Score와 유의한 관계가 있는 변수는  
      \_Credit_Mix, Num_of_Delayed_Payment, Changed_Credit_Limit, Num_Bank_Accounts, Payment_of_Min_Amount\_    
       <img width="1219" height="990" alt="image" src="https://github.com/user-attachments/assets/fffcb7ac-8839-4386-9153-16960b31c7e8" />  
       따라서 피처 셀렉션에 위 변수들을 먼저 고려했음.  
       (피처 셀렉션 결과 일부 변수만 포함하는 것보다 타겟 제외 모든 변수들로 학습했을 때 accuracy가 더 높게 출력됨.)

2. 딥러닝 기반 분류 모델 구축
   1. MLP, TabNet, TabTransformer 총 세 가지 모델 학습
   2. Accuracy 기준으로 성능 올리기 위한 튜닝 및 모델 업데이트

3. 데이터 전처리 및 Feature Engineering 수행
   1. 범주형 변수 Encoding 진행

      ```python
      for col in categorical_cols:
      # LabelEncoder는 순서가 있다고 착각할 수 있으므로 OrdinalEncoder 사용
      encoder = OrdinalEncoder(
          handle_unknown='use_encoded_value',
          unknown_value=-1
      )
      data[[col]] = encoder.fit_transform(data[[col]])
      data[col] = data[col].astype(int)   # float 타입으로 변환되었으므로 int로 변환해줌

      encoders[col] = encoder
      ```

   2. 연속형 변수 Scaling 적용
      ```python
      # 피처 스케일링 - 이상치가 많으므로 중앙값과 IQR로 스케일링하는 RobustScaler 사용
      scaler = RobustScaler()
      X = scaler.fit_transform(X)
      ```
   3. 학습 데이터와 테스트 데이터 분리 후 DataLoader 구성

      ```python
      # train, test split
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
      ```

      ```python
      # 데이터셋 생성
      train_dataset = CreditScoreDataset(X_train, y_train)
      test_dataset = CreditScoreDataset(X_test, y_test)

      # 데이터셋 생성, 배치 쪼개어서 전달 (미니배치)
      train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
      test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
      ```

4. 모델 학습 및 하이퍼파라미터 설정
   1. 최종 선정 모델: **TabTransformer**
   2. 주요 하이퍼파라미터:
      - dim = 32
      - depth = 6
      - heads = 8
      - dropout = 0.1
   3. 평가 데이터 Accuracy: **76%**
