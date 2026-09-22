### 3주차 퍼셉트론(Perceptron, 단 하나의 뉴런만을 사용) - 복습 ~ 본수업 진도
---

3.1. 학습

- (독립)변수가 하나 있으면 직선의 방정식이고, 
  
    $y = w_1*x_1 + b$ (선형 회귀의 문제)

- (독립)변수가 두 개 있으면 평면의 방정식이며,
  
    $y = w_1*x_1 + w_2*x_2 + b$

- (독립)변수가 세 개 있으면 초평면이다.
  
    $y = w_1*x_1 + w_2*x_2 + w_3*x_3 + \cdots + b$

- AI란 독립변수 X로 종속변수 y를 구하는 함수를 찾아주는 학습을 의미한다. 또한, 퍼셉트론이란 신경망이 스스로 가중치를 찾아 설정해주는 알고리즘을 말한다.
- 모델이 데이터에 맞도록 학습시키는 것을 적합(fitting)이라 한다.
  
    $\frac{\partial Loss(W,b)}{\partial W} = \frac{1}{n} \Sigma_{i=1}^n \ 2((Wx_i + b)) - y_i)(x_i) \frac{2}{n} \Sigma_{i=1}^n x_i((Wx_i+b)-y_i)$

    https://udlbook.github.io/udlfigures/ # 손실함수 오차 최소화 과정 시각화

    ![poster](<손실함수 시각화_학습_vdlbook.png>)

3.2. 퍼셉트론 실습 관련 리뷰

- scikit-learn tool 적용 과정

    ```
    # 퍼셉트론 (AND 게이트 연산)

    X = [[0,0], [0,1], [1,0], [1,1]]
    # 입력 데이터: AND 게이트의 4가지 입력 조합
    y = [0,0,0,1]
    # 정답 레이블: 둘 다 1일때만 1

    clf = Perceptron(tol = 1e-3, random_state = 0)
    # 퍼셉트론 객체 clf 생성
    # tol: 손실 개선폭이 tolerance 값보다 작으면 학습 조기 종료시키는 기준
    # random_state: 가중치 초기화, 난수 시드 고정

    clf.fit(X, y)
    # 입력 X와 정답 y로 가중치(W)와 편향(b)를 자동으로 학습

    print(clf.predict(X))
    # 학습된 모델로 같은 입력을 예측 
    # → [0 0 0 1] 출력
    ```
    ```
    # 퍼셉트론의 한계 (EXclusive OR 연산) → 다층 퍼셉트론으로 XOR 문제 해결 (line이 여러 개)
    
    y_hat = [0,1,0,1]
    # 정답 레이블: 변형

    clf = Perceptron(tol = 1e-3, random_state = 0)
    # 퍼셉트론 객체 생성

    clf.fit(X, y_hat)
    # 입력 X와 정답 y로 가중치(W)와 편향(b)를 자동으로 학습

    print(clf.predict(X))
    # 학습된 모델로 같은 입력을 예측 → 원하는 출력이 나오지 않음 즉, 학습되지 않음 → underfitting
    ```



---


### 4주차 다층 퍼셉트론(Multilayer Perceptron, MLP) - 다음 시간
---
