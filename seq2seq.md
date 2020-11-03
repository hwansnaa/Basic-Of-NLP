# seq2seq
+ seq2seq
  + 입력 텍스트에 대해 학습된 모델에 따라 결과값을 출력해주는 모델
  + ex) input : i am a student -> output : 나는 학생이다.
  + RNN기반 seq2seq의 문제점
    + 하나의 고정된 크기의 벡터에 모든 정보를 압축해 정보 손실이 발생
    + Vanishing Gradient Problem
  + 따라서, Attention Mechanism을 통해 문제 해결

+ 어텐션(Attention)
  + 디코더에서 출력단어를 예측하는 매 시점마다 인코더의 전체 입력 문장을 다시한번 참고한다. 단, 예측해야할 단어와 연관이 있는 부분을 집중해서 참고한다.
  + 어텐션 함수
    + 공식 : Attention(Query, Key, Value) = Attention Value
      + Query : t 시점의 디코더 셀에서의 은닉 상태
      + Key : 모든 시점의 인코더 셀의 은닉 상태들
      + Value : 모든 시점의 인코더 셀의 은닉 상태들
