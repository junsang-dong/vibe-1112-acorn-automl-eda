# Streamlit Cloud 배포 가이드

## 🚀 Streamlit Cloud에 앱 배포하기

### 1. Streamlit Cloud 접속
1. [Streamlit Cloud](https://share.streamlit.io/)에 접속
2. GitHub 계정으로 로그인

### 2. 새 앱 배포
1. "New app" 버튼 클릭
2. GitHub 리포지토리 선택: `junsang-dong/vibe-1112-acorn-automl-eda`
3. 브랜치 선택: `main`
4. 메인 파일 경로: `streamlit_app.py`
5. "Deploy!" 버튼 클릭

### 3. 배포 완료
- 배포가 완료되면 자동으로 URL이 생성됩니다
- 예상 URL: `https://share.streamlit.io/junsang-dong/vibe-1112-acorn-automl-eda/main/streamlit_app.py`

### 4. 앱 사용법
1. 웹 브라우저에서 배포된 URL 접속
2. 사이드바에서 CSV 파일 업로드
3. 자동으로 데이터 분석 및 모델 학습 수행
4. 결과 확인 및 새로운 데이터 예측

## 🔧 로컬에서 Streamlit 실행하기

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. Streamlit 앱 실행
streamlit run streamlit_app.py

# 3. 브라우저에서 접속
# http://localhost:8501
```

## 📊 지원하는 기능

### Streamlit 버전
- ✅ CSV 파일 업로드 (드래그 앤 드롭)
- ✅ 자동 데이터 분석 (100개 샘플링)
- ✅ 기본 통계 정보 표시
- ✅ 타겟 변수 분포 시각화
- ✅ 상관관계 히트맵
- ✅ 3가지 머신러닝 모델 학습
- ✅ 모델 성능 비교
- ✅ 피처 중요도 시각화
- ✅ 새로운 데이터 예측
- ✅ 반응형 UI

### Flask 버전 (로컬)
- ✅ 모든 Streamlit 기능
- ✅ 추가적인 시각화 옵션
- ✅ 더 세밀한 UI 커스터마이징

## 🎯 사용 예시

### 1. 통신사 고객 이탈 데이터 (churn.csv)
- State, Account_Length, Intl_Plan, Vmail_Plan
- Day_Mins, Eve_Mins, Night_Mins, CustServ_Calls
- Churn (타겟 변수)

### 2. 테스트 데이터 (sample_data.csv)
- age, income, education, city
- experience, satisfaction
- target (타겟 변수)

## 🔍 문제 해결

### 일반적인 문제
1. **파일 업로드 실패**: CSV 파일 형식 확인
2. **모델 학습 실패**: 데이터에 결측치가 있는지 확인
3. **예측 실패**: 입력 데이터 형식 확인

### 지원 문의
- GitHub Issues: [이슈 등록](https://github.com/junsang-dong/vibe-1112-acorn-automl-eda/issues)
- 이메일: junsang.dong@example.com
