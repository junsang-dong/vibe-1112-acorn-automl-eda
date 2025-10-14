# AutoML 데이터 분석 대시보드

CSV 파일을 업로드하면 자동으로 데이터 분석과 머신러닝 모델 학습을 수행하는 웹 애플리케이션입니다.

## 🚀 주요 기능

- **자동 데이터 분석**: CSV 파일 업로드 시 100개 샘플을 자동으로 분석
- **EDA (탐색적 데이터 분석)**: 기본 통계, 상관관계 히트맵, 타겟 분포 시각화
- **자동 모델 선택**: Decision Tree, Random Forest, XGBoost 모델 자동 학습 및 비교
- **피처 중요도 분석**: 각 모델의 피처 중요도 시각화
- **실시간 예측**: 학습된 모델을 사용한 새로운 데이터 예측
- **반응형 대시보드**: 모바일 친화적인 현대적인 UI
- **드래그 앤 드롭**: 파일 업로드를 위한 직관적인 인터페이스
- **자동 타겟 감지**: Churn, Target, Label 등 타겟 변수 자동 인식

## 📦 설치 및 실행

### 🌐 온라인 버전 (Streamlit Cloud)
**즉시 사용 가능**: [AutoML 데이터 분석 대시보드](https://share.streamlit.io/junsang-dong/vibe-1112-acorn-automl-eda/main/streamlit_app.py)

### 💻 로컬 실행

#### Flask 버전
```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 애플리케이션 실행
python3 app.py

# 3. 웹 브라우저에서 접속
# http://localhost:8080
```

#### Streamlit 버전
```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. Streamlit 앱 실행
streamlit run streamlit_app.py

# 3. 웹 브라우저에서 접속
# http://localhost:8501
```

> **참고**: macOS에서 포트 5000이 사용 중인 경우 자동으로 8080 포트로 실행됩니다.

## 📋 사용 방법

1. **CSV 파일 업로드**
   - 웹 페이지에서 CSV 파일을 선택하거나 드래그 앤 드롭
   - 자동으로 100개 샘플 추출 및 분석 시작
   - 파일 크기 제한: 최대 16MB

2. **분석 결과 확인**
   - 데이터 기본 정보 (행/열 수, 변수 타입 등)
   - 타겟 변수 분포 시각화 (막대 그래프 + 파이 차트)
   - 상관관계 히트맵 (수치형 변수 간 상관관계)
   - 모델 성능 비교 (정확도, 교차검증 점수, AUC)
   - 피처 중요도 그래프 (상위 10개 피처)

3. **새로운 데이터 예측**
   - 원하는 모델 선택 (Decision Tree, Random Forest, XGBoost)
   - 새로운 데이터 입력 (수치형/범주형 변수)
   - 예측 결과 확인 (예측값 + 확률)

## 📊 지원하는 데이터 형식

- **CSV 파일**: 콤마로 구분된 값
- **타겟 변수**: 자동 감지 (Churn, Target, Label, Class 등)
- **변수 타입**: 수치형, 범주형 자동 분류
- **전처리**: 결측치 제거, 범주형 변수 인코딩
- **샘플링**: 100개 이상 데이터는 100개로 자동 샘플링

## 🤖 모델 정보

### Decision Tree
- 해석 가능성이 높은 트리 기반 모델
- 과적합 방지를 위한 max_depth=5 설정
- 교차검증을 통한 성능 평가

### Random Forest
- 앙상블 기반 모델로 안정적인 성능
- n_estimators=100, max_depth=5 설정
- 피처 중요도 제공

### XGBoost
- 그래디언트 부스팅 기반 고성능 모델
- max_depth=3, n_estimators=100 설정
- 이진 분류 시 AUC 점수 제공

## 🛠️ 기술 스택

### 백엔드
- **Flask**: 웹 프레임워크 (로컬 버전)
- **Streamlit**: 웹 앱 프레임워크 (클라우드 버전)
- **pandas**: 데이터 처리 및 분석
- **scikit-learn**: 머신러닝 모델
- **XGBoost**: 그래디언트 부스팅
- **matplotlib/seaborn**: 시각화
- **numpy**: 수치 계산
- **scipy**: 통계 분석

### 프론트엔드
- **HTML5/CSS3**: 반응형 웹 디자인 (Flask 버전)
- **JavaScript**: 동적 인터랙션 (Flask 버전)
- **Bootstrap 5**: UI 프레임워크 (Flask 버전)
- **Font Awesome**: 아이콘 (Flask 버전)
- **Streamlit Components**: UI 컴포넌트 (Streamlit 버전)

### 배포
- **GitHub Pages**: 정적 사이트 호스팅
- **Streamlit Cloud**: 클라우드 앱 호스팅
- **GitHub Actions**: CI/CD 파이프라인

## 📁 파일 구조

```
vibe-1112-acorn-automl-eda/
├── app.py                 # Flask 애플리케이션 메인 파일
├── streamlit_app.py      # Streamlit 애플리케이션 메인 파일
├── requirements.txt       # Python 의존성
├── README.md             # 프로젝트 설명서
├── run.sh                # 실행 스크립트
├── test_data.py          # 테스트 데이터 생성 스크립트
├── templates/
│   └── index.html        # Flask 메인 웹 페이지
├── .streamlit/
│   └── config.toml       # Streamlit 설정 파일
├── .github/
│   └── workflows/
│       └── deploy.yml    # GitHub Actions 배포 설정
├── uploads/              # 업로드된 파일 임시 저장소
├── churn.csv            # 예시 데이터 파일 (통신사 이탈)
├── sample_data.csv      # 테스트 데이터 파일
└── ML_churn_eda.ipynb   # 참고용 EDA 노트북
```

## ⚠️ 주의사항

- 업로드 가능한 파일 크기: 최대 16MB
- 샘플링: 데이터가 100개 이상인 경우 100개로 샘플링
- 임시 파일: 분석 완료 후 자동 삭제
- 브라우저 호환성: Chrome, Firefox, Safari, Edge 지원
- 포트 충돌: macOS에서 포트 5000 사용 시 자동으로 8080 포트 사용

## 📊 예시 데이터

프로젝트에 포함된 `churn.csv` 파일은 통신사 고객 이탈 예측 데이터로, 다음과 같은 컬럼을 포함합니다:

- **State**: 주(State)
- **Account_Length**: 계정 기간
- **Intl_Plan**: 국제 플랜 가입 여부
- **Vmail_Plan**: 음성메일 플랜 가입 여부
- **Day_Mins/Eve_Mins/Night_Mins**: 시간대별 통화 시간
- **CustServ_Calls**: 고객 서비스 통화 횟수
- **Churn**: 이탈 여부 (타겟 변수)

## 🔧 문제 해결

### 포트 충돌 문제
- **문제**: macOS Control Center가 포트 5000을 사용 중
- **해결**: 포트를 8080으로 변경하여 해결

### HTTP 403 에러
- **문제**: 업로드 폴더 권한 및 정적 파일 서빙 설정 부족
- **해결**: 
  - 업로드 폴더 권한을 755로 설정
  - Flask에 정적 파일 서빙 라우트 추가
  - 호스트를 `127.0.0.1`로 변경하여 보안 강화

### JSON 직렬화 오류
- **문제**: `float32` 타입이 JSON으로 직렬화되지 않는 문제
- **해결**: 모든 수치 데이터를 `float()` 또는 `int()`로 명시적 변환

## 🚀 개발 과정

1. **프로젝트 구조 분석**: 기존 EDA 노트북과 데이터 파일 확인
2. **Flask 백엔드 구현**: CSV 업로드, 데이터 분석, 모델 학습 기능
3. **HTML/CSS/JS 프론트엔드**: 반응형 대시보드 UI 구현
4. **데이터 분석 로직**: EDA, 모델 선택, 예측 기능 구현
5. **시각화 컴포넌트**: 차트, 히트맵, 그래프 구현
6. **오류 수정**: 포트 충돌, 권한 문제, JSON 직렬화 문제 해결
7. **테스트 및 검증**: 샘플 데이터로 전체 기능 테스트

## 📝 라이선스

이 프로젝트는 교육 및 연구 목적으로 자유롭게 사용할 수 있습니다.
