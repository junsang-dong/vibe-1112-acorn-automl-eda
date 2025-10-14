import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import xgboost as xgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="AutoML 데이터 분석 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class DataAnalyzer:
    def __init__(self):
        self.df = None
        self.numeric_cols = []
        self.categorical_cols = []
        self.target_col = None
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, uploaded_file):
        """CSV 파일 로드 및 기본 전처리"""
        try:
            self.df = pd.read_csv(uploaded_file)
            
            # 100개 샘플링
            if len(self.df) > 100:
                self.df = self.df.sample(n=100, random_state=42).reset_index(drop=True)
            
            # 기본 전처리
            self.df = self.df.dropna()
            
            # 컬럼 타입 분류
            self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_cols = self.df.select_dtypes(include=['object', 'bool']).columns.tolist()
            
            # 타겟 변수 자동 감지
            target_candidates = ['Churn', 'Target', 'Label', 'Class', 'churn', 'target', 'label', 'class']
            for col in target_candidates:
                if col in self.df.columns:
                    self.target_col = col
                    break
            
            # 타겟 변수가 없으면 마지막 컬럼을 타겟으로 사용
            if self.target_col is None:
                self.target_col = self.df.columns[-1]
            
            return True
        except Exception as e:
            st.error(f"데이터 로드 오류: {e}")
            return False
    
    def preprocess_data(self):
        """데이터 전처리"""
        try:
            # 범주형 변수 인코딩
            for col in self.categorical_cols:
                if col != self.target_col:
                    le = LabelEncoder()
                    self.df[col] = le.fit_transform(self.df[col].astype(str))
                    self.label_encoders[col] = le
            
            # 타겟 변수 인코딩
            if self.df[self.target_col].dtype == 'object':
                le_target = LabelEncoder()
                self.df[self.target_col] = le_target.fit_transform(self.df[self.target_col])
                self.label_encoders[self.target_col] = le_target
            
            return True
        except Exception as e:
            st.error(f"전처리 오류: {e}")
            return False
    
    def get_basic_stats(self):
        """기본 통계 정보"""
        stats = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'numeric_columns': self.numeric_cols,
            'categorical_columns': self.categorical_cols,
            'target_column': self.target_col,
            'missing_values': self.df.isnull().sum().to_dict(),
            'target_distribution': self.df[self.target_col].value_counts().to_dict()
        }
        return stats
    
    def train_models(self):
        """모델 학습"""
        try:
            # 피처와 타겟 분리
            X = self.df.drop(columns=[self.target_col])
            y = self.df[self.target_col]
            
            # 데이터 분할
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # 스케일링
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # 모델 정의
            models = {
                'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5),
                'XGBoost': xgb.XGBClassifier(random_state=42, max_depth=3, n_estimators=100)
            }
            
            results = {}
            
            for name, model in models.items():
                # 모델 학습
                if name == 'XGBoost':
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1] if len(np.unique(y)) == 2 else None
                else:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if len(np.unique(y)) == 2 else None
                
                # 성능 평가
                accuracy = accuracy_score(y_test, y_pred)
                
                # 교차 검증
                if name == 'XGBoost':
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                else:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
                
                results[name] = {
                    'accuracy': float(accuracy),
                    'cv_mean': float(cv_scores.mean()),
                    'cv_std': float(cv_scores.std()),
                    'feature_importance': None
                }
                
                # 피처 중요도
                if hasattr(model, 'feature_importances_'):
                    feature_importance = {k: float(v) for k, v in zip(X.columns, model.feature_importances_)}
                    results[name]['feature_importance'] = feature_importance
                
                # AUC
                if y_pred_proba is not None and len(np.unique(y)) == 2:
                    try:
                        auc = roc_auc_score(y_test, y_pred_proba)
                        results[name]['auc'] = float(auc)
                    except:
                        pass
                
                self.models[name] = model
            
            return results
            
        except Exception as e:
            st.error(f"모델 학습 오류: {e}")
            return {}

def main():
    st.title("📊 AutoML 데이터 분석 대시보드")
    st.markdown("CSV 파일을 업로드하면 자동으로 데이터 분석과 머신러닝 모델 학습을 수행합니다.")
    
    # 사이드바
    st.sidebar.title("📁 파일 업로드")
    uploaded_file = st.sidebar.file_uploader(
        "CSV 파일을 선택하세요",
        type=['csv'],
        help="최대 16MB까지 업로드 가능합니다."
    )
    
    if uploaded_file is not None:
        # 데이터 로드
        analyzer = DataAnalyzer()
        
        with st.spinner("데이터를 로드하고 분석 중..."):
            if analyzer.load_data(uploaded_file):
                if analyzer.preprocess_data():
                    st.success("✅ 데이터 로드 및 전처리 완료!")
                    
                    # 기본 통계
                    stats = analyzer.get_basic_stats()
                    
                    # 메인 대시보드
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("총 행 수", f"{stats['shape'][0]:,}")
                    with col2:
                        st.metric("총 열 수", stats['shape'][1])
                    with col3:
                        st.metric("수치형 변수", len(stats['numeric_columns']))
                    with col4:
                        st.metric("범주형 변수", len(stats['categorical_columns']))
                    
                    # 타겟 분포
                    st.subheader("🎯 타겟 변수 분포")
                    target_dist = stats['target_distribution']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 막대 그래프
                        fig, ax = plt.subplots(figsize=(8, 6))
                        bars = ax.bar(target_dist.keys(), target_dist.values(), 
                                    color=['#2ecc71', '#e74c3c'])
                        ax.set_title('타겟 분포 (건수)', fontweight='bold')
                        ax.set_xlabel('타겟 값')
                        ax.set_ylabel('빈도')
                        
                        # 값 표시
                        for bar, value in zip(bars, target_dist.values()):
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                                   str(value), ha='center', fontweight='bold')
                        
                        st.pyplot(fig)
                    
                    with col2:
                        # 파이 차트
                        fig, ax = plt.subplots(figsize=(8, 6))
                        colors = ['#2ecc71', '#e74c3c']
                        wedges, texts, autotexts = ax.pie(target_dist.values(), 
                                                        labels=target_dist.keys(), 
                                                        autopct='%1.1f%%',
                                                        colors=colors, startangle=90)
                        ax.set_title('타겟 분포 (비율)', fontweight='bold')
                        st.pyplot(fig)
                    
                    # 상관관계 히트맵
                    if len(stats['numeric_columns']) > 0:
                        st.subheader("🔥 상관관계 히트맵")
                        
                        corr_matrix = analyzer.df[stats['numeric_columns'] + [analyzer.target_col]].corr()
                        
                        fig, ax = plt.subplots(figsize=(12, 10))
                        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r',
                                   center=0, square=True, linewidths=0.5, ax=ax)
                        ax.set_title('변수 간 상관관계 히트맵', fontsize=16, fontweight='bold')
                        st.pyplot(fig)
                    
                    # 모델 학습
                    st.subheader("🤖 머신러닝 모델 학습")
                    
                    with st.spinner("모델을 학습하고 있습니다..."):
                        model_results = analyzer.train_models()
                    
                    if model_results:
                        st.success("✅ 모델 학습 완료!")
                        
                        # 모델 성능 비교
                        st.subheader("📈 모델 성능 비교")
                        
                        # 성능 테이블
                        performance_data = []
                        for model_name, result in model_results.items():
                            performance_data.append({
                                '모델': model_name,
                                '정확도': f"{result['accuracy']*100:.2f}%",
                                '교차검증 평균': f"{result['cv_mean']*100:.2f}%",
                                '교차검증 표준편차': f"{result['cv_std']*100:.2f}%",
                                'AUC': f"{result.get('auc', 'N/A'):.3f}" if result.get('auc') else 'N/A'
                            })
                        
                        performance_df = pd.DataFrame(performance_data)
                        st.dataframe(performance_df, use_container_width=True)
                        
                        # 최고 성능 모델 찾기
                        best_model = max(model_results.keys(), 
                                       key=lambda x: model_results[x]['accuracy'])
                        
                        st.success(f"🏆 최고 성능 모델: **{best_model}** "
                                 f"(정확도: {model_results[best_model]['accuracy']*100:.2f}%)")
                        
                        # 피처 중요도
                        st.subheader("⭐ 피처 중요도")
                        
                        for model_name, result in model_results.items():
                            if result['feature_importance']:
                                st.write(f"**{model_name}**")
                                
                                # 상위 10개 피처
                                sorted_features = sorted(result['feature_importance'].items(), 
                                                       key=lambda x: x[1], reverse=True)[:10]
                                
                                fig, ax = plt.subplots(figsize=(10, 6))
                                features, importance = zip(*sorted_features)
                                bars = ax.barh(range(len(features)), importance)
                                ax.set_yticks(range(len(features)))
                                ax.set_yticklabels(features)
                                ax.set_xlabel('피처 중요도')
                                ax.set_title(f'{model_name} - 피처 중요도 (상위 10개)', 
                                           fontweight='bold')
                                ax.invert_yaxis()
                                
                                st.pyplot(fig)
                    
                    # 예측 섹션
                    st.subheader("🔮 새로운 데이터 예측")
                    
                    if model_results:
                        # 모델 선택
                        selected_model = st.selectbox(
                            "예측에 사용할 모델을 선택하세요:",
                            list(model_results.keys())
                        )
                        
                        # 입력 폼
                        st.write("새로운 데이터를 입력하세요:")
                        
                        input_data = {}
                        
                        # 수치형 변수 입력
                        for col in stats['numeric_columns']:
                            if col != analyzer.target_col:
                                input_data[col] = st.number_input(
                                    f"{col}",
                                    value=0.0,
                                    step=0.1
                                )
                        
                        # 범주형 변수 입력
                        for col in stats['categorical_columns']:
                            if col != analyzer.target_col:
                                input_data[col] = st.text_input(f"{col}")
                        
                        # 예측 버튼
                        if st.button("예측하기"):
                            try:
                                # 데이터프레임으로 변환
                                df_predict = pd.DataFrame([input_data])
                                
                                # 전처리
                                for col in stats['categorical_columns']:
                                    if col != analyzer.target_col and col in df_predict.columns:
                                        if col in analyzer.label_encoders:
                                            try:
                                                df_predict[col] = analyzer.label_encoders[col].transform(df_predict[col].astype(str))
                                            except:
                                                df_predict[col] = 0
                                
                                # 예측
                                model = analyzer.models[selected_model]
                                
                                if selected_model == 'XGBoost':
                                    prediction = model.predict(df_predict)[0]
                                    prediction_proba = model.predict_proba(df_predict)[0] if hasattr(model, 'predict_proba') else None
                                else:
                                    df_predict_scaled = analyzer.scaler.transform(df_predict)
                                    prediction = model.predict(df_predict_scaled)[0]
                                    prediction_proba = model.predict_proba(df_predict_scaled)[0] if hasattr(model, 'predict_proba') else None
                                
                                # 결과 표시
                                st.success(f"🎯 예측 결과: **{prediction}**")
                                
                                if prediction_proba is not None:
                                    st.write("**클래스별 확률:**")
                                    for i, prob in enumerate(prediction_proba):
                                        st.write(f"클래스 {i}: {prob*100:.2f}%")
                                
                            except Exception as e:
                                st.error(f"예측 중 오류가 발생했습니다: {e}")
                
                else:
                    st.error("데이터 전처리에 실패했습니다.")
            else:
                st.error("데이터 로드에 실패했습니다.")
    
    else:
        # 기본 화면
        st.info("👈 사이드바에서 CSV 파일을 업로드하여 분석을 시작하세요.")
        
        # 예시 데이터 다운로드
        st.subheader("📊 예시 데이터")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**통신사 고객 이탈 데이터 (churn.csv)**")
            st.write("- State: 주(State)")
            st.write("- Account_Length: 계정 기간")
            st.write("- Intl_Plan: 국제 플랜 가입 여부")
            st.write("- Vmail_Plan: 음성메일 플랜 가입 여부")
            st.write("- Day_Mins/Eve_Mins/Night_Mins: 시간대별 통화 시간")
            st.write("- CustServ_Calls: 고객 서비스 통화 횟수")
            st.write("- Churn: 이탈 여부 (타겟 변수)")
        
        with col2:
            st.write("**테스트 데이터 (sample_data.csv)**")
            st.write("- age: 나이")
            st.write("- income: 소득")
            st.write("- education: 교육 수준")
            st.write("- city: 도시")
            st.write("- experience: 경력")
            st.write("- satisfaction: 만족도")
            st.write("- target: 타겟 변수")

if __name__ == "__main__":
    main()
