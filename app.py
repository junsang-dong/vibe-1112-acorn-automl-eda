import os
import pandas as pd
import numpy as np
import json
import base64
import io
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')  # GUI 없이 사용
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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 업로드 폴더 생성
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 정적 파일 서빙 설정
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

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
        
    def load_data(self, file_path):
        """CSV 파일 로드 및 기본 전처리"""
        try:
            self.df = pd.read_csv(file_path)
            
            # 100개 샘플링
            if len(self.df) > 100:
                self.df = self.df.sample(n=100, random_state=42).reset_index(drop=True)
            
            # 기본 전처리
            self.df = self.df.dropna()
            
            # 컬럼 타입 분류
            self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_cols = self.df.select_dtypes(include=['object', 'bool']).columns.tolist()
            
            # 타겟 변수 자동 감지 (Churn, Target, Label 등)
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
            print(f"데이터 로드 오류: {e}")
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
            print(f"전처리 오류: {e}")
            return False
    
    def get_basic_stats(self):
        """기본 통계 정보"""
        # float32를 float64로 변환하여 JSON 직렬화 문제 해결
        numeric_stats = {}
        if self.numeric_cols:
            desc_stats = self.df[self.numeric_cols].describe()
            for col in desc_stats.columns:
                numeric_stats[col] = {k: float(v) for k, v in desc_stats[col].to_dict().items()}
        
        stats = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'numeric_columns': self.numeric_cols,
            'categorical_columns': self.categorical_cols,
            'target_column': self.target_col,
            'missing_values': {k: int(v) for k, v in self.df.isnull().sum().to_dict().items()},
            'target_distribution': {k: int(v) for k, v in self.df[self.target_col].value_counts().to_dict().items()},
            'numeric_stats': numeric_stats
        }
        return stats
    
    def get_correlation_matrix(self):
        """상관관계 매트릭스"""
        if not self.numeric_cols:
            return None
        
        corr_matrix = self.df[self.numeric_cols + [self.target_col]].corr()
        # float32를 float64로 변환하여 JSON 직렬화 문제 해결
        return {k: {kk: float(vv) for kk, vv in v.items()} for k, v in corr_matrix.to_dict().items()}
    
    def create_correlation_heatmap(self):
        """상관관계 히트맵 생성"""
        if not self.numeric_cols:
            return None
        
        plt.figure(figsize=(12, 10))
        corr_matrix = self.df[self.numeric_cols + [self.target_col]].corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                   center=0, square=True, linewidths=0.5)
        plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 이미지를 base64로 변환
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return img_base64
    
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
                    'predictions': y_pred.tolist(),
                    'feature_importance': None
                }
                
                # 피처 중요도 (가능한 경우)
                if hasattr(model, 'feature_importances_'):
                    feature_importance = {k: float(v) for k, v in zip(X.columns, model.feature_importances_)}
                    results[name]['feature_importance'] = feature_importance
                
                # AUC (이진 분류인 경우)
                if y_pred_proba is not None and len(np.unique(y)) == 2:
                    try:
                        auc = roc_auc_score(y_test, y_pred_proba)
                        results[name]['auc'] = float(auc)
                    except:
                        pass
                
                self.models[name] = model
            
            return results
            
        except Exception as e:
            print(f"모델 학습 오류: {e}")
            return {}
    
    def create_feature_importance_plot(self, model_name):
        """피처 중요도 그래프 생성"""
        if model_name not in self.models or not hasattr(self.models[model_name], 'feature_importances_'):
            return None
        
        model = self.models[model_name]
        feature_importance = {k: float(v) for k, v in zip(self.df.drop(columns=[self.target_col]).columns, model.feature_importances_)}
        
        # 상위 10개 피처만 표시
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        features, importance = zip(*sorted_features)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(features)), importance)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'{model_name} - Feature Importance (Top 10)', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        # 이미지를 base64로 변환
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return img_base64
    
    def create_target_distribution_plot(self):
        """타겟 변수 분포 그래프"""
        plt.figure(figsize=(10, 6))
        
        target_counts = self.df[self.target_col].value_counts()
        
        # 막대 그래프
        plt.subplot(1, 2, 1)
        target_counts.plot(kind='bar', color=['#2ecc71', '#e74c3c'])
        plt.title('Target Distribution (Count)', fontweight='bold')
        plt.xlabel('Target Value')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # 파이 차트
        plt.subplot(1, 2, 2)
        plt.pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%', 
                colors=['#2ecc71', '#e74c3c'], startangle=90)
        plt.title('Target Distribution (Percentage)', fontweight='bold')
        
        plt.tight_layout()
        
        # 이미지를 base64로 변환
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return img_base64

# 전역 분석기 인스턴스
analyzer = DataAnalyzer()

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """CSV 파일 업로드 및 분석"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
        
        if file and file.filename.lower().endswith('.csv'):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # 데이터 로드
            if not analyzer.load_data(file_path):
                return jsonify({'error': '데이터 로드에 실패했습니다.'}), 400
            
            # 데이터 전처리
            if not analyzer.preprocess_data():
                return jsonify({'error': '데이터 전처리에 실패했습니다.'}), 400
            
            # 기본 통계
            basic_stats = analyzer.get_basic_stats()
            
            # 상관관계 매트릭스
            correlation_matrix = analyzer.get_correlation_matrix()
            
            # 시각화
            correlation_heatmap = analyzer.create_correlation_heatmap()
            target_distribution = analyzer.create_target_distribution_plot()
            
            # 모델 학습
            model_results = analyzer.train_models()
            
            # 피처 중요도 그래프
            feature_importance_plots = {}
            for model_name in model_results.keys():
                plot = analyzer.create_feature_importance_plot(model_name)
                if plot:
                    feature_importance_plots[model_name] = plot
            
            # 결과 정리
            result = {
                'success': True,
                'basic_stats': basic_stats,
                'correlation_matrix': correlation_matrix,
                'correlation_heatmap': correlation_heatmap,
                'target_distribution': target_distribution,
                'model_results': model_results,
                'feature_importance_plots': feature_importance_plots
            }
            
            # 임시 파일 삭제
            os.remove(file_path)
            
            return jsonify(result)
        
        else:
            return jsonify({'error': 'CSV 파일만 업로드 가능합니다.'}), 400
            
    except Exception as e:
        return jsonify({'error': f'서버 오류: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """새로운 데이터 예측"""
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({'error': '예측할 데이터가 없습니다.'}), 400
        
        features = data['features']
        model_name = data.get('model', 'Random Forest')
        
        if model_name not in analyzer.models:
            return jsonify({'error': '선택된 모델이 없습니다.'}), 400
        
        # 데이터프레임으로 변환
        df_predict = pd.DataFrame([features])
        
        # 전처리 (범주형 변수 인코딩)
        for col in analyzer.categorical_cols:
            if col != analyzer.target_col and col in df_predict.columns:
                if col in analyzer.label_encoders:
                    try:
                        df_predict[col] = analyzer.label_encoders[col].transform(df_predict[col].astype(str))
                    except:
                        # 새로운 카테고리인 경우 가장 빈번한 값으로 대체
                        df_predict[col] = 0
        
        # 예측
        model = analyzer.models[model_name]
        
        if model_name == 'XGBoost':
            prediction = model.predict(df_predict)[0]
            prediction_proba = model.predict_proba(df_predict)[0] if hasattr(model, 'predict_proba') else None
        else:
            df_predict_scaled = analyzer.scaler.transform(df_predict)
            prediction = model.predict(df_predict_scaled)[0]
            prediction_proba = model.predict_proba(df_predict_scaled)[0] if hasattr(model, 'predict_proba') else None
        
        result = {
            'prediction': int(prediction),
            'prediction_proba': [float(p) for p in prediction_proba.tolist()] if prediction_proba is not None else None
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'예측 오류: {str(e)}'}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 AutoML 데이터 분석 대시보드 시작")
    print("=" * 60)
    print("📊 웹 브라우저에서 http://localhost:8080 접속")
    print("📁 CSV 파일을 업로드하여 자동 분석 시작")
    print("=" * 60)
    app.run(debug=True, host='127.0.0.1', port=8080)
