#!/usr/bin/env python3
"""
테스트용 샘플 데이터 생성 스크립트
"""

import pandas as pd
import numpy as np
import random

def generate_sample_data(n_samples=100):
    """샘플 데이터 생성"""
    
    # 랜덤 시드 설정
    np.random.seed(42)
    random.seed(42)
    
    # 샘플 데이터 생성
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'city': np.random.choice(['Seoul', 'Busan', 'Incheon', 'Daegu', 'Daejeon'], n_samples),
        'experience': np.random.randint(0, 30, n_samples),
        'satisfaction': np.random.uniform(1, 10, n_samples),
        'target': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    # income이 음수가 되지 않도록 조정
    data['income'] = np.abs(data['income'])
    
    # satisfaction을 1-10 범위로 조정
    data['satisfaction'] = np.clip(data['satisfaction'], 1, 10)
    
    df = pd.DataFrame(data)
    
    # CSV 파일로 저장
    df.to_csv('sample_data.csv', index=False)
    print(f"✅ {n_samples}개 샘플 데이터가 'sample_data.csv'로 저장되었습니다.")
    print(f"📊 데이터 형태: {df.shape}")
    print(f"🎯 타겟 분포: {df['target'].value_counts().to_dict()}")
    
    return df

if __name__ == "__main__":
    generate_sample_data(100)
