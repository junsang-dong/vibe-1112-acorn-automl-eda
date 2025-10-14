#!/bin/bash

echo "🚀 AutoML 데이터 분석 대시보드 시작"
echo "=================================="

# Python 가상환경 확인 및 생성
if [ ! -d "venv" ]; then
    echo "📦 가상환경 생성 중..."
    python3 -m venv venv
fi

# 가상환경 활성화
echo "🔧 가상환경 활성화 중..."
source venv/bin/activate

# 의존성 설치
echo "📚 의존성 패키지 설치 중..."
pip install -r requirements.txt

# 애플리케이션 실행
echo "🌟 웹 애플리케이션 시작 중..."
echo "📊 브라우저에서 http://localhost:8080 접속하세요"
echo "=================================="

python app.py
