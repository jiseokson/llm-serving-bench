## Overview
기초적인 LLM Serving 스크립트를 직접 구현하고, 상용 LLM Serving 시스템인 vLLM을 함께 구성하여 벤치마킹 도구를 개발하고 성능을 측정,비교 분석합니다.

- 기초적인 Serving 스크립트 작성
- OpenAI API 스타일의 요청 처리 (completion/chat-completion API 호환)
- vLLM 배포 및 설정
- Benchmark 툴 구현과 성능 측정 및 비교, 프롬프트 전처리
 

## Environment Setup

이 프로젝트에 포함된 Serving 스크립트와 벤치마크를 실행하려면 프로젝트 루트에서 아래 명령어로 필수 패키지를 설치해야 합니다:

```pip install
pip install -r requirements.txt
```

[vLLM](https://github.com/vllm-project/vllm)의 설치와 실행 환경 구성은 [공식 문서](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)를 참고했습니다.

> 본 실험은 [RunPod](https://www.runpod.io/)의 NVIDIA A100 80GB GPU 인스턴스 환경에서 수행되었으며, 동일하거나 유사한 환경에서 실행할 것을 권장합니다.

## How to Run

### 1. Serving Script

FastAPI 기반 서버는 다음과 같이 nohup을 사용하여 백그라운드로 실행할 수 있습니다:

```bash
nohup uvicorn app:app --host 0.0.0.0 --port 8000 &
```

### 2. Benchmark Script

모델 종류(--model), 태스크 종류(--task), API 요청 모드(--mode), 반복 횟수(--repeat)를 지정하여 실행할 수 있으며, 결과는 .csv 파일로 저장됩니다:

```bash
# Latency 계열 지표 측정, 10회 반복
python experiment.py --model gpt2 --task chat --mode stream --repeat 10

# Throughput 계열 지표 측정, 10회 반복, 8 동시 요청
python experiment.py --model gpt2 --task qa --mode stream --repeat 10 --throughput-only --parallel 8
```

## Dataset

## Code Overview

## Results

## Additional Resources
[Google slide - 중간 발표](https://docs.google.com/presentation/d/1dIXP-vJu0QszjQoBENqWEVvm_JJ5_WqBpWjJHkA5G68/edit?usp=sharing)
