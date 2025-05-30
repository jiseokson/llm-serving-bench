## Overview
기초적인 **LLM Serving** 스크립트를 직접 구현하고, 상용 **LLM Serving** 시스템인 vLLM을 함께 구성하여 벤치마킹 도구를 개발하고 성능을 측정, 비교 분석합니다.

- 기초적인 **Serving** 스크립트 작성
- **OpenAI API** 스타일의 요청 처리 (**completion**/**chat**-**completion** **API** 호환)
- **vLLM** 배포 및 설정
- **Benchmark** 툴 구현과 성능 측정 및 비교, 프롬프트 전처리
 
## Environment Setup

이 프로젝트에 포함된 **Serving** 스크립트와 벤치마크를 실행하려면 프로젝트 루트에서 아래 명령어로 필수 패키지를 설치해야 합니다:

```pip install
pip install -r requirements.txt
```

[**vLLM**](https://github.com/vllm-project/vllm)의 설치와 실행 환경 구성은 [공식 문서](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)를 참고했습니다.

> 본 실험은 [**RunPod**](https://www.runpod.io/)의 **NVIDIA A100 80GB GPU** 인스턴스 환경에서 수행되었으며, 동일하거나 유사한 환경에서 실행할 것을 권장합니다.

## How to Run

### 1. Serving Script

**FastAPI** 기반 서버는 다음과 같이 nohup을 사용하여 백그라운드로 실행할 수 있습니다:

```bash
nohup uvicorn app:app --host 0.0.0.0 --port 8000 &
```

서버는 기본적으로 포트 8000에서 실행합니다. 이는 벤치마킹 툴이 in-host HTTP 방식으로 해당 포트(8000)를 통해 요청을 보내도록 구성되어 있기 때문입니다. 다른 포트를 사용할 경우 `benchmarks/**experiment.py**` 내부의 `host` 값을 함께 수정해야 합니다.

```python
host = "http://localhost:8000"
```

### 2. Benchmark Script

모델 종류(--model), 태스크 종류(--task), API 요청 모드(--mode), 반복 횟수(--repeat)를 지정하여 실행할 수 있으며, 결과는 .csv 파일로 저장됩니다:

```bash
# Latency 계열 지표 측정, 10회 반복
python **experiment.py --model gpt2 --task **chat** --mode stream --repeat 10

# Throughput 계열 지표 측정, 10회 반복, 8 동시 요청
python experiment.py --model gpt2 --task qa --mode stream --repeat 10 --throughput-only --parallel 8
```

Benchmark 툴은 다음과 같은 지표들을 수집합니다.
Sync 방식과 Streaming 방식은 생성 요청 이후 응답을 수신하는 방식에 차이가 있습니다.
- **Sync** 방식은 전체 문장이 완성된 뒤, 하나의 `JSON` 응답으로 결과를 받아오는 동기적 처리 방식입니다.
- **Streaming** 방식은 생성된 토큰을 실시간으로 **SSE**(Server Sent Event) 형태로 스트리밍하여 수신합니다.

보다 자세한 사용 옵션은 `python experiment.py --help` 명령어를 통해 확인할 수 있습니다.

| 지표 | Sync 방식 | Streaming 방식 | 비교 / 측정 방식 요약 |
|-|-|-|-|
| **Total Latency** | 전체 응답이 완전히 도착한 시점까지 측정 | 마지막 토큰이 수신된 시점까지 측정 | `start_time ~ end_time` |
| **Prompt-to-First-Token**<br>(**P2FT**) | 측정 불가 (토큰 단위 응답 아님) | 첫 토큰 수신 시점까지 측정 가능 | `start_time ~ 첫 **SSE** 수신 시점` |
| **Token Generation Speed**<br>(tok/s) | 생성된 토큰 수를 총 응답 시간으로 나눔 | 생성된 토큰 수를 마지막 토큰 수신 시점까지의 시간으로 나눔 | 토큰 수 카운트 + 응답 시간 측정 필요 |
| **Request Throughput**<br>(req/s) | 요청 수를 총 시간으로 나눔 | 요청 수를 마지막 토큰 수신 시점까지의 시간으로 나눔 | `httpx + asyncio` 기반 동시 요청 |
| **Token Throughput**<br>(tok/s) | 생성된 토큰 수를 총 시간으로 나눔 | 생성된 토큰 수를 마지막 토큰 수신 시점까지의 시간으로 나눔 | 위와 동일 |
| **GPU Memory Usage** | 동일 방식으로 측정 | 동일 방식으로 측정 | `pynvml` 사용 |
| **Host Memory Usage** | 동일 방식으로 측정 | 동일 방식으로 측정 | `psutil` 사용 |

## Dataset

많이 활용되는 LLM의 대표적인 4가지 태스크를 선정하고 각 태스크에 적합한 데이터셋을 기반으로 다양한 입출력 길이와 생성 방식에 따라 **Serving** System의 성능을 평가합니다.
각 데이터셋에서는 100개의 프롬프트를 추출하여 모델 종류와 요청 방식에 맞게 사용할 수 있도록 `JSONL` 형식으로 전처리해두었습니다.

| Task Type | Description | Dataset (Hugging Face) |
|-|-|-|
| **Chat** | 자유로운 질의응답, 어시스턴트 대화 스타일 | [`OpenAssistant/oasst1`](https://huggingface.co/datasets/OpenAssistant/oasst1) |
| **Summarization** | 뉴스 기사 요약 | [`abisee/cnn_dailymail`](https://huggingface.co/datasets/abisee/cnn_dailymail) |
| **QA** | 일반 상식 기반의 질의응답 (closed-book) | [`mandarjoshi/trivia_qa`](https://huggingface.co/datasets/mandarjoshi/trivia_qa) |
| **Code Generation** | 함수 수준의 파이썬 코드 생성 | [`openai/humaneval`](https://huggingface.co/datasets/openai/openai_humaneval) |

## Results

### GPT-2 Latency & P2FT

GPT-2 모델에 대해 태스크별 Latency를 직접 구현한 **Serving** 스크립트와 vLLM에서 각각 측정했습니다.
왼쪽 그래프는 전체 응답을 수신하기까지의 시간(Total **Latency**), 오른쪽 그래프는 첫 토큰을 수신하기까지의 시간(Prompt-to-First-Token, **P2FT**)을 나타냅니다.

<p align="center">
 <img src="https://github.com/user-attachments/assets/22582dc0-b95b-4c7a-b9ae-c761ab7885c4" width="45%"/>
 <img src="https://github.com/user-attachments/assets/030de9cd-07cd-4999-b4e1-49647ee752a1" width="45%"/>
</p>

### Llama 2 Latency & P2FT

Llama 2 모델에 대해 태스크별 Latency를 직접 구현한 **Serving** 스크립트와 vLLM에서 각각 측정했습니다.
왼쪽 그래프는 전체 응답을 수신하기까지의 시간(Total **Latency**), 오른쪽 그래프는 첫 토큰을 수신하기까지의 시간(Prompt-to-First-Token, **P2FT**)을 나타냅니다.

<p align="center">
 <img src="https://github.com/user-attachments/assets/d5119c33-d807-4f4c-9f24-39cf97877924" width="45%"/>
 <img src="https://github.com/user-attachments/assets/8188701c-d076-4944-82a4-2e3218016c97" width="45%"/>
</p>

### Token Throughput on Chat

동시 요청 수에 따른 throughput(`**token**/s`)을 측정했습니다. 왼쪽 그래프는 동시 요청수 `p=2`에서의 throughput, 오른쪽 그래프는 동시 요청수 `p=4`에서의 throughput을 나타냅니다.

<p align="center">
 <img src="https://github.com/user-attachments/assets/b1099dd6-0ba1-4836-b60b-bc18fbbf3ef8" width="45%"/>
 <img src="https://github.com/user-attachments/assets/5448a134-f81f-4284-b814-a28af52c5a02" width="45%"/>
</p>

### GPU VRAM Usage on Chat

동시 요청 수에 따른 **GPU VRAM** 사용률을 측정했습니다. 왼쪽 그래프는 동시 요청수 `p=2`에서의 **VRAM** 사용률, 오른쪽 그래프는 동시 요청수 `p=4`에서의 **VRAM** 사용률을 나타냅니다.

<p align="center">
 <img src="https://github.com/user-attachments/assets/82c42e21-7011-4a92-9780-ea9c4fc590ec" width="45%"/>
 <img src="https://github.com/user-attachments/assets/29b27697-2e8e-435e-b17a-90a5ce91d9d0" width="45%"/>
</p>

## Additional Resources
[Google slide - 중간 발표](https://docs.google.com/presentation/d/1dIXP-vJu0QszjQoBENqWEVvm_JJ5_WqBpWjJHkA5G68/edit?usp=sharing)
