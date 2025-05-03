## 실험 목적

- 다양한 서빙 프레임워크(Hugging Face 🤗 Transformers, TGI, vLLM)의 구조적 차이 및 성능을 비교
- 각기 다른 LLM 활용 태스크에 대해 latency, throughput, memory usage 등을 측정하고 분석

## 실험 구성 요소

### 모델

| 모델 이름 | Hugging Face 모델 ID | 파라미터 수 | Transformers | TGI | vLLM |
| --- | --- | --- | --- | --- | --- |
| **GPT-2 Small** | `openai-community/gpt2` | 124M | ✅ | ✅ | ✅ |
| **Phi-2** | `microsoft/phi-2` | 2.7B | ✅ | ✅ | ✅ |
| **LLaMA 2 7B** | `meta-llama/Llama-2-7b-chat-hf` | 6.7B | ✅ | ✅ | ✅ |
| **Mistral 7B Instruct** | `mistralai/Mistral-7B-Instruct-v0.1` | 7.3B | ✅ | ✅ | ✅ |

### 서빙 프레임워크
| 항목               | 🤗 Transformers                         | TGI (Text Generation Inference)               | vLLM                                                 |
| ---------------- | --------------------------------------- | --------------------------------------------- | ---------------------------------------------------- |
| **개발 주체**        | Hugging Face                            | Hugging Face                                  | UC Berkeley / LMSys                                  |
| **설치 방식**        | Python 패키지<br>(`transformers`, `torch`) | Docker / CLI 기반<br>(`text-generation-server`) | CLI 기반 (`vllm`)<br>+ CUDA 설정 필요                      |
| **서빙 구조**        | 단일 요청 중심<br>Python 함수 기반 추론             | FastAPI 기반 REST 서버<br>멀티 요청 및 Streaming 지원    | C++ 커널 + Python 서버<br>PagedAttention 기반 빠른 Streaming |
| **Streaming 지원** | ❌ 미지원<br>FastAPI 등으로 커스텀 필요        | ✅ SSE 기반 Streaming 응답 지원                      | ✅ 고성능 Streaming 최적화 내장                               |
| **Batching 지원**  | ❌ 미지원                              | ✅ 자동 동적 배치<br>(Dynamic batching)              | ✅ Token-level batching<br>+ Prefill/Decode 병렬화       |
| **최적화 특징**       | 표준 PyTorch 실행                           | DeepSpeed 기반 옵티마이저 내장                         | PagedAttention 사용<br>큰 context 길이 효율적 처리             |
| **적합한 용도**       | 연구 및 프로토타이핑                             | 실서비스/프로덕션 환경                                  | 대용량, 고성능 서빙                                          |
| **장점**           | 사용법 간단<br>PyTorch 모델과 일체화               | 안정적 API 서버<br>Hugging Face Ecosystem 통합       | 고속 추론 + 낮은 latency<br>Context scaling 우수             |
| **단점**           | 성능 최적화 한계                               | 초기 셋업 복잡<br>(Docker 등)                        | 실험적 기능 포함<br>설정 학습 필요                                |

## 실험 환경

- RunPod A100 (40GB)
- CUDA 11.8.0
- Python 3.10
- PyTorch 2.1.0

## 실험 대상 태스크

| 태스크 유형 | 입력 길이 | 출력 길이 | 예시 |
| --- | --- | --- | --- |
| Chat | 20~100 tokens | 20~100 tokens | 자유 질의응답 |
| Summarization | 200~800 tokens | 50~100 tokens | 문서 요약 |
| QA (closed-book) | 100~500 tokens | 10~30 tokens | 단답형 질문 |
| Code Generation | 10~50 tokens | 50~150 tokens | 함수 자동 생성 |

## 생성 요청 파라미터

### GPT-2 Small
| **태스크** | **temperature** | **top_p** | **max_tokens** |
| --- | --- | --- | --- | 
| Chat | 0.9 | 0.95 | 100 | 
| Summarization | 0.5 | 0.9 | 60 | 
| QA | 0.3 | 1.0 | 40 | 
| Code generation | 0.8 | 0.9 | 100 | 

### Phi-2
| **태스크** | **temperature** | **top_p** | **max_tokens** | 
| --- | --- | --- | --- | 
| Chat | 0.8 | 0.9 | 100 | 
| Summarization | 0.4 | 0.9 | 60 | 
| QA | 0.2 | 1.0 | 40 | 
| Code generation | 0.7 | 0.9 | 90 | 

### LLaMA 2 7B Chat
| **태스크** | **temperature** | **top_p** | **max_tokens** | 
| --- | --- | --- | --- | 
| Chat | 0.7 | 0.95 | 150 | 
| Summarization | 0.3 | 0.9 | 80 | 
| QA | 0.1 | 1.0 | 50 | 
| Code generation | 0.6 | 0.85 | 120 | 

### Mistral 7B Instruct
| **태스크** | **temperature** | **top_p** | **max_tokens** | 
| --- | --- | --- | --- | 
| Chat | 0.8 | 0.95 | 120 | 
| Summarization | 0.3 | 0.9 | 70 | 
| QA | 0.2 | 1.0 | 50 | 
| Code generation | 0.7 | 0.9 | 110 | 

## 측정 지표 및 수집 방식

| 지표 | Sync 방식 | Streaming 방식 | 비고 / 측정 방법 요약 |
| --- | --- | --- | --- |
| **Total Latency** | ✅ 전체 응답 도착까지 시간 측정 | ✅ 마지막 토큰 수신까지 시간 측정 | start_time ~ end_time 측정 |
| **Prompt-to-First-Token (P2FT)** | ❌ 측정 어려움, 토큰 단위 수신이 아님 | ✅ 첫 토큰 도착 시간 측정 | start_time ~ 첫 SSE line 수신 시점 |
| **Token Generation Speed** (tok/s) | ❌ 측정 어려움, 토큰 단위 수신이 아님 | ✅ 토큰 수 / 응답 시간 | 토큰 수 카운팅 & 타이밍 필요 |
| **Throughput** (req/s) | ✅ 멀티 요청 실행 시간 측정 | ✅ 멀티 요청 실행 시간 측정 | 병렬 처리: asyncio.gather or threading |
| **GPU Memory Usage** | ✅ 동일 | ✅ 동일 | pynvml or nvidia-smi |
| **Host Memory Usage** | ✅ 동일 | ✅ 동일 | psutil |
| **Context Length Scaling** | ✅ 가능 | ✅ 가능 | 프롬프트 길이별 latency 비교 |
