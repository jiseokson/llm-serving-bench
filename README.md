## 실험 목적

- 다양한 서빙 프레임워크(Hugging Face 🤗 Transformers, TGI, vLLM)의 구조적 차이 및 성능을 비교
- 각기 다른 LLM 활용 태스크에 대해 latency, throughput, memory usage 등을 측정하고 분석

## 실험 구성 요소

### 모델

| 모델 이름 | Hugging Face 모델 ID | 파라미터 수 | Transformers | TGI | vLLM |
| --- | --- | --- | --- | --- | --- |
| **GPT-2 Small** | `gpt2` | 124M | ✅ | ✅ | ✅ |
| **Phi-2** | `microsoft/phi-2` | 2.7B | ✅ | ✅ | ✅ |
| **LLaMA 2 7B** | `meta-llama/Llama-2-7b-chat-hf` | 6.7B | ✅ | ✅ | ✅ |
| **Mistral 7B Instruct** | `mistralai/Mistral-7B-Instruct-v0.1` | 7.3B | ✅ | ✅ | ✅ |

## 서빙 프레임워크

- Hugging Face 🤗 Transformers
- TGI (Text Generation Inference)
- vLLM

## 환경

RunPod A100 (40GB)

Python 3.X, CUDA X.X, PyTorch 2.X, …

## 실험 대상 태스크

| 태스크 유형 | 입력 길이 | 출력 길이 | 예시 |
| --- | --- | --- | --- |
| Chat | 20~100 tokens | 20~100 tokens | 자유 질의응답 |
| Summarization | 200~800 tokens | 50~100 tokens | 문서 요약 |
| QA (closed-book) | 100~500 tokens | 10~30 tokens | 단답형 질문 |
| Code Generation | 10~50 tokens | 50~150 tokens | 함수 자동 생성 |

## 측정 지표 및 수집 방식

| 지표 | 측정 방법 | 비고 |
| --- | --- | --- |
| **Latency (ms)** | 요청 시작~응답 완료 시간 | 평균, 표준편차, 중앙값 등 포함 |
| **Throughput (req/sec)** | 초당 처리 가능한 요청 수 | 동시 요청 실험 포함 |
| **Memory usage (MB)** | `nvidia-smi`, `torch.cuda.memory_allocated()` | peak 및 steady state |
| **Token generation rate** | `tokens/sec` 기준 | 특히 generation 태스크에서 중요 |
| **Stability under load** | locust or custom async client | 동시 요청 1~20명 시 실험 |
