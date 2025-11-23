
# UCWARE LLM API  
___
UCWARE LLM API는 PDF 문서 및 채팅 기록에 대한 요약 및 질의응답 기능을 제공하는 서비스입니다.

## I. Features
___
- PDF 문서 / 채팅 기록 요약: PDF 문서 및 채팅 데이터를 업로드하면 핵심 내용을 자동으로 요약하여 제공합니다.
- 사용자 질의응답: 문서나 대화 기록을 바탕으로 한 Q&A 기능을 제공합니다. Tavily API Key가 제공될 경우 웹 검색 또한 가능합니다.
- 멀티모달 PDF 자습서 생성: PDF 문서의 텍스트와 이미지를 활용하여 학습자 친화적인 자습서를 자동으로 생성합니다. 이미지 캡션 생성 및 의미 단위 그룹화를 통해 체계적인 학습 가이드를 제공합니다.
- 캐싱 및 재사용: 동일한 파일에 대한 반복 요청 시 결과를 Redis 캐시에서 즉시 반환하여 응답 속도를 높입니다. 벡터 임베딩도 최초 1회 생성 후 DB에 저장되므로 이후부터 빠르게 검색합니다.  
- 피드백 수집: 사용자로부터 요약/답변에 대한 평점(1~5)과 의견을 받아 품질 향상에 활용할 수 있습니다.

## II. Setup
___
### 0. Requirements
- 이 서비스를 이용하기 위해서 아래와 같은 환경이 요구됩니다: 
	- OpenAI API Token (OpenAI LLM 이용시)
	- Tavily API Key (Web Search 이용시)
	- HuggingFace Hub Token (vLLM 이용시)

### 1. 의존성 설치 및 가상환경 구성
```bash
bash setup_env.sh
```
   - 서비스 사용을 위한 사전 환경 설정을 진행합니다.
   - 수행 기능 목록:
	   - 시스템 패키지 및 requirements.txt 기반 pip 패키지 설치
	   - llm 제공자 선택(OpenAI, HuggingFace)
	   - 서비스 사용에 필요한 환경 변수 설정(`OPENAI_API_KEY`, `TAVILY_API_KEY`를 받아옴. 다른 환경 변수들은 쉘파일 내 하드코딩되어있으므로 유동적으로 변경.)
	   - 환경변수를 저장해둘 `.env` 파일 자동 생성

### 2. (선택) 로컬 LLM 서버 실행

```bash
bash run_vllm.sh
```

   - vllm을 이용해 llm 모델을 서빙합니다. `1. 의존성 설치 및 가상환경 구성` 에서 llm 제공자를 OpenAI로 선택했다면 이 단계는 생략합니다.
   - `HUGGING_FACE_HUB_TOKEN`을 입력받습니다. 필요하지 않은 경우 공란으로 넘어갑니다.
   - 기본적으로 12000번 포트, `Qwen/Qwen3-30B-A3B-GPTQ-Int4` 모델을 이용하도록 설계되어있으며 유동적으로 변경이 가능합니다.
   - 아래 명령어를 이용해 서버를 종료할 수 있습니다
```bash
bash scripts/stop_vllm.sh   
```
  

### 3. 서비스 전체 실행

   ```bash
   bash run_all.sh
   ```

   - 서비스 시작을 위해 Redis, Chroma(DB), FastAPI 서버를 구동합니다.
   - 기본 포트는 아래와 같습니다.

|     서버     |        포트        |
| :--------: | :--------------: |
|   Redis    |       6379       |
| Chroma(DB) |       9000       |
|  FastAPI   | 사용자 입력(기본값 8000) |
- 아래 명령어로 서버를 종료할 수 있습니다.
```bash
bash scripts/stop_services.sh
```
## III. Usage
___

### 1. PDF 요약 및 질의 (POST `/api/summary`)
- `query`가 `SUMMARY_ALL` 인 경우 `summary` 필드에 전체 요약문을 출력합니다.
- `SUMMARY_ALL` 이외의 `query`인 경우 자체적으로 사용자의 질문이라 판단해 답변을 `answer` 필드에 출력합니다.

-  요청 바디 예문:
```json
{
  "file_id": "fid_abc123",
  "pdf_url": "https://example.com/sample.pdf",
  "query": "SUMMARY_ALL",
  "lang": "KO"
}
```

- 응답 바디 예문:
```json
{
  "file_id": "fid_abc123",
  "summary": "이 문서는 AI 기술에 관한 논문으로...",
  "cached": false,
  "log": ["load_pdf attempt 1 [120ms]", "..."]
}
```
### 2. 채팅 요약 및 질의 (POST `/api/chat-summary`)
- PDF Summary와 마찬가지로 `query`가 `SUMMARY_ALL` 인지 아닌지에 따라 요약, 또는 답변을 각각 `summary`, `answer` 필드에 출력합니다.
- 요청 바디로 `chats` , `query` , `lang`, 필드를 전달합니다.
- `chats`는 채팅 메시지 객체들의 배열이며, 각 객체는 `chat_id`, `plaintext` , `sender` , `timestamp` 필드를 포함해야 합니다.

- 요청 바디 예문:
```json
{
  "chats": [
    {
      "chat_id": "room1",
      "plaintext": "오늘 회의 시작하겠습니다.",
      "sender": "user1",
      "timestamp": "2025-07-28T09:00:00"
    }
  ],
  "query": "SUMMARY_ALL",
  "lang": "ko"
}
```

- 응답 바디 예문:
```json
{
  "summary": "오늘 회의에서는 프로젝트 일정에 대해 논의했습니다.",
  "log": ["summarize:1", "translate:1"]
}
```
### 3. 피드백 제출 (POST `/api/feedback`)
- 서비스에 대한 만족도 평가를 제출용 엔드포인트로, 의무적으로 사용할 필요는 없습니다.
  
- 요청 바디 예문:
```json
{
  "file_id": "fid_abc123",
  "pdf_url": "https://example.com/sample.pdf",
  "lang": "KO",
  "rating": 5,
  "comment": "정확하고 빠른 요약 감사합니다!",
  "usage_log": ["load_pdf:1", "summary:1"]
}
```

- 응답 바디 예문문:
```json
{
  "id": "fb_123e4567-e89b-12d3-a456-426614174000",
  "created_at": "2025-07-28T02:15:30.123456",
  "ok": true
}
```

### 4. 멀티모달 PDF 자습서 생성 (POST `/api/tutorial`)
- PDF 문서를 기반으로 이미지와 텍스트를 포함한 자습서를 자동 생성합니다.
- 생성된 자습서는 Markdown 형식으로 반환되며, 이미지는 base64 data-URI로 임베딩됩니다.
- 지원 언어: `ko` (한국어), `en` (영어), `ja` (일본어), `zh` (중국어)
- 요청 바디로 `file_id`, `pdf_url`, `lang` 필드를 전달합니다.

- 요청 바디 예문:
```json
{
  "file_id": "fid_abc123",
  "pdf_url": "https://example.com/sample.pdf",
  "lang": "ko"
}
```

- 응답 바디 예문:
```json
{
  "file_id": "fid_abc123",
  "tutorial": "# 자습서 가이드\n\n## 섹션 1\n\n이 섹션에서는...\n\n![이미지 캡션](data:image/png;base64,...)\n\n...",
  "cached": false,
  "log": ["load attempt 1 [250ms]", "generate attempt 1 [1200ms]", "combine attempt 1 [800ms]"]
}
```

## IV. Structure
___

- 서비스의 전반적이 구조는 아래와 같습니다.
```
.
├── run_all.sh, run_vllm.sh       # 실행 스크립트
├── setup_env.sh                  # 환경 변수 설정
├── scripts/                      # 실행/종료 스크립트 모음
└── app/                          # 애플리케이션 루트
    ├── controller/               # HTTP 요청 라우팅 및 엔드포인트 핸들러
    ├── dto/                      # 요청/응답용 데이터 클래스
    ├── service/                  # LangGraph 기반 서비스 로직
    ├── domain/                   # 인터페이스 정의
    ├── infra/                    # 인터페이스 구현체
    ├── cache/, vectordb/         # 저장 계층
    │
    ├── main.py                   # FastAPI 진입점
    └── prompts.py                # 프롬프트 문자열 모음
```

## V. LangGraph 파이프라인 - ColPali 및 Docling
___

- PDF 요약과 질의 파이프라인은 `app/service/summary_graph_builder.py` 에 정의된 **LangGraph 기반 상태 머신**으로 동작합니다.
- `RAG_router` 노드에서 사용자의 질의 유형을 판별하여, 아래 3가지 경로 중 하나로 분기합니다.
  - **텍스트 RAG 경로** - 문서 벡터 검색용 노드 `retrieve_vector`
  - **웹과 문서 하이브리드 경로** - 웹 검색과 문서 RAG를 함께 사용하는 노드 `retrieve_web`
  - **구조적 질의 경로 - 그림, 표, 페이지 등** - ColPali 비전 RAG 노드 `retrieve_colpali`

### 1. PDF 요약과 질의 LangGraph 구조

아래는 PDF 요약과 질의에 사용되는 LangGraph의 주요 노드와 분기 구조입니다.

```mermaid
flowchart LR
    E[entry\n요청 초기화/캐시 확인]
    L[load\nPDF 로드]
    EMB[embed\n임베딩/벡터스토어 저장]
    R[RAG_router\n질문 유형/ColPali/Web 판단]

    RV[retrieve_vector\n텍스트 RAG 검색]
    RW[retrieve_web\n웹+문서 혼합 검색]
    RC[retrieve_colpali\nColPali 비전 RAG]
    S[summarize\n전체 요약]

    G[grade\n관련 청크 필터링]
    GEN[generate\n답변 생성]
    V[verify\n답변 검증]
    RF[refine\n질문 리파인]
    SV[save\n요약 캐시 저장]
    T[translate\n언어 번역]
    F[finish\n로그 기록/종료]

    %% entry 분기
    E -->|캐시 HIT & SUMMARY_ALL| T
    E -->|캐시 HIT & 질의 모드| R
    E -->|임베딩 없음| L
    E -->|임베딩 있음| R
    E -->|에러| F

    %% 로딩/임베딩
    L --> EMB
    EMB --> R

    %% RAG 라우터 분기
    R -->|SUMMARY_ALL| S
    R -->|구조적 질의 - 표, 그림, 페이지| RC
    R -->|웹 검색 필요| RW
    R -->|문서 기반 RAG| RV

    %% 요약 모드
    S --> SV
    SV --> T

    %% 텍스트/웹 RAG 경로
    RV --> G
    RW --> G

    G -->|충분한 근거| GEN
    G -->|근거 부족/미관련| T

    GEN --> V

    %% 검증 단계 - 현재 구현에서는 질의 모드에서만 사용
    V -->|답변 양호 - 질의 모드| T
    V -->|답변 부족 또는 불충분| RF
    V -->|에러 발생| F

    %% 리파인
    RF -->|리파인 쿼리로 재시도| R
    RF -->|비관련/리파인 한계| T

    %% ColPali 경로
    RC -->|답변 생성 성공| T
    RC -->|에러| F

    %% 종료
    T --> F
```

### 2. 채팅 요약과 질의 LangGraph 구조

- 채팅 로그 기반 요약/질의 파이프라인은 `app/service/chat_graph_builder.py` 의 `ChatGraphBuilder` 로 정의되며, `chat_summary_graph.py` 를 통해 FastAPI에서 사용됩니다.

```mermaid
flowchart LR
    E[entry\n요약/질의 모드 판별]
    S[summarize\n채팅 로그 요약]
    A[answer_node\n채팅 기반 1차 답변 생성]
    V[verify\n답변 적절성 검증]
    R[refine\n답변 리파인]
    T[translate\n요약/답변 번역]
    F[finish\n종료]

    %% entry 분기
    E -->|query == SUMMARY_ALL| S
    E -->|그 외| A

    %% 요약 모드
    S --> T

    %% 질의 모드
    A --> V

    %% 검증과 리파인 루프
    V -->|정확함 - true| T
    V -->|부족하거나 부분적으로 부정확함 - false| R
    V -->|전혀 무관함 - bad 또는 에러| F

    R -->|리파인 답변 재검증| V
    R -->|에러| F

    %% 번역과 종료
    T --> F
```

### 3. Docling 기반 PDF 전처리 파이프라인

- PDF 콘텐츠는 LangGraph에 들어오기 전에 `app/infra/pdf_receiver.py` 와 `app/infra/semantic_chunker.py` 를 통해 Docling 기반 전처리를 거칩니다.
- 전체 흐름은 아래와 같습니다.
  - PDF URL 입력
  - `PDFReceiver` 가 Docling을 사용해 텍스트와 그림 정보를 가진 `PageElement` 리스트로 변환
  - `SemanticChunker` 가 `PageElement` 리스트를 의미 단위 청크로 묶어서 텍스트 청크 또는 `PageChunk` 리스트로 변환
  - `PdfLoader` 가 이 청크들을 LangGraph의 `load` 노드에서 사용할 수 있도록 반환

```mermaid
flowchart LR
    U[PDF URL]
    R[PDFReceiver - Docling 변환]
    E[PageElement 리스트]
    C[SemanticChunker - 의미 단위 청킹]
    CH[텍스트 청크 또는 PageChunk 리스트]
    G[Summary LangGraph - entry와 load 노드]

    U --> R --> E --> C --> CH --> G
```

### 4. 멀티모달 PDF 자습서 생성 LangGraph 구조

- PDF 자습서 생성 파이프라인은 `app/service/guide_graph_builder.py` 의 `GuideGraphBuilder` 로 정의되며, `guide_service_graph.py` 를 통해 FastAPI에서 사용됩니다.
- PDF의 텍스트와 이미지를 함께 처리하여 학습자 친화적인 자습서를 생성합니다.
- 전체 흐름은 아래와 같습니다:
  - PDF 로드: Docling을 통해 텍스트와 이미지를 추출하고, VLM을 사용해 이미지 캡션 생성
  - 의미 단위 그룹화: 유사도 기반으로 청크들을 학습 단위로 그룹화
  - 섹션 생성: 각 그룹에 대해 병렬로 자습서 섹션 생성 (이미지 포함)
  - 번역 및 통합: 생성된 섹션들을 병렬로 번역하고 하나의 Markdown 문서로 통합
  - 이미지 임베딩: 이미지 ID를 base64 data-URI로 교체하여 최종 자습서 완성

```mermaid
flowchart LR
    L[load\nPDF 로드 및 이미지 캡션 생성]
    G[generate\n의미 단위 그룹화 및 섹션 생성]
    C[combine\n섹션 번역 및 통합]
    F[finish\n이미지 임베딩 및 완성]

    %% 로드 단계
    L -->|PageChunk 리스트| G
    L -->|에러| F

    %% 생성 단계
    G -->|섹션 리스트| C
    G -->|에러| F

    %% 통합 단계
    C -->|번역된 섹션들| F

    %% 종료
    F -->|최종 자습서| END[완료]
```
