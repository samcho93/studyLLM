# CLAUDE.md — LLM Lab 강의 사이트

이 파일은 Claude Code가 이 저장소에서 작업할 때 따라야 할 프로젝트 규칙입니다.

---

## 1. 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 정식 교과목명 | 대규모 언어 모델(LLM) 원리와 활용 |
| 부제 | 다음 토큰 예측부터 미니 GPT까지, 만들어 보며 배우는 LLM |
| 사이트 명 | **LLM Lab** |
| 배포 URL | `https://samcho93.github.io/studyLLM` |
| 대상 | 한국폴리텍대학 분당융합기술교육원 AI응용소프트웨어과 학생 |
| 기간 | 12주 정규 + 응용 실습 4주 (주 1회 3시간 기준) |
| 언어 | 한국어 (코드 주석·변수명은 영어) |

### 학습 경로 (시리즈)

```
① studyMLBasic (머신러닝 기초) → ② MLStudio (ML Node Studio) → ③ studyLLM (이 과정) → ④ studyRAG (RAG 시스템 구축)
```

- 선수: MLBasic의 신경망 · 소프트맥스 · 역전파 · 텍스트 분류(토큰화, BoW)
- 후속: studyRAG. **같은 학과 문서 코퍼스**를 쓰고, 12주차가 RAG 1주차로 이어진다
- 주차별 연결 고리는 학생용 페이지의 `.bridge` 상자로 표시한다 (예: 2주차 BPE → RAG 1주차 토큰 카운터)

### 이 사이트가 다른 LLM 강의와 다른 점

1. **블랙박스를 열어 본다** — 빈도표·BPE·어텐션·역전파를 JavaScript로 직접 구현한다
2. **브라우저에서 GPT를 학습한다** — 7주차에 미니 GPT(문자 단위)를 처음부터 학습시킨다
3. **RAG로 이어진다** — 내가 만든 모델의 환각이 곧 RAG가 필요한 이유가 된다

---

## 2. 기술 스택과 절대 제약

- **순수 HTML + CSS + Vanilla JavaScript** (ES Modules), 빌드 도구·번들러·프레임워크 **없음**
- GitHub Pages 정적 호스팅

| 제약 | 이유 |
|---|---|
| Node 빌드 과정을 도입하지 말 것 | `git push` 만으로 배포되는 구조를 유지한다 |
| 서버 코드를 만들지 말 것 | 정적 호스팅이라 실행되지 않는다 |
| API 키를 코드나 저장소에 넣지 말 것 | 공개 저장소다. 키는 사용자가 입력한다 (`sessionStorage`만) |
| `http://` 리소스를 참조하지 말 것 | mixed content로 차단된다 |
| CDN은 `https://` + 버전 고정 | 재현성 확보. `@latest` 금지 |

외부 라이브러리는 **추가하기 전에 반드시 사용자에게 확인**받는다. 현재 CDN 의존성은 글꼴(Pretendard, JetBrains Mono)과 16주차 로컬 LLM용 `@huggingface/transformers@3.8.1`(studyRAG와 같은 버전 고정)이다.
`tools/`의 Node 스크립트는 체크포인트·사전 계산 데이터를 만드는 **오프라인 도구**이며 배포에 필요하지 않다.

---

## 3. 디렉토리 구조

```
studyLLM/
├── index.html · student.html · teacher.html   # 랜딩 / 학생용·교사용 진입점
├── assets/
│   ├── css/            # tokens · base · components(3단 레이아웃) · slides · wNN(주차별)
│   ├── js/
│   │   ├── core/       # 순수 JS 엔진 (DOM 없음 — Node와 Worker에서도 동작)
│   │   │   ├── rng.js        # 재현 가능한 난수 (mulberry32) · 확률 샘플링
│   │   │   ├── text.js       # 코퍼스 로딩 · 문자 어휘
│   │   │   ├── ngram.js      # n-gram 빈도표 · 확률 · 생성 · NLL
│   │   │   ├── bpe.js        # BPE 학습 · 인코딩 · 디코딩
│   │   │   ├── value.js      # 스칼라 자동 미분 (micrograd 방식, 4주차 교육용)
│   │   │   ├── tensor.js     # 텐서 자동 미분 (미니 GPT 엔진)
│   │   │   ├── gpt.js        # 미니 GPT 모델 · 학습 · 생성 · 직렬화
│   │   │   ├── gpt-worker.js # 학습을 Web Worker에서 돌린다 (7주차)
│   │   │   ├── sampling.js   # 소프트맥스 · 온도 · top-k · top-p
│   │   │   └── llm.js        # 사용자 API 키로 LLM 호출 (11·12주차, 선택)
│   │   ├── widgets/    # 주차별 실습 위젯 (위젯 하나 = 파일 하나)
│   │   └── site/       # site.js(틀·목차·진도·테마) · weeks.js(주차 메타) · slides.js · ink.js · result.js · runner.js
│   └── data/
│       ├── corpus/     # 전 과정 공통 문서셋 (studyRAG 학과 문서 + LLM 교안)
│       ├── models/     # 미리 학습한 미니 GPT 체크포인트 (tools/로 생성)
│       └── wNN/        # 주차별 보조 데이터
├── weeks/wNN/index.html · teacher.html   # 학생용 문서 / 교사용 슬라이드
├── teacher/index.html                    # 교수자용 허브
└── tools/                                # 오프라인 Node 스크립트 (체크포인트 학습 등)
```

---

## 4. 커리큘럼 (12주 + 응용 실습 4주)

| 주차 | 주제 | 실습 위젯 | LLM 키 | studyRAG 연결 |
|---|---|---|---|---|
| 01 | 언어 모델: 다음 토큰 예측 | 바이그램 언어 모델 | — | |
| 02 | 토큰화와 BPE | **BPE 병합 스텝퍼** | — | RAG 1주차 토큰 카운터 |
| 03 | 임베딩: 단어를 벡터로 | 임베딩 공간 탐색기 | — | RAG 5주차 문장 임베딩 |
| 04 | 신경망 언어 모델과 학습 | 소프트맥스 · 교차 엔트로피 · 자동 미분 | — | |
| 05 | 어텐션 | **어텐션 계산기** | — | |
| 06 | 트랜스포머 블록 | 트랜스포머 해부도 | — | |
| 07 | 미니 GPT 학습 | **브라우저 GPT 학습기** | — | |
| 08 | 디코딩 전략 | **샘플링 조절판** | — | RAG 2주차 (온도와 사실성) |
| 09 | 평가와 스케일링 | 퍼플렉서티 · 메모리 계산기 | — | RAG 9주차 평가 |
| 10 | 사전학습에서 챗봇까지 | 채팅 템플릿 · LoRA 계산기 | — | RAG 1주차 (파인튜닝 vs RAG) |
| 11 | LLM API와 프롬프트 | API 플레이그라운드 | 선택 | RAG 2주차 프롬프트 |
| 12 | LLM의 한계와 RAG로 | 환각 해부 · 종합 프로젝트 | 선택 | RAG 1·7주차 |
| **PART 5 · LLM 응용 실습** (정규 12주 뒤 심화·보충) | | | | |
| 13 | 응용 ① 텍스트 처리 자동화 | 문서 일괄 처리기 (요약·분류·추출) | 선택 | RAG 3주차 파싱 |
| 14 | 응용 ② 대화형 챗봇 만들기 | 챗봇 빌더 (페르소나·메모리·스트리밍) | 선택 | RAG 11주차 웹 UI |
| 15 | 응용 ③ AI 에이전트 | 에이전트 루프 추적기 | 선택 | RAG 7주차 (검색 도구) |
| 16 | 응용 ④ 브라우저 로컬 LLM | 온디바이스 LLM 실행기 | — | RAG 5주차 (Transformers.js) |

**굵게 표시된 위젯이 핵심 자산**이다. **LLM 키 없이도 모든 주차의 실습이 동작해야 한다**(키가 없으면 수업용 예시 응답).

---

## 5. 실습 위젯 작성 규칙 (studyRAG와 동일)

1. **하나의 개념만** 다룬다
2. **입력을 바꾸면 결과가 즉시 바뀐다** — 설명보다 조작이 먼저다
3. **실패를 보여준다** — "이 설정에서는 왜 망가지는가"를 볼 수 있어야 한다
4. `<template>` + ES Module, 위젯 하나 = 파일 하나. `export function mount(el, options)` / `unmount(el)`
5. 초기 상태에서 이미 뭔가 보여야 한다. 빈 화면 + "시작하기" 버튼 금지
6. 모바일 세로 화면(375px)에서 조작 가능해야 한다
7. 결과는 `result.js`의 `registerOutput()`으로 오른쪽 터미널에 LIVE 카드로 보낸다
8. 무거운 계산(학습)은 Web Worker에서 돌려 화면이 멈추지 않게 한다

---

## 6. 콘텐츠 작성 규칙

### 학생용 (`weeks/wNN/index.html`) — 6개 섹션 고정

1. **학습 목표** — "~할 수 있다" 3개 이내
2. **왜 필요한가** — 이 기법이 없을 때 생기는 문제부터
3. **개념** — 그림/도식 우선(인라인 SVG), 수식은 직관 설명 뒤에
4. **실습** — 위젯 조작 → 관찰 질문
5. **Challenge** — 4~5개, 난이도 오름차순, 코드 스켈레톤(`pre[data-run]`, ▶ 실행)
6. **정리 & 다음 주 예고** — 관련 studyRAG/MLBasic 주차가 있으면 `.bridge` 상자로 연결

### 교수자용 (`weeks/wNN/teacher.html`) — PPT형 슬라이드 덱

- 16:9 슬라이드(`<section class="slide">`) + 교사 노트(`<aside class="notes">`)
- 데모 슬라이드(`slide--demo`)에 실제 위젯, 코드 슬라이드(`pre[data-run]`) ▶ 실행
- 덱 끝 **부록**(`data-appendix`): A 강의안(시간 배분) · B Challenge 정답 · C 자주 막히는 지점 · D 토론 · E 루브릭

### 문체와 용어

- 학생용은 평서체(`~한다`). 존댓말 혼용 금지
- 영문 용어는 첫 등장 시 1회만 병기: 대규모 언어 모델(LLM) → 이후 LLM
- 표기 통일: `토큰`, `토크나이저`, `임베딩`, `어텐션`, `트랜스포머`, `소프트맥스`, `교차 엔트로피`, `퍼플렉서티`, `디코딩`, `미세조정`

---

## 7. 디자인 규칙

- studyRAG · studyMLBasic과 **같은 형태·디자인** (3단: 목차 · 내용 · 실행 결과 터미널)
- 틀은 `site.js`가 만든다. 페이지는 `<div class="layout" data-layout data-role data-week data-crumb>` 안에 `<main class="pane center">`만 둔다
- 색·간격·폰트는 `tokens.css`의 CSS 변수만 사용. 다크모드 · 375px 모바일 · 키보드 포커스 필수
- 아이콘은 인라인 SVG/이모지. 아이콘 폰트 금지
- localStorage 키 접두사는 `llmlab:`

---

## 8. 작업 방식

- 새 주차: **학생용 → 위젯 → 교수자용** 순서
- 로컬 확인: `python -m http.server 8766` (ES Module·fetch는 `file://`에서 안 됨)
- 새 주차가 완성되면 `weeks.js`의 `ready: true`로 바꾼다
- 커밋 메시지: `feat(w04): 자동 미분 위젯 추가` 형식
