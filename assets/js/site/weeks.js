// Single source of truth for the 12-week curriculum.
// `ready: true` marks weeks whose pages exist; update it when a week ships.

export const PARTS = [
  { no: 1, title: '언어를 숫자로', sub: '다음 토큰 예측 · 토큰화 · 임베딩', weeks: [1, 2, 3] },
  { no: 2, title: '신경망과 트랜스포머', sub: '신경망 언어 모델 · 어텐션 · 트랜스포머 블록', weeks: [4, 5, 6] },
  { no: 3, title: '미니 GPT 만들기', sub: '학습 · 디코딩 · 평가와 스케일링', weeks: [7, 8, 9] },
  { no: 4, title: '실전 LLM으로', sub: '챗봇 만들기 · API 활용 · 한계와 RAG', weeks: [10, 11, 12] },
  { no: 5, title: 'LLM 응용 실습', sub: '텍스트 처리 · 챗봇 · 에이전트 · 로컬 LLM', weeks: [13, 14, 15, 16] },
];

export const WEEKS = [
  { no: 1, icon: '🎲', title: '언어 모델: 다음 토큰 예측', widget: '바이그램 언어 모델 (빈도표 · 확률 · 생성)', llm: '—', ready: true },
  { no: 2, icon: '✂️', title: '토큰화와 BPE', widget: 'BPE 병합 스텝퍼', llm: '—', ready: false, core: true },
  { no: 3, icon: '🧭', title: '임베딩: 단어를 벡터로', widget: '임베딩 공간 탐색기 (최근접 · 벡터 산술)', llm: '—', ready: false },
  { no: 4, icon: '🧮', title: '신경망 언어 모델과 학습', widget: '소프트맥스 · 교차 엔트로피 · 자동 미분', llm: '—', ready: false },
  { no: 5, icon: '👀', title: '어텐션', widget: '어텐션 계산기 (Q·K·V · 인과 마스크)', llm: '—', ready: false, core: true },
  { no: 6, icon: '🧱', title: '트랜스포머 블록', widget: '트랜스포머 해부도 (위치 인코딩 · 파라미터 수)', llm: '—', ready: false },
  { no: 7, icon: '🏋️', title: '미니 GPT 학습', widget: '브라우저 GPT 학습기', llm: '—', ready: false, core: true },
  { no: 8, icon: '🎛️', title: '디코딩 전략', widget: '샘플링 조절판 (온도 · top-k · top-p)', llm: '—', ready: false, core: true },
  { no: 9, icon: '📏', title: '평가와 스케일링', widget: '퍼플렉서티 · 메모리 계산기', llm: '—', ready: false },
  { no: 10, icon: '💬', title: '사전학습에서 챗봇까지', widget: '채팅 템플릿 · LoRA 계산기', llm: '—', ready: false },
  { no: 11, icon: '🔌', title: 'LLM API와 프롬프트', widget: 'API 플레이그라운드 (구조화 출력 · 도구 호출)', llm: '선택', ready: false },
  { no: 12, icon: '🌉', title: 'LLM의 한계와 RAG로', widget: '환각 해부 · 종합 프로젝트', llm: '선택', ready: false },
  { no: 13, icon: '🗂️', title: '응용 ① 텍스트 처리 자동화', widget: '문서 일괄 처리기 (요약 · 분류 · 추출)', llm: '선택', ready: false, applied: true },
  { no: 14, icon: '🤖', title: '응용 ② 대화형 챗봇 만들기', widget: '챗봇 빌더 (페르소나 · 메모리 · 스트리밍)', llm: '선택', ready: false, applied: true },
  { no: 15, icon: '🕹️', title: '응용 ③ AI 에이전트', widget: '에이전트 루프 추적기 (계획 · 도구 · 관찰)', llm: '선택', ready: false, applied: true },
  { no: 16, icon: '💻', title: '응용 ④ 브라우저 로컬 LLM', widget: '온디바이스 LLM 실행기', llm: '—', ready: true, applied: true },
];

// The learning path this course belongs to: MLBasic → ML Node Studio → (this) → studyRAG.
export const PATHWAY = [
  { step: '①', icon: '🤖', title: '머신러닝 기초', sub: 'studyMLBasic 파이썬 scikit-learn 신경망', url: 'https://samcho93.github.io/studyMLBasic/' },
  { step: '②', icon: '🧩', title: 'ML Node Studio', sub: 'MLStudio 노드 시각 실습', url: 'https://samcho93.github.io/MLStudio/' },
  { step: '③', icon: '🧠', title: 'LLM 원리와 활용', sub: 'studyLLM 이 과정', current: true },
  { step: '④', icon: '🔎', title: 'RAG 시스템 구축', sub: 'studyRAG 검색증강생성', url: 'https://samcho93.github.io/studyRAG/' },
];

export const weekId = (no) => `w${String(no).padStart(2, '0')}`;
