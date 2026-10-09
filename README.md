# Multi-label tool prediction + multi-step ReAct: ALL vs SELECTED

## 무엇을 비교하는가

두 조건 모두 **gpt-6-luna가 매 스텝 여러 tool 이름과 query 인자를 예측**한다. 단순 문서 선택만 하는 실험이 아니다.

- ALL: multi-label tool prediction → 각 tool의 BM25 검색 → 누적 ranking들의 RRF → 검색된 전체 문서를 다음 ReAct 입력에 전달.
- SELECTED: multi-label tool prediction → 각 tool의 BM25 검색 → 동일한 RRF 방식 → gpt-6-luna가 관련 문서 인덱스 `[1,4]` 출력 → 선택된 문서만 다음 ReAct 입력에 전달.

예측 예시:

```json
{"action":"search","calls":[
  {"tool":"search_A_F","query":"entity biography"},
  {"tool":"search_M_R","query":"related entity birthplace"}
]}
```

이는 한 스텝의 두 레이블 선택이며 query는 각 호출의 argument다. 같은 tool을 서로 다른 query로 여러 번 호출할 수도 있다. 결과를 받은 다음 스텝에서 레이블과 query를 다시 예측한다. `finish`는 종료 action이며 검색 tool 레이블 집합에는 넣지 않는다. 이 실험은 상위 router의 `react` 레이블 선택 이후, ReAct 내부에서 검색 도구들을 선택하는 구간을 비교한다. 상위 router 자체의 성능은 평가하지 않는다.

## 데이터와 레이블의 출처 — 반드시 읽기

[HotpotQA](https://hotpotqa.github.io/) dev distractor: multi-hop 질문, 정답 answer, supporting_facts를 사용한다. 원본은 CC BY-SA 4.0으로 제공된다. 데이터는 ZIP에 포함하지 않고 prepare에서 다운로드한다.

**원본 HotpotQA에는 native tool-call 정답이 없다.** 따라서 아래 네 개 검색 tool을 만드는 adapted benchmark다. 임의의 상용 API function calling 성능으로 일반화하면 안 된다.

| 실제 실행하는 tool | BM25 검색 대상 문서 title 첫 문자 |
|---|---|
| search_A_F | A–F 및 숫자·기호 등 A 이전 문자 |
| search_G_L | G–L |
| search_M_R | M–R |
| search_S_Z | S–Z 및 그 이후 문자 |

각 tool은 별도의 BM25 index를 실제 검색한다. agent가 title 범위를 알고 도구를 선택하며, 첫 검색에서 발견한 bridge entity를 활용해 다른 tool을 추가 호출할 수 있다. 정답 tool 집합은 gold supporting document의 shard 집합이다. 두 gold 문서가 같은 shard면 단일 레이블, 다른 shard면 멀티 레이블이다. 이것은 **필요한 근거가 들어 있는 tool 집합**에 대한 진단 지표다. 탐색을 위해 다른 shard도 검색하는 정상 행동이 tool precision/EM에서 감점될 수 있다. 정확한 각 스텝의 유일한 정답 tool sequence는 원본에 없으므로 step-level tool EM은 만들지 않는다.

검색 corpus는 전체 dev 질문들의 context 문서를 title로 합친 공통 corpus다. 질문별 gold 10문서만 주는 설정보다 검색이 어렵고 공식 fullwiki corpus와도 다르다. 정답·supporting facts·gold tools는 agent/selector 입력에 전달하지 않는다. dataset dev 원본을 이용해 corpus를 구성하므로 official leaderboard 점수로 보고하지 않는다. 같은 title의 서로 다른 excerpt는 중복 문장을 제거해 합친다.

## 실행

Python 3.10+ 권장.

```bash
unzip react_rrf_multilabel_bench.zip
cd react_rrf_bench
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY='YOUR_KEY'
python3 benchmark.py prepare --n 100 --seed 42
python3 benchmark.py run --model gpt-6-luna --effort none \
  --topk 10 --pool-k 0 --steps 4 --concurrency 2 --out results_k10
```

모델은 사용자의 요청에 따른 기본값이다. 계정/endpoint에 해당 모델과 `reasoning_effort=none` 지원이 필요하다. 지원하지 않는 effort를 조용히 바꾸지 않는다. 필요한 경우 명시적으로 `--effort omit`을 지정한다. `OPENAI_BASE_URL`도 SDK가 지원한다.

다운로드 실패 시 [공식 데이터 페이지](https://hotpotqa.github.io/)의 **Dev set (distractor)**를 직접 다운로드해서:

```bash
python3 benchmark.py prepare --source /path/hotpot_dev_distractor_v1.json --n 100
```

HF gated 데이터에 의존하지 않는다. 자동 다운로드 URL들이 실패하면 위 로컬 입력을 사용한다.

먼저 비용과 응답 형식을 5문제로 확인하고 같은 디렉터리에서 이어서 100문제 실행:

```bash
python3 benchmark.py run --limit 5 --out results_k10
python3 benchmark.py run --out results_k10
```

완료 episode를 자동 건너뛰며 질문/조건별 JSON checkpoint가 남는다. 실패도 기록되어 전체 n의 EM 분모에 포함된다. 원인 수정 후 `--retry-errors`로 실패 episode만 재실행할 수 있다. 모델·retrieval·데이터 등 설정이 바뀌면 새 `--out`을 사용한다. 모델 출력이 잘리면 `--max-output`을 늘리고 새 output directory에서 시작한다. 서버/API 오류와 invalid JSON은 해당 episode 실패로 기록하며 전체 작업을 멈추지 않는다.

## RRF와 context의 정확한 동작

- 각 호출에서 shard BM25 top-k 순위를 얻는다. BM25는 영어 lowercase word tokenization을 사용한다.
- 지금까지의 모든 호출 순위를 누적하여 `score(doc)=sum(1/(60+rank))`로 RRF한다. 서로 다른 query/tool ranking을 합치는 용도이며 단일 BM25 목록에만 적용하면 순위 변화는 없다.
- 동일 문서는 stable doc_id로 중복 제거한다. 완전히 동일한 tool/query 호출은 다시 실행하지 않는다.
- `--pool-k 0`은 **현재까지 검색된 전체 고유 문서**를 의미한다. corpus 전체를 입력한다는 의미가 아니다. 유한 pool-k는 두 조건에 공통 적용되는 별도 top-k ablation이다.
- ALL은 누적 pool 전체 원문을 매 스텝 agent에 전달한다.
- SELECTED는 매 스텝 누적 pool 전체 원문을 selector에 전달하고, 선택 문서 원문만 agent에 전달한다. 이전에 탈락했던 문서도 다음 selector 후보에 다시 들어간다.
- action_log에는 tool/query/검색 개수만 보존한다. 원문은 evidence 필드에 한 번만 전달한다. 이전 원문을 messages history에도 중복 누적하면 SELECTED가 실질적으로 ALL을 보게 되므로 그렇게 구현하지 않았다.
- selector는 question과 action_log를 참고한다. bridge facts와 다음 검색을 돕는 부분 근거도 보존하도록 prompt에 지시한다.
- selector 인덱스는 0-based, 해당 스텝 후보 목록 기준이다. finish의 evidence_ids는 stable doc_id다. 빈 선택을 허용하며 invalid index는 실패 처리한다. ALL fallback으로 결과를 왜곡하지 않는다.
- selector 추가 호출은 전량 비용/지연에 포함한다. 따라서 agent 입력 토큰 감소와 시스템 총 토큰 감소는 별개다.
- 최대 검색 round 수는 4이며 마지막에 강제로 answer를 내는 추가 agent 호출이 있다. hidden chain-of-thought 출력은 요구하지 않는다.
- 한 round의 여러 검색은 독립 shard에 수행된다. 로컬 CPU 검색은 직렬 실행하며 LLM 관점에서는 한 round의 multi-call batch다. 별도 네트워크 도구 병렬화 속도를 평가하는 실험은 아니다.

## 지표

| 지표 | 의미 |
|---|---|
| answer_em | 최종 답변 정규화 Exact Match. 주 지표 |
| answer_f1 | 정규화 token F1. yes/no/noanswer special case 포함 |
| tool_set.em / precision / recall / f1 | trajectory 전체 예측 tool 레이블 union vs gold support shard 집합 |
| citation_support.em / precision / recall / f1 | 답변에 인용한 문서 title 집합 vs gold supporting title 집합 |
| final_context_support.* | 최종 agent 입력 문서 title 집합 vs gold supporting title 집합 |
| traces[].pool_support / visible_support | 각 스텝의 검색 후보 coverage와 전달 후 coverage |
| input_tokens / output_tokens | agent + selector의 API usage 합계 평균 |
| agent_input_tokens / selector_input_tokens | 역할별 입력 토큰 평균 |
| llm_calls / tool_calls | LLM 호출 횟수와 실제 BM25 호출 횟수 평균 |
| wall_seconds | episode wall-clock 평균, selector·검색·재시도 포함 |
| errors | 실패 episode 수; 주 지표 분모에서 제외하지 않음 |
| paired.selected_minus_all_em | 같은 질문의 SELECTED−ALL 답변 EM 차이 |
| paired.bootstrap_95_ci | 질문 단위 paired bootstrap 2,000회, EM 차이 CI |

support 지표는 문서 title 수준이다. HotpotQA 공식 sentence-level supporting fact EM과 다르다. RRF pool coverage가 부족하면 검색 실패, pool에는 gold가 있지만 visible에서 빠지면 selection 손실로 분리해서 볼 수 있다. 도구 EM은 실패 episode에서 호출이 일부 진행된 상태도 포함하는 진단 지표다. 실패 자체는 errors로 별도 표시한다.

## 권장 비교

같은 n/seed/model/steps에서 top-k만 변경한다:

```bash
python3 benchmark.py run --topk 5 --out results_k5
python3 benchmark.py run --topk 10 --out results_k10
python3 benchmark.py run --topk 20 --out results_k20
```

각 실행은 ALL과 SELECTED 둘 다 수행한다. 먼저 n=100, 이후 seed를 바꾼 데이터 디렉터리로 반복하거나 n=500으로 늘린다. 질문 순서는 고정 seed로 shuffle하며 조건 실행 순서를 질문마다 교대한다. 모델의 서버 측 비결정성은 남아 있으므로 근소한 차이는 반복 실행으로 확인한다. API pricing은 하드코딩하지 않는다.

이 비교는 다음 tool/query도 context에 따라 달라지는 **end-to-end 정책 비교**다. 두 조건의 retrieval trajectory가 달라지는 것은 의도된 결과다. '똑같은 후보에서 selector만 바꾼 인과 효과'는 별도의 고정 retrieval replay 실험이 필요하며 여기서는 제공하지 않는다.

판단 기준: 답변 EM과 CI를 먼저 보고, final context gold recall 손실, tool recall, 총 토큰·wall time을 함께 비교한다. SELECTED의 agent 토큰만 낮아졌다고 비용 절감이라고 결론내리지 않는다.

## 출력과 검증

`results_*/summary.json`, `config.json`, `<question-id>.all.json`, `<question-id>.selected.json`을 생성한다. episode 파일에 각 스텝 tool prediction, RRF pool, selected indices, answer, gold, usage, error가 남는다. 결과 분석 시 summary와 실패 episode를 함께 확인한다.

```bash
python3 -m unittest discover -s tests -v
```

패키지 검증은 외부 API를 호출하지 않는 mock LLM로 multi-label 호출 → RRF → selector → answer 전체 흐름과 지표를 확인한다. 실제 gpt-6-luna 실행 결과/성능 수치는 포함하지 않는다.
