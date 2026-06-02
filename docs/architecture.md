# zorch 아키텍처

> 상태: **초안 (bootstrap).** 핵심 spine 설계가 진행 중이며, fusion 방향은
> 아직 최종 확정 전이다. "미해결 질문" 절을 참고할 것.

## 1. zorch란 무엇인가

`zorch`는 `jax`와 proof system(예: `whir-zorch`) 사이에 있는 **building block
라이브러리**다. 의존 방향은 `whir-zorch → zorch → jax/zkx`. JAX는 tracing과
codegen(ZKX = Fractalyze XLA fork, native finite-field dtype)을 담당하고,
`zorch`는 proof system을 조립하는 재사용 가능한 부품을 제공한다.

Modern SNARK는 **IOP + PCS**다. `zorch`는 이를 조립 가능한 블록으로 준다 —
딥러닝이 `Layer`로 쌓듯, `zorch`는 **`Round`**로 쌓는다.

## 2. 설계 원칙

1. **Proving-scheme-agnostic.** 블록은 특정 scheme 하나가 아니라 *모든* proving
   scheme을 포착해야 한다. `Round` / Fiat-Shamir / `Polynomial` / `PCS` / fold /
   zero-check 조합으로 FRI, sumcheck, GKR, STARK, Basefold, WHIR 등을 표현한다.
   pairing 기반 scheme은 `PCS` 블록을 KZG형으로 교체해 흡수한다.
2. **zkVM-agnostic.** `zorch`는 어떤 zkVM도 알지 못한다. zkVM 전용 지식이 블록에
   새어 들어가면 그것은 consumer(`whir-zorch` 등)의 몫이다.
3. **Fusion-first.** 각 `Round`, 각 `commit`/`open`, 각 `absorb`/`squeeze`, 각
   fold step, 그리고 해시 permutation의 내부 round는 **하나의 fused kernel**로
   내려가야 한다. 컴파일러의 primitive별 pattern-match가 아니라 *구성(by
   construction)*으로 달성한다.
4. **조립 용이성.** 빌딩 블록인 만큼 API는 "끼워 맞추기" 쉬움을 최우선한다.

## 3. Building blocks

| 블록            | 역할                                         | whir-zorch 출처             |
| --------------- | -------------------------------------------- | --------------------------- |
| `Round`         | Fiat-Shamir interface를 가진 조립 단위       | (신규 추상화)               |
| Fiat-Shamir     | duplex sponge transcript: `absorb`/`squeeze` | `challenger/`               |
| `Polynomial`    | univariate / multilinear 표현                | `poly/`, `multilinear/`     |
| `PCS`           | `commit` / `open` / `verify`                 | `commit/`, `merkle_tree/`   |
| Fold            | 2-to-1 reduction, round마다 random challenge | `basefold/`                 |
| Zero-check      | 외부에서 주입되는 constraint                 | `sumcheck/`                 |

## 4. `Round` 추상화

`Round`는 zorch의 핵심 조립 단위다. 각 `Round`는 **Fiat-Shamir interface**를
가진다.

- `commit(...)` → transcript에 **absorb**
- `challenge(...)` → transcript에서 **squeeze**

상태(state)는 JAX 관례대로 **functional + pytree**로 흐른다(불변, 새 상태를
반환). `whir-zorch`의 `challenger`가 이미 이 방식이며 `observe_and_sample`처럼
absorb+squeeze를 단일 JIT kernel로 묶는 형태를 prototype 해 두었다 — 단,
SP1 FFI / zkVM 전용 cruft는 zorch로 가져오지 않는다(원칙 2).

문제 크기를 줄이는 **2-to-1 fold**: 프로그램은 같고 입력만 절반씩 준다. 줄일
때마다 random challenge를 받으므로 적어도 round 단위가 하나로 fuse 되어야 한다.

## 5. Fusion 계약 (핵심)

### 5.1 지금(zkx)의 방식 — pattern matching

`poseidon2`는 현재 zkx가 **패턴 매칭 후 custom fusion codegen**으로 처리한다:

- `zkx`의 `poseidon2_permute_rewriter` (HLO transform, `gpu_compiler.cc`의
  `poseidon2-pre-inline` 단계)가 외부-내부-외부 3개의 `while` 패턴과 상수를
  인식 → `kCustom` fusion으로 치환.
- `Poseidon2Fusion` emitter가 permutation 전체를 field op로 **한 커널에 직접
  emit**.

`whir-zorch`의 `poseidon2/poseidon2.py`가 `lax.while_loop` 3개 +
`jnp.dot(mds)` 모양인 것은 *이 패턴매처에 인식되기 위해서*다(코드 주석에
명시). 문제: 해시마다 인식 코드를 새로 짜야 하고, 모양이 조금만 달라도 인식
실패하며, 커널도 수작업이다. 이걸 그만두려는 것이 zorch의 동기다.

### 5.2 왜 `reduce`가 fusion을 깨는가

MDS는 `jnp.dot(mds, s)` — HLO에서 `reduce`/`gather`가 되고 이는 **`kInput`
fusion**을 강제한다. `kInput`은 주변 element-wise **`kLoop`** fusion과 합쳐질
수 없어 **fusion 경계(벽)**가 생긴다. 이것이 "reduce 때문에 fusion이 안 된다"의
정체다. (`prime-ir/opt_attempts.md`: MDS의 reduce가 `kInput`의 지배적 원인;
full matmul로 `kInput` 166→15로 줄였지만 연산량이 +40~91% 폭증.)

### 5.3 검증된 GPU 파이프라인 사실

zkx 소스를 직접 확인한 결과(중요 — 초기 가정 수정):

- GPU 경로: `JAX → StableHLO → zkx HLO → XLA instruction fusion
  (PriorityFusion: kLoop/kInput/kCustom) → MLIR emitter → LLVM → PTX`.
- **prime-ir는 GPU 경로에서 타입 변환만** 한다(`Field→ModArith→Arith`,
  `EllipticCurve→Field`). **`affine-loop-fusion`은 GPU에서 돌지 않는다** —
  그것은 **CPU 백엔드 전용**이다(`cpu_kernel_emitter`). (`prime-ir-fix-loop-fusion`의
  visited-set hang 수정과 "~28s에 한 커널"도 CPU 이야기였다.)
- GPU에서 `while_loop`은 `kWhile` → `WhileThunk`로 남고, body는 iteration마다
  op별 kernel 시퀀스로 발사된다 — **iteration 간 fusion이 없다**.
- 즉 지금 `poseidon2`가 1커널인 이유는 generic fusion이 아니라 **전용
  emitter** 덕분이다.

### 5.4 보정된 결론

따라서 **순수 정규형(normal form)만으로는 GPU에서 "permutation = 1 kernel"이
불가능**하다. 정규형은 직선(straight-line) 코드 내부의 `reduce` 경계를 없애 한
`kLoop`로 묶는 데까지는 되지만, round를 loop로 돌리면 GPU에선 안 묶이고,
unroll하면 `LAUNCH_OUT_OF_RESOURCES` + 컴파일 폭증에 걸린다.

그러므로 **pattern matching 없이 1커널을 유지**하려면:

> 해시별 (matcher + emitter) N쌍을 → **일반 fused-region primitive 1개 +
> 일반 emitter 1개**로 대체한다. zorch가 "이 region은 한 커널로 emit 되어야 할
> round-structured region"임을 *명시적으로* 표시하고, zkx는 그 표시를 받는
> generic emitter 하나만 둔다. body는 정규형으로 작성해 generic emitter의 lowering을
> 깨끗하게 한다.

이것이 item 12.3("zkx가 어떤 knowledge가 있어야 하나로 묶나?")의 답이다:
*패턴으로 추론하게 두지 말고, region 구조를 zorch가 건네준다.* 단, 이는 zkx
무변경이 아니라 **재사용 가능한 zkx 변경 1개를 확정**한다는 뜻이다. 이 generic
emitter 작업은 별도 cross-repo 단계로 분리한다.

## 6. 첫 마일스톤 범위

**core spine + poseidon2.** 설계하는 것:

- `Round` 추상화
- Fiat-Shamir / Challenge (`absorb` / `squeeze`)
- `Polynomial` 타입 (univariate, multilinear)
- fusion 계약

검증: `whir-zorch`의 `poseidon2`를 zorch로 migration 하여, *pattern matcher 없이*
1커널로 fuse 되는지 end-to-end로 증명한다. PCS / Fold / Zero-check는 이후
사이클로 미룬다.

## 7. 미해결 질문 (OPEN)

1. **Fusion 방향 최종 확정.** §5.4의 "일반 fused-region primitive + 일반
   emitter" 방향을 채택할지, 아니면 zorch는 순수 JAX(A)로 두고 GPU 1커널 fusion을
   별도 후속 zkx 작업으로 완전히 분리할지 — 사용자 확인 대기 중.
2. **fused-region primitive의 구체 형태.** `custom_call` vs 새 op vs
   `scf.for` + generic loop-kernel lowering. unroll vs 커널-내-loop. Triton
   codegen 경로(현재 `kTriton` fusion kind는 미구현) 검토 포함.
3. **Proving-scheme 범위.** 1차 타깃은 IOP+PCS로 분해되는 Modern SNARK 일가
   (transparent 전부 + KZG형 PCS로 pairing 계열). Groth16 같은 R1CS/QAP 전용
   회로 형식을 1급으로 넣을지는 별도.
4. **Round API 세부.** prover/verifier가 transcript 상호작용 기술을 공유해
   drift를 막을지(단일 기술), 분리할지.
