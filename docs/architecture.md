# zorch 아키텍처 — fusion 계약과 미해결 설계 결정

> 비전·building block 목록·설계 원칙은 [`../README.md`](../README.md)에 있다.
> 이 문서는 README가 한 줄로만 담는 **fusion 계약**의 근거와, 아직 확정 전인
> **설계 결정**만 다룬다. 상태: **초안 (bootstrap).**

## 1. Fusion 계약

각 `Round`, 각 `commit`/`open`, 각 `absorb`/`squeeze`, 각 fold step, 해시
permutation의 내부 round는 **하나의 fused kernel**로 내려가야 한다 — 컴파일러의
primitive별 pattern-match가 아니라 *구성(by construction)*으로.

### 1.1 지금(zkx)의 방식 — pattern matching

`poseidon2`는 현재 zkx가 패턴 매칭 후 custom codegen으로 처리한다:
`Poseidon2PermuteRewriter`(HLO transform, `gpu_compiler.cc`의 `poseidon2-pre-inline`
단계)가 외부-내부-외부 3개의 `while` 패턴과 상수를 인식해 `kCustom` fusion으로
치환하고, `Poseidon2Fusion` emitter가 permutation 전체를 field op로 한 커널에
emit한다. `whir-zorch`의 `poseidon2/poseidon2.py`가 `lax.while_loop` 3개 +
`jnp.dot(mds)` 모양인 것은 *이 패턴매처에 인식되기 위해서*다(코드 주석에 명시).
문제: 해시마다 인식 코드를 새로 짜야 하고, 모양이 조금만 달라도 실패하며, 커널도
수작업이다. 이걸 그만두려는 것이 zorch의 동기다.

### 1.2 왜 `reduce`가 fusion을 깨는가

MDS는 `jnp.dot(mds, s)` — HLO에서 `reduce`/`gather`가 되고 이는 **`kInput`**
fusion을 강제한다. `kInput`은 주변 element-wise **`kLoop`** fusion과 합쳐질 수
없어 **fusion 경계(벽)**가 생긴다. 이것이 "reduce 때문에 fusion이 안 된다"의
정체다. (`prime-ir/opt_attempts.md`: MDS의 reduce가 `kInput`의 지배적 원인;
full matmul로 `kInput` 166→15로 줄였지만 연산량이 +40~91% 폭증.)

### 1.3 검증된 GPU 파이프라인 사실

zkx 소스를 직접 확인한 결과:

- GPU 경로: `JAX → StableHLO → zkx HLO → XLA instruction fusion
  (PriorityFusion: kLoop/kInput/kCustom) → MLIR emitter → LLVM → PTX`.
- **prime-ir는 GPU 경로에서 타입 변환만** 한다(`Field→ModArith→Arith`,
  `EllipticCurve→Field`). **`affine-loop-fusion`은 GPU에서 돌지 않는다** — 그것은
  **CPU 백엔드 전용**이다(`cpu_kernel_emitter`). `prime-ir-fix-loop-fusion`의
  visited-set hang 수정과 "~28s에 한 커널"도 CPU 이야기였다.
- GPU에서 `while_loop`은 `kWhile` → `WhileThunk`로 남고, body는 iteration마다
  op별 kernel 시퀀스로 발사된다 — **iteration 간 fusion이 없다**.
- 즉 지금 `poseidon2`가 1커널인 이유는 generic fusion이 아니라 **전용 emitter**
  덕분이다.

### 1.4 보정된 결론

**순수 정규형(normal form)만으로는 GPU에서 "permutation = 1 kernel"이 불가능**
하다. 정규형은 직선(straight-line) 코드 내부의 `reduce` 경계를 없애는 데까지는
되지만, round를 loop로 돌리면 GPU에선 안 묶이고, unroll하면
`LAUNCH_OUT_OF_RESOURCES` + 컴파일 폭증에 걸린다.

따라서 **pattern matching 없이 1커널을 유지**하려면 — 이것이 원 요청 item 12.3
("zkx가 어떤 knowledge가 있어야 하나로 묶나?")의 답이다:

> 해시별 (matcher + emitter) N쌍을 → **일반 fused-region primitive 1개 + 일반
> emitter 1개**로 대체한다. zorch가 "이 region은 한 커널로 emit 되어야 할
> round-structured region"임을 *명시적으로* 건네주고, zkx는 그 표시를 받는
> generic emitter 하나만 둔다. body는 정규형으로 작성해 lowering을 깨끗이 한다.

이는 zkx 무변경이 아니라 **재사용 가능한 zkx 변경 1개를 확정**한다는 뜻이며, 그
generic emitter 작업은 별도 cross-repo 단계로 분리한다.

## 2. 미해결 설계 결정 (OPEN)

1. **Fusion 방향 확정.** §1.4의 "일반 fused-region primitive + 일반 emitter"로
   갈지, 아니면 zorch는 순수 JAX로 두고 GPU 1커널 fusion을 별도 후속 zkx 작업으로
   완전히 분리할지 — 사용자 확인 대기 중.
2. **primitive의 구체 형태.** `custom_call` vs 새 op vs `scf.for` + generic
   loop-kernel lowering. unroll vs 커널-내-loop. Triton 경로(`kTriton` fusion
   kind는 현재 미구현) 검토 포함.
3. **Proving-scheme 범위.** 1차 타깃은 IOP+PCS로 분해되는 Modern SNARK 일가
   (transparent 전부 + KZG형 PCS로 pairing 계열). Groth16 같은 R1CS/QAP 전용
   회로 형식을 1급으로 넣을지는 별도.
4. **Round API.** prover/verifier가 transcript 상호작용 기술을 공유해 drift를
   막을지(단일 기술), 분리할지.
