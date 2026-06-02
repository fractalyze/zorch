1. jax 와 whir-zorch 사이에 있는 repo zorch 를 만들거야
2. Modern SNARK = IOP + PCS
3. zorch 는 딥러닝에서 Layer 로 빌딩 블록을 쌓듯이, Round 로 빌딩 블록을 쌓으려고 해
4. 각 Round는 Fiat-Shamir interface 가 있어
   1. commit 을 통해서 absorb 할 수 있고
   2. challenge 를 통해서 squeeze 할 수 있어
5. Polynomial 을 어떻게 design 할까?
   1. Polynomial 은 일단 아래 처럼 있어
      1. Univaraite poly
      2. Multilinear poly
6. 또한 PCS 를 만들수 있는 building block 이 있어
   1. 어떻게 commit 하고
   2. 어떻게 open, verify 하는지 있어
7. 그리고 Challenge 를 만들수 있는 building block 이 있어
   1. 어떻게 abosrb
   2. 어떻게 squeeze
8. 적어도 Round 는 다 fusion 되어야 해
9. Challenge 에서도 absort, squeeze 다 fusion 되어야 해
10. Commit, Open 도 마찬가지
11. 이런 building block을 바탕으로 whir-zorch 에 있는 poseidon2 부터 migration 하고 싶어
12. poseidon2는 현재 zkx에서 패턴 매칭 후 custum fusion 으로 code generation을 하는데 이걸 안하고 싶어
    1.  poseidon2와 같은 해시 함수들은 안에도 round(여기서 round는 IOP의 round와는 다름)로 이루어져있어
    2.  각 round 를 포함해서 전체 round, 혹은 permutation 은 하나로 fusion 되어야 하는데 지금은 reduce, 등으로 인해 fusion 이 안되.
    3.  어떤 hint를 주거나? zkx는 어떤 knowledge가 있어야 모두를 하나의 fusion 으로 묶을 수 있을까?
13. Modern SNARK에서 문제크기를 줄여주는 방법 중 하나가 2-to-1 fold 야
    1.  그 말은 즉슨 프로그램은 같은데 input만 1/2 씩 주는거지
    2.  best는 줄일 때마다 random challenge를 받아야 하니 적어도 각 round마다는 하나로 fusion 되어야해
14. 그리고 zero check를 할 때 constraint는 외부에서 주입이 될거야.
15. 그리고 일단 zorch 는 zkVM 을 모르게 작성하고 싶어.
16. 이거는 빌딩블록이니 만큼 조립하기 쉽게 설계되어야해.
17. 이 repo를 처음부터 셋업할거니 ~/Workspace/riscv-witness 에서 잘했던 것을 답습하고 싶고
    1.  .claude/
    2.  CLAUDE.md
    3.  README.md
    4.  docs/*.md
18. 느리지만 working 하는 ~/Workspace/whir-zorch에서의 코드도 가져오고 싶어

