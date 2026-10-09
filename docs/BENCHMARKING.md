# Benchmarking

## PBQ

The PBQ benchmark runs one generated PBQ through every configured runtime profile.

For each profile it records:

- runtime profile and requested difficulty;
- candidate workspace tree;
- reference workspace tree;
- reference validation/build/test command output;
- private-test mutation score;
- difficulty judgement;
- repair count;
- timing and model-call metrics;
- final status.

The reference must pass its runtime commands before a PBQ can be considered successful. If local tooling is missing, such as `vite`, `vitest`, `jest`, or `mvn`, the result is recorded as an infrastructure/runtime failure.

Mutation testing creates intentionally incorrect variants and runs the private tests against them. A mutation is counted only when the test command ran conclusively. Missing tools are marked inconclusive rather than counted as a caught mutation.

## MCQ

The MCQ benchmark runs one easy, one medium, and one hard MCQ.

For each question it records:

- difficulty blueprint;
- generated question/options/answer;
- independent solver answer;
- answer agreement;
- distractor score;
- difficulty judgement;
- model-call metrics;
- final status.

