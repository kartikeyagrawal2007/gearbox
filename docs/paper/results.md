# Draft: Results (§5.1–5.2), HumanEval+

*Draft 1, 2026-10-05.* Numbers come from the A5000 runs (`docs/paper/data/humaneval_interim.json`, transcribed from `bench/results.py`). The pass counts are exact; the false-done rates are rounded to whole percents. **Before submission:**
- regenerate everything from the full result files (`python bench/results.py --export`)
- add the hatch-off numbers for the remaining 8 models and the MBPP+ results
- replace the rough uncertainty below with paired tests

Figures: `docs/paper/figures/humanevalplus_{size_ladder,vendors,hatch}.pdf`.

---

## 5.1 Harness validation

Our harness reproduces published scores to within about one point. With Qwen2.5-Coder-1.5B-Instruct (Ollama 4-bit, greedy decoding), we measure 65.2% on HumanEval+ and 58.7% on MBPP+. The Qwen2.5-Coder technical report gives 66.5% and 59.4%. The remaining gap is consistent with quantization and prompt differences.

Grading follows the official EvalPlus evaluator: we copy its input deserialization and special oracles verbatim. Every one of the 542 reference solutions passes its own generated check.

## 5.2 How reliable are a cheap worker's "done"s? (RQ2)

**Setup.** 12 open models answer all 164 HumanEval+ problems as delegated workers, at temperature 0, with hidden tests. An answer is *claimed done* unless the model replies UNSURE. The **false-done rate** is the share of claimed-done answers whose code is defined but wrong; format failures (the requested function missing) are counted separately. With 164 problems, a pass rate near 50% carries roughly ±7.5 points of 95% uncertainty, and roughly ±6 points near 80%.

**Table 2: HumanEval+, UNSURE hatch on.**

| Model | Params | Passed | False-done | Unsure | Format | Fences repaired |
|---|---|---|---|---|---|---|
| Qwen3.5 | 0.8B | 38/164 (23%) | 70% | 8 | 29 | 2 |
| Qwen3.5 | 2B | 83/164 (51%) | 48% | 0 | 3 | 4 |
| Qwen3.5 | 4B | 126/164 (77%) | 23% | 0 | 1 | 0 |
| Qwen3.5 | 9B | 132/164 (80%) | 20% | 0 | 0 | 4 |
| Qwen3.5 | 27B | 152/164 (93%) | 7% | 0 | 0 | 0 |
| Granite 4.2 | 3B | 121/164 (74%) | 23% | 0 | 5 | 0 |
| Granite 4.2 | 8B | 143/164 (87%) | 10% | 0 | 2 | 0 |
| Ministral 3 | 3B | 106/164 (65%) | 34% | 0 | 3 | 0 |
| Ministral 3 | 8B | 130/164 (79%) | 21% | 0 | 0 | 38 |
| Gemma 3 | 4B | 104/164 (63%) | 36% | 0 | 1 | 0 |
| Gemma 3 | 12B | 127/164 (77%) | 23% | 0 | 0 | 0 |
| Qwen2.5-Coder (older) | 0.5B | 90/164 (55%) | 45% | 0 | 1 | 0 |

**Finding 1: reliability rises steeply with size, then flattens (Figure 1).** Across the Qwen3.5 ladder, the pass rate rises from 23% (0.8B) to 93% (27B), and the false-done rate falls from 70% to 7%. Between 4B and 9B the curve is flat: 77% vs 80% is within noise. Even the 27B presents 12 wrong answers as finished. So size reduces false dones but does not remove them.

**Finding 2: current models almost never abstain.** Across the 11 current models (1,804 delegated problems), only the smallest, Qwen3.5 0.8B, ever answered UNSURE, and only 8 times, while 70% of the answers it did claim were wrong. The other ten never abstained, including on the 12–34% of answers they got wrong. This extends the false-success findings for frontier agents (2606.09863) to small open models, and it has a direct consequence for delegation: **escalation schemes that wait for a worker to ask for help would almost never trigger.** An external, executable check is needed.

**Finding 3: offering the escape hatch helps nobody, and hurts the smallest model (Figure 3).** With the hatch off, Qwen3.5's pass rate is unchanged within noise at 2B, 4B and 9B, so they never needed it. For the 0.8B, offering the hatch *lowered* the pass rate (23% vs 27%) and tripled format failures (29 vs 9). The extra instruction degraded its output rather than making it honest. This is consistent with abstention behaving as a prompt artifact (2507.16199). *Pending: hatch-off for the 27B and the other vendors.*

**Finding 4: vendor matters as much as size (Figure 2).** At matched sizes, the spread across vendors is comparable to a doubling of parameters within one family:
- **Small (3–4B):** Qwen3.5 4B 77%, Granite 3B 74%, Ministral 3B 65%, Gemma 4B 63%.
- **Mid (8–12B):** Granite 8B 87% with a 10% false-done rate, the best mid-size model, ahead of Gemma's larger 12B (77%).

*Significance to confirm with paired McNemar tests on per-problem outcomes; Granite 8B vs Qwen3.5 9B (87% vs 80%) is the comparison that most needs it.*

**Finding 5: code specialization beats newer and larger general models at the low end.** The year-older, code-tuned Qwen2.5-Coder 0.5B (55%) outperforms Qwen3.5 0.8B (23%) and roughly matches Qwen3.5 2B (51%), despite being 4× smaller. For cheap workers on code subtasks, specialization matters more than generation or size.

**Finding 6: output format alone can decide measured accuracy.** Ministral 3 8B left its code fences unbalanced in 38 of 164 answers. A strict parser would have scored all of those as failures. Under fence repair it reaches 79%. We report fence repairs and format failures separately, so formatting slips are visible without being counted as wrong code.

---

## Notes for the final version

- Report Wilson 95% intervals on every pass rate, plus McNemar tests for the pairs we discuss.
- The false-done rate is close to "1 − pass rate among claimed answers", because abstention is so rare. State this explicitly, so readers don't think it's a separate measurement. The interesting part is the abstention rate.
- MBPP+ (378 problems) will either replicate Findings 1–6 or show where they don't hold. Both outcomes are reportable.
