"""
Improvement plan B2: investigate why Qwen's E2 calibration-grounding (CG) is
so much lower on 05052026 (0/54, CG=0.0) than 02182026 (24/54, CG=0.44), and
whether this shares a root cause with E3's session asymmetry (05052026 also
failed the E3 control question far more often, 1/6 vs 6/6).

Answer, established by inspecting raw calibration answers directly: this is
a DIFFERENT root cause from E3's, not the same one. E3's problem was footage
visibility (impact surface compressed into a low-contrast strip). This is a
task-comprehension error specific to how 05052026's ruler is labeled: Qwen
echoes the PRINTED NUMBER on the ruler ("30") as if it were the answer to
"how many mm does one tick represent", rather than reasoning about the tick
spacing. On 02182026 (ruler printed "12"), Qwen's wrong guesses are still
small, plausible mm-per-tick values (0.5 or 1.0) that sometimes coincidentally
match the true answer (1.0) -- a "guessing a plausible small number" error,
not a "misread the label as the answer" error. Confirmed deterministically:
18/18 base-condition 05052026 answers are exactly 30.0 or 3.0 (all sessions'
true calibration is 1.0 mm/tick); at alpha=0.5 (relabeled "15"), 18/18 answers
are exactly 3.75 = 15/4, a fixed transform of the printed digit, still not the
correct 0.5.
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sgf_paths  # noqa: E402  (code/sgf_paths.py: all repo paths)

with open(os.path.join(sgf_paths.RESULTS_DIR, "e2", "e2_qwen_results.json")) as f:
    qwen = json.load(f)
with open(os.path.join(sgf_paths.RESULTS_DIR, "e2", "e2_gemini_flash_lite_results.json")) as f:
    gemini = json.load(f)


def cg_by_session(results, name):
    from collections import defaultdict
    by_session = defaultdict(list)
    for x in results:
        cal = x.get("calibration_mm_per_tick")
        hit = (cal is not None and
               abs(cal - x["asserted_calibration_mm_per_tick"]) / x["asserted_calibration_mm_per_tick"] <= 0.10)
        by_session[x["session"]].append(hit)
    print(f"\n=== {name}: CG by session ===")
    for s, hits in by_session.items():
        print(f"  {s}: CG={sum(hits)/len(hits):.3f} ({sum(hits)}/{len(hits)})")


def value_distribution(results, session, condition, name):
    from collections import Counter
    sub = [x for x in results if x["session"] == session and x["condition"] == condition]
    cals = [x.get("calibration_mm_per_tick") for x in sub]
    asserted = sub[0]["asserted_calibration_mm_per_tick"] if sub else None
    print(f"  {name} session={session} cond={condition} (true={asserted}): {Counter(cals)}")


cg_by_session(qwen, "Qwen2.5-VL-7B-Instruct")
cg_by_session(gemini, "gemini-2.5-flash-lite")

print("\n=== Root cause check: does Qwen echo the printed ruler digit? ===")
value_distribution(qwen, "05052026", "base", "Qwen")
value_distribution(qwen, "05052026", "0.5", "Qwen")
value_distribution(qwen, "05052026", "2", "Qwen")
print("  (05052026's ruler prints '30' at base, '15' at alpha=0.5, '60' at alpha=2 -- "
      "30.0/3.0, 3.75=15/4, and 60.0 all show the model anchoring to the PRINTED LABEL, "
      "not computing true mm-per-tick)")

value_distribution(qwen, "02182026", "base", "Qwen")
print("  (02182026's ruler prints '12' at base -- 0.5/1.0 are plausible small mm-per-tick "
      "GUESSES, not echoes of '12'; qualitatively different error)")

print("\nConclusion: E2's CG session asymmetry and E3's control-question session asymmetry "
      "are TWO DIFFERENT root causes that happen to affect the same session. E3 = footage "
      "visibility (impact surface framing). E2/CG = task-comprehension (echoing the printed "
      "ruler label instead of reasoning about tick spacing), likely specific to '30' being a "
      "more 'answerable-looking' number than '12' rotated 90 degrees. Gemini shows a much "
      "smaller session gap (0.43 vs 0.33) -- consistent with this being a Qwen-specific "
      "misreading tendency, not a property of the footage itself (which is what E3's cause was).")
