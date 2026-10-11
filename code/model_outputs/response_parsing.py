"""
Pure-Python response parsers shared across every inference backend
(local Qwen2.5-VL, Qwen3-VL, and API-based models via OpenRouter).

Split out of run_qwen_inference.py 2026-08-11: that file also defines the
local Qwen2.5-VL model-loading code and imports torch at module level, so any
caller that only needed the parser (the OpenRouter and Qwen3-VL paths, which
never touch the local model) was forced to have torch installed too -- broke
running the OpenRouter path on a machine with no GPU/conda env, which is
exactly where it's meant to run (see run_e2_batch_frontier.py's docstring).
"""
import re


def parse_quantity_calibration(raw_text):
    cal_match = re.search(r"CALIBRATION_MM_PER_TICK:\s*([\d.]+)", raw_text)
    vel_match = re.search(r"VELOCITY_MPS:\s*([\d.]+)", raw_text)
    return {
        "calibration_mm_per_tick": float(cal_match.group(1)) if cal_match else None,
        "velocity_mps": float(vel_match.group(1)) if vel_match else None,
        "parse_failed": cal_match is None or vel_match is None,
    }


def parse_velocity_only(raw_text):
    """For the temporal-relabeling condition (prompt_design_temporal.py),
    which asks only for VELOCITY_MPS -- the asserted frame rate is given by
    the prompt itself, not read from the response."""
    vel_match = re.search(r"VELOCITY_MPS:\s*([\d.]+)", raw_text)
    return {
        "velocity_mps": float(vel_match.group(1)) if vel_match else None,
        "parse_failed": vel_match is None,
    }


def parse_hidden_prior(raw_text):
    """
    DEPRECATED for E3 scoring: this prompt alone cannot distinguish a genuine
    video-grounded decline from a content-independent default "UNKNOWN" (see
    memory project_sgf_e3_results.md -- the same "UNKNOWN" was returned with NO
    video attached at all). Kept only for reference; use
    parse_hidden_prior_with_control instead.
    """
    m = re.search(r"RELEASE_HEIGHT_CM:\s*(UNKNOWN|[\d.]+)", raw_text, re.IGNORECASE)
    if m is not None:
        val = m.group(1)
        if val.upper() == "UNKNOWN":
            return {"declined": True, "asserted_value_cm": None, "parse_failed": False}
        return {"declined": False, "asserted_value_cm": float(val), "parse_failed": False}

    stripped = raw_text.strip()
    if re.fullmatch(r"UNKNOWN\.?", stripped, re.IGNORECASE):
        return {"declined": True, "asserted_value_cm": None, "parse_failed": False}
    bare_num = re.fullmatch(r"([\d.]+)\s*(cm)?\.?", stripped, re.IGNORECASE)
    if bare_num:
        return {"declined": False, "asserted_value_cm": float(bare_num.group(1)), "parse_failed": False}

    return {"declined": None, "asserted_value_cm": None, "parse_failed": True}


def parse_hidden_prior_with_control(raw_text):
    control_match = re.search(r"CONTROL_IMPACT:\s*(YES|NO|UNKNOWN)", raw_text, re.IGNORECASE)
    height_match = re.search(r"RELEASE_HEIGHT_CM:\s*(UNKNOWN|[\d.]+)", raw_text, re.IGNORECASE)

    control_answer = control_match.group(1).upper() if control_match else None
    control_correct = control_answer == "YES"

    if height_match is None:
        return {
            "control_answer": control_answer, "control_correct": control_correct,
            "declined": None, "asserted_value_cm": None, "parse_failed": True,
        }
    val = height_match.group(1)
    declined = val.upper() == "UNKNOWN"
    return {
        "control_answer": control_answer, "control_correct": control_correct,
        "declined": declined,
        "asserted_value_cm": None if declined else float(val),
        "parse_failed": False,
    }
