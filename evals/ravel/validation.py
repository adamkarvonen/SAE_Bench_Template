import datetime
import re
from typing import Optional
from zoneinfo import ZoneInfo


def _timezone_name_to_utc_offset(name: str) -> Optional[str]:
    """
    Convert a timezone name to its UTC offset.

    Args:
        name (str): Timezone name.

    Returns:
        Optional[str]: UTC offset as a string, or None if conversion fails.
    """
    try:
        offset = ZoneInfo(name).utcoffset(datetime.datetime.now()).seconds
        sign = "+" if offset < 12 * 3600 else "-"
        if offset >= 12 * 3600:
            offset = 24 * 3600 - offset
        fmt_offset = str(datetime.timedelta(seconds=offset)).rsplit(":", 1)[0]
        if fmt_offset.startswith("0") and offset >= 1800:
            fmt_offset = fmt_offset[1:]
        return f"{sign}{fmt_offset}"
    except Exception:
        return None

def _is_summer_dst_case(norm_label: str, label: str) -> bool:
    """Check if the case is a summer daylight saving time scenario."""
    return (re.search(r"\-[5-8]", norm_label) and label.startswith("America")) or (
        re.search(r"\+[0-3]", norm_label)
        and (label.startswith("Europe") or label.startswith("Africa"))
    )


def _evaluate_utc_completion(label: str, norm_out: str) -> bool:
    """Helper method to evaluate UTC-related completions."""
    norm_label = _timezone_name_to_utc_offset(label)
    if not norm_label:
        return False

    correct = norm_out.startswith(norm_label.split(":")[0])
    if not correct and re.search(r"[+\-]0\d", norm_out):
        correct = norm_out.replace("0", "", 1).startswith(norm_label.split(":")[0])

    # Handle summer daylight saving time
    if not correct and _is_summer_dst_case(norm_label, label):
        out_offset_match = re.search(r"[+\-]?(\d\d?):\d+", norm_out)
        label_offset_match = re.search(r"[+\-]?(\d\d?):\d+", norm_label)
        if out_offset_match and label_offset_match:
            norm_out_offset = int(out_offset_match.group(1))
            norm_label_offset = int(label_offset_match.group(1))
            correct = (
                norm_out_offset <= norm_label_offset + 1
                and norm_out_offset >= norm_label_offset - 1
            )

    if (
        not correct
        and re.search(r"[+\-](\d+)", norm_out)
        and int(re.search(r"[+\-](\d+)", norm_out).group(1)) > 11
    ):
        offset = 24 - int(re.search(r"[+\-](\d+)", norm_out).group(1))
        correct = str(offset) in norm_label

    return correct


def evaluate_completion(
    text: str,
    expected_label: str,
    completion: str,
    # prompt: Prompt,
) -> bool:
    """
    Evaluate if a completion is correct for a given text w.r.t. a label.
    """
    # expected_label = self.entity_attributes[prompt.entity][prompt.attribute]
    if not expected_label:
        return False

    norm_label = expected_label.lower()
    norm_out = completion.split('"')[0].strip(' "').replace("\\/", "/").lower()
    if not norm_out:
        return False

    correct = (
        norm_out.startswith(norm_label)
        if len(norm_label) < len(norm_out)
        else norm_label.startswith(norm_out)
    )

    # Handle special cases
    if "coord" in text or "latitude" in text or "longitude" in text:
        try:
            correct = (
                abs(float(norm_label.strip("-−")) - float(re.findall(r"\d+", norm_out)[0])) <= 2
            )
        except:
            correct = False
    elif any(country in expected_label for country in ["United States", "United Kingdom"]):
        norm_label = expected_label.strip().replace("the ", "")
        norm_out = completion.strip().replace("the ", "")
        correct = norm_out.startswith(norm_label) or norm_out.startswith("England")
    elif "South Korea" in expected_label:
        correct = norm_out.startswith("korea") or norm_out.startswith("south korea")
    elif "North America" in expected_label:
        correct = norm_label in norm_out or norm_out == "na" or norm_out.startswith("america")
    elif "Mandarin" in expected_label:
        correct = norm_out in norm_label or norm_out == "chinese"
    elif "language" in text and "," in norm_label:
        correct = any(lang in norm_out for lang in norm_label.split(","))
    elif "UTC" in text and "/" in norm_label:
        correct = _evaluate_utc_completion(expected_label, norm_out)

    return correct
