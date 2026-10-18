"""
Plagiarism highlighting engine: Dark luxury cyber styling with glowing neon tiers.
"""

import html
import re

def _tier(similarity: float) -> tuple[str, str, str, str]:
    """Returns (background, text_color, border_color, label)."""
    if similarity >= 0.90:
        return "rgba(239, 68, 68, 0.28)", "#FCA5A5", "#EF4444", "Very High"
    if similarity >= 0.75:
        return "rgba(249, 115, 22, 0.28)", "#FDBA74", "#F97316", "High"
    return "rgba(234, 179, 8, 0.25)", "#FDE047", "#EAB308", "Moderate"


def _escape(text: str) -> str:
    return html.escape(text)


def build_highlight_html(
    full_text: str,
    matches: list[dict],
    doc_key: str,          # "a" or "b"
    doc_label: str,        # displayed above the block
    accent_color: str,     # CSS colour for the document label bar
) -> str:
    sentence_key = f"sentence_{doc_key}"

    highlight_map: dict[str, float] = {}
    for m in matches:
        sent = m[sentence_key]
        highlight_map[sent] = max(highlight_map.get(sent, 0.0), m["similarity"])

    sorted_phrases = sorted(highlight_map.keys(), key=len, reverse=True)

    text = full_text
    intervals: list[tuple[int, int, float]] = []

    for phrase in sorted_phrases:
        sim = highlight_map[phrase]
        for m in re.finditer(re.escape(phrase), text, re.IGNORECASE):
            s, e = m.start(), m.end()
            if any(s < ie and e > is_ for is_, ie, _ in intervals):
                continue
            intervals.append((s, e, sim))

    intervals.sort(key=lambda t: t[0])

    parts = []
    cursor = 0
    for rank, (start, end, sim) in enumerate(intervals, 1):
        if cursor < start:
            parts.append(_escape(text[cursor:start]))
        bg, text_color, border_color, tier_label = _tier(sim)
        parts.append(
            f'<mark style="background:{bg};color:{text_color};border-bottom:2px solid {border_color};'
            f'border-radius:4px;padding:2px 3px;position:relative;text-shadow:0 0 8px {bg};">'
            f'<span style="display:inline-block;font-size:0.7rem;font-weight:800;'
            f'color:{border_color};vertical-align:super;line-height:1;margin-right:3px;">'
            f'#{rank}</span>'
            f'{_escape(text[start:end])}'
            f'</mark>'
        )
        cursor = end

    if cursor < len(text):
        parts.append(_escape(text[cursor:]))

    inner = "".join(parts)

    return (
        f'<div style="border-left:4px solid {accent_color};border-radius:14px;'
        f'background:rgba(15, 23, 42, 0.75);border:1px solid rgba(255, 255, 255, 0.08);'
        f'box-shadow:0 10px 25px rgba(0,0,0,0.3);padding:18px 20px;font-size:0.95rem;line-height:1.75;'
        f'white-space:pre-wrap;word-break:break-word;font-family:system-ui,sans-serif;color:#F1F5F9;">'
        f'<div style="font-size:0.8rem;font-weight:800;color:{accent_color};'
        f'margin-bottom:12px;letter-spacing:0.05em;text-transform:uppercase;">{_escape(doc_label)}</div>'
        f'{inner}'
        f'</div>'
    )


def build_legend_html() -> str:
    tiers = [
        ("rgba(234, 179, 8, 0.25)", "#FDE047", "#EAB308", "Moderate (50–74%)"),
        ("rgba(249, 115, 22, 0.28)", "#FDBA74", "#F97316", "High (75–89%)"),
        ("rgba(239, 68, 68, 0.28)", "#FCA5A5", "#EF4444", "Very High (90%+)"),
    ]
    items = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:7px;'
        f'margin-right:20px;font-size:0.84rem;color:#E2E8F0;font-weight:600;">'
        f'<span style="display:inline-block;width:14px;height:14px;'
        f'background:{bg};border-radius:4px;border:1.5px solid {bc};box-shadow:0 0 6px {bc};"></span>'
        f'{label}</span>'
        for bg, tc, bc, label in tiers
    )
    return (
        '<div style="display:flex;flex-wrap:wrap;align-items:center;gap:6px;'
        'padding:12px 18px;background:rgba(15, 23, 42, 0.7);border:1px solid rgba(255, 255, 255, 0.08);'
        'border-radius:12px;margin-bottom:16px;box-shadow:0 4px 12px rgba(0,0,0,0.2);">'
        '<span style="font-size:0.75rem;font-weight:800;color:#94A3B8;letter-spacing:0.06em;margin-right:12px;">'
        'HIGHLIGHT KEY</span>'
        + items +
        '</div>'
    )


def build_match_index_html(matches: list[dict]) -> str:
    if not matches:
        return '<p style="color:#94A3B8;font-size:0.9rem;">No matches above threshold.</p>'

    rows = ""
    for rank, m in enumerate(matches, 1):
        bg, text_color, border_color, tier_label = _tier(m["similarity"])
        rows += (
            f'<tr style="border-bottom:1px solid rgba(255,255,255,0.06);">'
            f'<td style="padding:10px 14px;vertical-align:top;white-space:nowrap;">'
            f'<span style="display:inline-block;background:{bg};color:{text_color};border:1px solid {border_color};'
            f'font-weight:800;font-size:0.75rem;padding:3px 10px;border-radius:999px;box-shadow:0 0 8px {bg};">'
            f'#{rank} · {tier_label}</span><br>'
            f'<span style="font-size:0.8rem;color:#94A3B8;font-weight:700;margin-top:4px;display:inline-block;">{m["similarity"]:.0%}</span>'
            f'</td>'
            f'<td style="padding:10px 14px;vertical-align:top;font-size:0.9rem;color:#F1F5F9;'
            f'border-left:3px solid #38BDF8;background:rgba(56, 189, 248, 0.05);line-height:1.6;">'
            f'{_escape(m["sentence_a"])}</td>'
            f'<td style="padding:10px 14px;vertical-align:top;font-size:0.9rem;color:#F1F5F9;'
            f'border-left:3px solid #F43F5E;background:rgba(244, 63, 94, 0.05);line-height:1.6;">'
            f'{_escape(m["sentence_b"])}</td>'
            f'</tr>'
        )

    return (
        '<div style="background:rgba(15, 23, 42, 0.7);border:1px solid rgba(255,255,255,0.08);border-radius:14px;overflow:hidden;box-shadow:0 8px 24px rgba(0,0,0,0.3);">'
        '<table style="width:100%;border-collapse:collapse;font-family:system-ui,sans-serif;">'
        '<thead><tr>'
        '<th style="padding:10px 14px;text-align:left;font-size:0.75rem;'
        'color:#94A3B8;font-weight:700;background:rgba(15,23,42,0.9);border-bottom:1px solid rgba(255,255,255,0.08);letter-spacing:0.04em;">'
        'MATCH TIER</th>'
        '<th style="padding:10px 14px;text-align:left;font-size:0.75rem;'
        'color:#38BDF8;font-weight:800;background:rgba(15,23,42,0.9);border-bottom:1px solid rgba(255,255,255,0.08);letter-spacing:0.04em;">'
        '📄 DOCUMENT A</th>'
        '<th style="padding:10px 14px;text-align:left;font-size:0.75rem;'
        'color:#F43F5E;font-weight:800;background:rgba(15,23,42,0.9);border-bottom:1px solid rgba(255,255,255,0.08);letter-spacing:0.04em;">'
        '📄 DOCUMENT B</th>'
        '</tr></thead>'
        f'<tbody>{rows}</tbody>'
        '</table>'
        '</div>'
    )
