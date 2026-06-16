"""Swiss bracket visualization for Masters London 2026 backtest page.

Renders a 3-round Swiss bracket using a CSS Grid table layout.
Match cards are placed in known grid cells — connector lines use table
cell borders, no pixel math. Cards are expandable via <details>.
"""

from __future__ import annotations

from html import escape
from typing import Any

# ── palette ────────────────────────────────────────────────────────────
R  = "#ff4655"
B  = "#38bdf8"
G  = "#86efac"
Y  = "#fbbf24"
BG = "#080b14"
S1 = "rgba(15,23,42,0.92)"
BD = "rgba(226,232,240,0.12)"
TX = "#f8fafc"
T2 = "#aab7cc"
T3 = "#75839a"

# ── CSS ────────────────────────────────────────────────────────────────
def _css() -> str:
    return f"""
<style>
.brkt-wrap {{ overflow-x:auto; padding:20px 16px 40px; }}
/* round title */
.brkt-rtitle {{ text-align:center; font-weight:850; font-size:0.78rem; letter-spacing:0.10em;
    text-transform:uppercase; color:{R}; padding:4px 0 6px;
    border-bottom:2px solid rgba(255,70,85,0.28); margin-bottom:8px; }}
.brkt-gtitle {{ font-size:0.64rem; font-weight:750; text-transform:uppercase; letter-spacing:0.06em;
    color:{T2}; padding:2px 8px; border-left:3px solid {B}; line-height:20px; }}
/* --- card --- */
.brkt-card {{ background:{S1}; border:1px solid {BD}; border-radius:8px; margin:4px 0; position:relative;
    transition:border-color 0.2s; }}
.brkt-card:hover {{ border-color:rgba(56,189,248,0.45); }}
.brkt-card.upset {{ border-color:rgba(255,70,85,0.55) !important; box-shadow:0 0 14px rgba(255,70,85,0.12); }}
.brkt-card.upset::after {{ content:"⚡ 冷门"; position:absolute; top:-10px; right:10px; background:{R}; color:{TX};
    font-size:0.56rem; font-weight:800; padding:2px 7px; border-radius:3px; letter-spacing:0.05em;
    text-transform:uppercase; z-index:5; font-family:system-ui,sans-serif; }}
.brkt-sum {{ cursor:pointer; list-style:none; display:block; padding:10px 12px; }}
.brkt-sum::-webkit-details-marker,.brkt-sum::marker {{ display:none; content:""; }}
.brkt-row {{ display:flex; align-items:center; gap:6px; }}
.brkt-team {{ flex:1; min-width:0; }}
.brkt-tname {{ font-weight:750; font-size:0.80rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.brkt-tname.win {{ color:{G}; }}
.brkt-tname.loss {{ color:{T2}; }}
.brkt-score {{ font-weight:850; font-size:0.92rem; text-align:center; min-width:32px; color:{TX}; }}
.brkt-pred {{ font-size:0.62rem; font-weight:700; padding:2px 6px; border-radius:3px; text-align:center;
    min-width:38px; white-space:nowrap; }}
.brkt-pred.ok {{ background:rgba(134,239,172,0.14); color:{G}; }}
.brkt-pred.bad {{ background:rgba(255,70,85,0.14); color:{R}; }}
.brkt-pred.close {{ background:rgba(251,191,36,0.14); color:{Y}; }}
/* detail */
.brkt-det {{ padding:0 12px 12px; font-size:0.72rem; color:{T2}; border-top:1px solid {BD}; margin:0 12px; }}
.brkt-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:3px 14px; margin-top:8px; }}
.brkt-dl {{ color:{T3}; font-size:0.64rem; text-transform:uppercase; letter-spacing:0.04em; }}
.brkt-dv {{ font-weight:700; color:{TX}; font-size:0.78rem; }}
.brkt-dv.big {{ font-size:1.02rem; }}
.brkt-bar {{ display:flex; height:4px; border-radius:2px; overflow:hidden; margin:6px 0; }}
.brkt-bar-a {{ background:{B}; }}
.brkt-bar-b {{ background:{R}; }}
.brkt-verdict {{ margin-top:6px; padding:5px 8px; border-radius:4px; font-weight:700; font-size:0.71rem; text-align:center; }}
.brkt-verdict.hit {{ background:rgba(134,239,172,0.12); color:{G}; }}
.brkt-verdict.miss {{ background:rgba(255,70,85,0.12); color:{R}; }}
/* --- connector cells --- */
.brkt-conn {{ position:relative; }}
.brkt-conn::before,.brkt-conn::after {{ content:""; position:absolute; left:0; }}
/* horizontal lines in gap cells */
.brkt-gap-hline {{ position:relative; }}
.brkt-gap-hline::before {{ content:""; position:absolute; top:50%; left:0; right:0; height:1.5px;
    background:rgba(170,183,204,0.22); }}
.brkt-gap-vline::before {{ content:""; position:absolute; left:50%; top:0; bottom:0; width:1.5px;
    background:rgba(170,183,204,0.22); }}
/* --- gap between round-2 groups --- */
.brkt-gap-row td {{ height:36px; }}
/* --- metrics --- */
.brkt-metrics {{ display:flex; gap:14px; padding:6px 0 16px; flex-wrap:wrap; }}
.brkt-mcard {{ background:{S1}; border:1px solid {BD}; border-radius:8px; padding:12px 18px; text-align:center;
    min-width:95px; flex:1; }}
.brkt-mval {{ font-weight:850; font-size:1.35rem; color:{TX}; }}
.brkt-mlabel {{ font-size:0.63rem; color:{T3}; text-transform:uppercase; letter-spacing:0.05em; margin-top:2px; }}
.brkt-mcard.good .brkt-mval {{ color:{G}; }}
.brkt-mcard.warn .brkt-mval {{ color:{Y}; }}
.brkt-mcard.bad .brkt-mval {{ color:{R}; }}
</style>
"""


# ── card HTML ───────────────────────────────────────────────────────────
def _card(m: dict[str, Any]) -> str:
    ta = escape(m["team_a"])
    tb = escape(m["team_b"])
    sa = m["score_a"]
    sb = m["score_b"]
    winner = m["winner"]
    prob_a = m["prob_a"]
    pred_winner = m["predicted_winner"]
    is_upset = m.get("is_upset", False)

    ucls = " upset" if is_upset else ""

    if abs(prob_a - 0.5) < 0.1:
        pcls, plabel = "close", "接近"
    elif pred_winner == winner:
        pcls, plabel = "ok", "✓ 正确"
    else:
        pcls, plabel = "bad", "✗ 错误"

    a_cls = "win" if winner == m["team_a"] else "loss"
    b_cls = "win" if winner == m["team_b"] else "loss"

    d = m.get("details", {})
    prob_b = 1.0 - prob_a
    pa = round(prob_a * 100)
    pb = round(prob_b * 100)

    h2h_matches = d.get("h2h_matches", 0)
    h2h_ratio = d.get("h2h_ratio", 0.5)
    blw = d.get("h2h_weight", 0.0)
    mode = d.get("mode", "pre_match")
    eff_weight = d.get("h2h_effective_weight", 0.0)
    raw_prob = d.get("raw_prob", prob_a)

    if h2h_matches > 0:
        if blw > 0.005:
            h2h_status_label = "✓ H2H 校准已启用"
            h2h_status_color = G
            h2h_detail = f"{ta} 历史交手 {h2h_ratio:.1%}（共 {h2h_matches} 场，累积权重 {eff_weight:.2f}）"
        else:
            h2h_status_label = "△ H2H 数据不足"
            h2h_status_color = Y
            h2h_detail = f"{ta} 历史交手 {h2h_ratio:.1%}（{h2h_matches} 场，累积权重 {eff_weight:.2f} < 阈值 2.0）"
    else:
        h2h_status_label = "— 无交手记录"
        h2h_status_color = T3
        h2h_detail = "两队无历史交手记录，H2H 校准不适用"

    prob_diff = prob_a - raw_prob
    calib_note = ""
    if abs(prob_diff) > 0.005:
        arrow = "↑" if prob_diff > 0 else "↓"
        calib_note = f"H2H 校准 {arrow}{abs(prob_diff):.1%}：原始 {raw_prob:.1%} → 校准后 {prob_a:.1%}"

    if abs(prob_a - 0.5) < 0.05:
        verdict = "模型判断接近五五开，无明显倾向"
    elif pred_winner == winner:
        verdict = f"模型正确预测 {escape(pred_winner)} 获胜（信心 {max(pa, pb):.0f}%）"
    else:
        verdict = f"模型预测 {escape(pred_winner)} 获胜（信心 {max(pa, pb):.0f}%），但实际 {escape(winner)} 取胜"
    vcls = "hit" if pred_winner == winner else "miss"

    cid = m.get("_cid", "")
    cid_attr = f' id="{cid}"' if cid else ""

    return f"""<div class="brkt-card{ucls}"{cid_attr}>
<details>
<summary class="brkt-sum"><div class="brkt-row">
<div class="brkt-team"><div class="brkt-tname {a_cls}">{ta}</div></div>
<div class="brkt-score">{sa} : {sb}</div>
<div class="brkt-team" style="text-align:right"><div class="brkt-tname {b_cls}">{tb}</div></div>
<div class="brkt-pred {pcls}">{plabel}<br>{pa}:{pb}</div>
</div></summary>
<div class="brkt-det">
<div class="brkt-grid">
<div><div class="brkt-dl">预测模式</div><div class="brkt-dv">{mode}</div></div>
<div><div class="brkt-dl">H2H 校准状态</div><div class="brkt-dv" style="color:{h2h_status_color};">{h2h_status_label}</div></div>
<div><div class="brkt-dl">{ta} 胜率</div><div class="brkt-dv big" style="color:{B};">{prob_a:.1%}</div></div>
<div><div class="brkt-dl">{tb} 胜率</div><div class="brkt-dv big" style="color:{R};">{prob_b:.1%}</div></div>
</div>
<div class="brkt-bar"><div class="brkt-bar-a" style="width:{pa}%"></div><div class="brkt-bar-b" style="width:{pb}%"></div></div>
<div style="font-size:0.66rem;color:{T3};margin-bottom:4px;">{h2h_detail}{' · ' + calib_note if calib_note else ''}</div>
<div class="brkt-verdict {vcls}">{verdict}</div>
</div>
</details>
</div>"""


# ── Table bracket builder ───────────────────────────────────────────────
def _td(content: str = "", cls: str = "", attrs: str = "", rowspan: int = 1) -> str:
    rs = f' rowspan="{rowspan}"' if rowspan > 1 else ""
    c = f' class="{cls}"' if cls else ""
    return f"<td{c}{rs}{attrs}>{content}</td>"


def render_swiss_bracket(matches: list[dict[str, Any]]) -> str:
    """Render Swiss bracket as HTML with CSS Grid and dynamic SVG connectors.

    Card positions are read via getBoundingClientRect() by JavaScript,
    after which SVG bezier curves are drawn over the bracket container
    with correct positions — green for winners, red for losers.
    """
    r1 = [m for m in matches if m["round"] == 1]
    r2_10 = [m for m in matches if m["round"] == 2 and "1-0" in m.get("group", "")]
    r2_01 = [m for m in matches if m["round"] == 2 and "0-1" in m.get("group", "")]
    r3 = [m for m in matches if m["round"] == 3]

    # Assign stable card IDs for JS to target
    for i, m in enumerate(r1):
        m["_cid"] = f"r1-{i}"
    for i, m in enumerate(r2_10):
        m["_cid"] = f"r2-10-{i}"
    for i, m in enumerate(r2_01):
        m["_cid"] = f"r2-01-{i}"
    for i, m in enumerate(r3):
        m["_cid"] = f"r3-{i}"

    r1_cards = "\n".join(_card(m) for m in r1)
    r2_cards_10 = "\n".join(_card(m) for m in r2_10)
    r2_cards_01 = "\n".join(_card(m) for m in r2_01)
    r3_cards = "\n".join(_card(m) for m in r3)

    # ── Build connection map based on actual team flows ──────────────────
    # Swiss format:
    #   R1 winner → R2 1-0 group (green line = won)
    #   R1 loser  → R2 0-1 group (red line = lost)
    #   R2 1-0 loser  → R3 1-1 group (red line = lost in 1-0)
    #   R2 0-1 winner → R3 1-1 group (green line = won in 0-1)
    #
    # We match by team name so ordering in the lists doesn't matter.

    winner_color = "rgba(134,239,172,0.55)"   # green  — won this match
    loser_color  = "rgba(255,70,85,0.55)"      # red    — lost this match

    # Helper: loser of a match
    def _loser(m: dict) -> str:
        return m["team_b"] if m["winner"] == m["team_a"] else m["team_a"]

    # Build team → card-id lookups for R2 and R3
    # R2 1-0: each card contains a team that won in R1
    r2_10_by_team: dict[str, str] = {}
    for m in r2_10:
        r2_10_by_team[m["team_a"]] = m["_cid"]
        r2_10_by_team[m["team_b"]] = m["_cid"]

    r2_01_by_team: dict[str, str] = {}
    for m in r2_01:
        r2_01_by_team[m["team_a"]] = m["_cid"]
        r2_01_by_team[m["team_b"]] = m["_cid"]

    r3_by_team: dict[str, str] = {}
    for m in r3:
        r3_by_team[m["team_a"]] = m["_cid"]
        r3_by_team[m["team_b"]] = m["_cid"]

    connections: list[tuple[str, str, str]] = []

    # R1 → R2
    for m in r1:
        winner = m["winner"]
        loser  = _loser(m)
        src    = m["_cid"]
        # winner goes to 1-0 group (green)
        if winner in r2_10_by_team:
            connections.append((src, r2_10_by_team[winner], winner_color))
        # loser goes to 0-1 group (red)
        if loser in r2_01_by_team:
            connections.append((src, r2_01_by_team[loser], loser_color))

    # R2 1-0 → R3: the loser drops to 1-1 (red — they just lost)
    for m in r2_10:
        loser = _loser(m)
        src   = m["_cid"]
        if loser in r3_by_team:
            connections.append((src, r3_by_team[loser], loser_color))

    # R2 0-1 → R3: the winner rises to 1-1 (green — they just won)
    for m in r2_01:
        winner = m["winner"]
        src    = m["_cid"]
        if winner in r3_by_team:
            connections.append((src, r3_by_team[winner], winner_color))

    connections_js = "[\n"
    for from_id, to_id, color in connections:
        connections_js += f'  ["{from_id}", "{to_id}", "{color}"],\n'
    connections_js += "]"

    connector_js = f"""
<script>
(function() {{
  function drawConnectors() {{
    var wrap = document.getElementById('brkt-outer');
    if (!wrap) return;
    var old = document.getElementById('brkt-svg-overlay');
    if (old) old.remove();

    var wRect = wrap.getBoundingClientRect();
    var svgNS = 'http://www.w3.org/2000/svg';
    var svg = document.createElementNS(svgNS, 'svg');
    svg.id = 'brkt-svg-overlay';
    svg.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;overflow:visible;z-index:0;';
    wrap.style.position = 'relative';
    wrap.insertBefore(svg, wrap.firstChild);

    var connections = {connections_js};

    connections.forEach(function(conn) {{
      var fromId = conn[0], toId = conn[1], color = conn[2];
      var fromEl = document.getElementById(fromId);
      var toEl   = document.getElementById(toId);
      if (!fromEl || !toEl) return;

      var fR = fromEl.getBoundingClientRect();
      var tR = toEl.getBoundingClientRect();

      // from: right-middle of source card
      var x1 = fR.right  - wRect.left;
      var y1 = (fR.top + fR.bottom) / 2 - wRect.top;
      // to: left-middle of target card
      var x2 = tR.left   - wRect.left;
      var y2 = (tR.top + tR.bottom) / 2 - wRect.top;

      var cx1 = x1 + (x2 - x1) * 0.5;
      var cx2 = x2 - (x2 - x1) * 0.5;

      var path = document.createElementNS(svgNS, 'path');
      path.setAttribute('d', 'M '+x1+' '+y1+' C '+cx1+' '+y1+' '+cx2+' '+y2+' '+x2+' '+y2);
      path.setAttribute('fill', 'none');
      path.setAttribute('stroke', color);
      path.setAttribute('stroke-width', '1.5');
      svg.appendChild(path);
    }});
  }}

  // Run after layout settles; also re-run on <details> toggle
  if (document.readyState === 'complete') {{
    setTimeout(drawConnectors, 80);
  }} else {{
    window.addEventListener('load', function() {{ setTimeout(drawConnectors, 80); }});
  }}
  document.addEventListener('toggle', function() {{ setTimeout(drawConnectors, 50); }}, true);
  window.addEventListener('resize', function() {{ setTimeout(drawConnectors, 50); }});
}})();
</script>"""

    return _css() + f"""
<div id="brkt-outer" class="brkt-wrap" style="position:relative;">
<div style="display:flex;gap:0;align-items:flex-start;position:relative;z-index:1;">

    <!-- Round 1 -->
    <div style="flex:1;min-width:260px;max-width:360px;display:flex;flex-direction:column;padding-right:40px;">
        <div class="brkt-rtitle">Round 1</div>
        {r1_cards}
    </div>

    <!-- Round 2 -->
    <div style="flex:1;min-width:260px;max-width:360px;display:flex;flex-direction:column;padding:0 40px;">
        <div class="brkt-rtitle">Round 2</div>
        <div class="brkt-gtitle">1-0 组</div>
        {r2_cards_10}
        <div class="brkt-gtitle" style="margin-top:28px;">0-1 组</div>
        {r2_cards_01}
    </div>

    <!-- Round 3 -->
    <div style="flex:1;min-width:260px;max-width:360px;display:flex;flex-direction:column;padding-left:40px;">
        <div class="brkt-rtitle">Round 3</div>
        <div class="brkt-gtitle">1-1 组</div>
        {r3_cards}
    </div>

</div>
</div>
{connector_js}"""


def render_metrics_bar(metrics: dict[str, Any]) -> str:
    brier   = metrics.get("brier", 0)
    acc     = metrics.get("accuracy", 0)
    auc     = metrics.get("roc_auc", 0)
    logloss = metrics.get("log_loss", 0)
    n_ok    = metrics.get("n_correct", 0)
    n_total = metrics.get("n_matches", 0)
    n_up    = metrics.get("n_upsets", 0)

    items = [
        ("good" if acc >= 0.7 else ("bad" if acc < 0.5 else ""),
         f"{n_ok}/{n_total}", f"正确 ({acc:.0%})"),
        ("good" if brier <= 0.2 else ("bad" if brier > 0.3 else ""),
         f"{brier:.4f}", "Brier Score"),
        ("", f"{logloss:.4f}", "Log Loss"),
        ("good" if auc >= 0.8 else "",
         f"{auc:.4f}", "ROC-AUC"),
        ("warn" if n_up > 0 else "",
         str(n_up), "冷门场次"),
    ]
    cards = []
    for cls, val, label in items:
        cards.append(
            f'<div class="brkt-mcard {cls}">'
            f'<div class="brkt-mval">{val}</div>'
            f'<div class="brkt-mlabel">{label}</div></div>'
        )
    return _css() + f'<div class="brkt-metrics">{"".join(cards)}</div>'
