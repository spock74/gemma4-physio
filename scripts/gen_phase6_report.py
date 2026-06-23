"""Generate a self-contained HTML report for the Phase 6 probing track.
Pulls real numbers from the latest E1/E2 result JSONs and inlines them for Plotly.

STALE TEMPLATE WARNING: docs/research/phase6-report.html has been HAND-MAINTAINED
since 2026-06-10 (owner reinterpretation + the E1c controls section). This script's
TEMPLATE predates those edits — regenerating would CLOBBER them. The guard below
refuses to overwrite unless --force is passed; port the hand edits into TEMPLATE
before forcing.
"""
import glob
import json
import sys
from pathlib import Path

ROOT = "/Users/moraes/Documents/PROJETOS/main-projects/gemma4-lab"

_target = Path(ROOT) / "docs" / "research" / "phase6-report.html"
if (_target.exists() and "Hand-maintained since" in _target.read_text(encoding="utf-8")
        and "--force" not in sys.argv):
    sys.exit("REFUSING to overwrite phase6-report.html: it is hand-maintained (owner edits "
             "+ E1c section) and this TEMPLATE is stale. Port the edits, then re-run with --force.")
e1 = json.load(open(sorted(glob.glob(f"{ROOT}/data/eval/results/entity_knowledge_*.json"))[-1]))
e2 = json.load(open(sorted(glob.glob(f"{ROOT}/data/eval/results/matformer_elastic_*.json"))[-1]))

s1 = e1["summary"]
sweep = e1["layer_auc_sweep_full"]
nec = e1["necessity"]
pl = e2["per_layer"]
s2 = e2["summary"]

DATA = {
    "e1": {
        "model_id": e1["model_id"],
        "instruction": e1["instruction"],
        "extraction_layer": e1["extraction_layer"],
        "n_layers": e1["n_layers"],
        "train_auc": round(s1["train_auc"], 3),
        "val_auc": round(s1["val_auc"], 3),
        "n_val_known": s1["n_val_known"],
        "sweep_layers": [r[0] for r in sweep],
        "sweep_auc": [r[1] for r in sweep],
        "spec_labels": ["d_know", "random (mean of 5)", "random (max of 5)", "orthogonal"],
        "spec_values": [round(s1["necessity_mean_logit_drop"], 2),
                        round(s1["necessity_random_mean_logit_drop"], 2),
                        round(s1["necessity_random_max_logit_drop"], 2),
                        round(s1["necessity_orth_mean_logit_drop"], 2)],
        "specificity_ratio": round(s1["specificity_ratio"], 2),
        "gate_pass": s1["specificity_gate_pass"],
        "item_answers": [x["answer"] for x in nec],
        "item_dknow": [round(x["delta"], 2) for x in nec],
        "item_randmax": [round(x["random_max_drop"], 2) for x in nec],
        "item_orth": [round(x["orth_drop"], 2) for x in nec],
        "suff_drop": round(s1["sufficiency_mean_entropy_drop"], 3),
        "suff_frac": round(s1["sufficiency_fraction_sharpened"], 2),
        "median_rank": s1["necessity_median_clean_rank"],
        "top5": round(s1["necessity_frac_recalled_top5"], 2),
    },
    "e2": {
        "d2": s2["d_model_e2b"], "d4": s2["d_model_e4b"],
        "n2": s2["n_layers_e2b"], "n4": s2["n_layers_e4b"],
        "n_repr": s2["n_repr_cka"],
        "mean_cka": s2["mean_cka"], "max_cka": s2["max_cka"],
        "mean_head": s2["mean_cka_head_slice"], "mean_rand": s2["mean_cka_random_slice"],
        "mean_tail": s2["mean_cka_tail_slice"],
        "head_beats_random": s2["nesting_head_beats_random"],
        "decode_e2b_max": s2["decode_auc_e2b_diffmeans_max"],
        "decode_e4b_max": s2["decode_auc_e4b_diffmeans_max"],
        "e2b_layer": [r["e2b_layer"] for r in pl],
        "e4b_layer": [r["e4b_layer"] for r in pl],
        "cka": [r["cka"] for r in pl],
        "head": [r["cka_head_slice"] for r in pl],
        "tail": [r["cka_tail_slice"] for r in pl],
        "rand": [r["cka_random_slice"] for r in pl],
        "dec_e2b": [r["decode_auc_e2b_diffmeans"] for r in pl],
        "dec_e4b": [r["decode_auc_e4b_diffmeans"] for r in pl],
    },
}

TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Gemma 4 E2B Probing — Phase 6 Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<style>
  :root{
    --bg:#0d1117; --panel:#161b22; --panel2:#1c2230; --ink:#e6edf3; --mut:#9aa7b4;
    --line:#2a3340; --accent:#58a6ff; --good:#3fb950; --bad:#f85149; --warn:#d29922; --pur:#bc8cff;
  }
  *{box-sizing:border-box}
  html,body{margin:0;background:var(--bg);color:var(--ink);
    font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif;}
  .wrap{max-width:1040px;margin:0 auto;padding:40px 22px 120px;}
  h1{font-size:34px;line-height:1.15;margin:0 0 6px;letter-spacing:-.02em}
  h2{font-size:25px;margin:54px 0 6px;padding-top:14px;border-top:1px solid var(--line);letter-spacing:-.01em}
  h3{font-size:19px;margin:30px 0 6px;color:var(--ink)}
  p{color:#c9d4df}
  .sub{color:var(--mut);font-size:15px;margin:0 0 8px}
  code,.mono{font-family:"SF Mono",ui-monospace,Menlo,Consolas,monospace;font-size:.92em}
  code{background:#1f2733;padding:1px 6px;border-radius:5px;color:#cfe3ff}
  a{color:var(--accent);text-decoration:none}
  .lede{font-size:18px;color:#d7e0ea}
  .grid{display:grid;gap:14px}
  .cards{grid-template-columns:repeat(auto-fit,minmax(225px,1fr));margin:22px 0}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px}
  .card .k{font-size:12.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--mut)}
  .card .v{font-size:26px;font-weight:650;margin-top:4px;letter-spacing:-.01em}
  .card .d{font-size:13.5px;color:var(--mut);margin-top:3px}
  .v.good{color:var(--good)} .v.bad{color:var(--bad)} .v.warn{color:var(--warn)} .v.acc{color:var(--accent)}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:8px 10px;margin:16px 0}
  .mermaid{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;margin:16px 0;overflow:auto;text-align:center}
  .plot{width:100%;height:430px}
  .cap{font-size:13.5px;color:var(--mut);margin:6px 4px 2px}
  .callout{border-left:3px solid var(--accent);background:#11202e;border-radius:0 10px 10px 0;padding:12px 16px;margin:16px 0}
  .callout.bad{border-color:var(--bad);background:#23161a}
  .callout.good{border-color:var(--good);background:#13231a}
  .callout.warn{border-color:var(--warn);background:#241f12}
  .callout b{color:#fff}
  table{border-collapse:collapse;width:100%;margin:14px 0;font-size:14.5px}
  th,td{border:1px solid var(--line);padding:8px 11px;text-align:left}
  th{background:var(--panel2);color:var(--mut);font-weight:600}
  td.mono{white-space:nowrap}
  .tag{display:inline-block;font-size:11.5px;font-weight:650;padding:2px 9px;border-radius:999px;vertical-align:middle}
  .tag.fail{background:#3a1620;color:#ff9d96;border:1px solid #5d2330}
  .tag.ok{background:#13311f;color:#74e090;border:1px solid #235437}
  .tag.warn{background:#33280f;color:#f0c24a;border:1px solid #5a4413}
  .foot{color:var(--mut);font-size:13px;margin-top:60px;border-top:1px solid var(--line);padding-top:16px}
  .two{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  @media(max-width:760px){.two{grid-template-columns:1fr}}
  .pill{font-size:12.5px;color:var(--mut);border:1px solid var(--line);border-radius:999px;padding:3px 10px;display:inline-block;margin:2px 4px 2px 0}
</style>
</head>
<body>
<div class="wrap">

  <h1>Probing the entity-knowledge geometry of Gemma 4 E2B</h1>
  <p class="sub">Phase 6 · affordable mechanistic interpretability on a Mac Mini M2 (16 GB, bf16, MPS+CPU split) · <span class="mono" id="modelid"></span> · transformers 5.7 · 2026-06-09</p>
  <p class="lede">Two experiments, diff-of-means + ablation only (no SAE). The honest headline:
  a <b>real, held-out linear</b> known/unknown direction that is <b>decodable across both MatFormer
  granularities</b> — but <b>no causal claim survives a specificity control</b>. A careful negative.</p>

  <div class="grid cards" id="cards"></div>

  <div class="callout warn">
    <b>Reading guide.</b> Every number here is held-out where it can be, and every causal claim is
    checked against random/orthogonal baselines. Where a result is an artifact, it is labelled an
    artifact — not tuned away. The value is in the <i>structure of the claim</i>, not the method cost.
  </div>

  <h2>1 · Method philosophy — the claim-strength ladder</h2>
  <p>The hardware caps "probing" to forward-pass activation capture, difference-of-means directions,
  directional ablation/steering, and logit readouts. That is enough for a <b>causal</b> result — what
  separates strong from weak is the claim, not the price:</p>
  <div class="mermaid">
flowchart LR
  R1["Rung 1 · Correlational<br/>geometry scalar vs label<br/><i>weak, confound-prone</i>"]
  R2["Rung 2 · Supervised separability<br/>linear probe / diff-of-means AUC<br/><i>decodability</i>"]
  R3["Rung 3 · Causal<br/>ablate removes · steer induces<br/><i>strong, even one-line method</i>"]
  R1 --> R2 --> R3
  E1["E1 lands here<br/>(reached for rung 3,<br/>fell back to rung 2)"]
  E2["E2 lands here"]
  R2 -.-> E1
  R2 -.-> E2
  style R3 fill:#13231a,stroke:#3fb950,color:#e6edf3
  style R2 fill:#11202e,stroke:#58a6ff,color:#e6edf3
  style R1 fill:#241f12,stroke:#d29922,color:#e6edf3
  style E1 fill:#1c2230,stroke:#bc8cff,color:#e6edf3
  style E2 fill:#1c2230,stroke:#bc8cff,color:#e6edf3
  </div>

  <h2>2 · The model — Gemma 4 E2B vs E4B (MatFormer)</h2>
  <p>E4B is the parent; E2B is a MatFormer sub-model. The design docs assumed E2B is <i>nested in E4B
  sharing the residual stream</i>. The shipped configs <b>falsify that</b>: canonical MatFormer (Gemma 3n)
  nests the FFN width only — Gemma 4 also compresses <code>d_model</code> and depth, so the two residual
  streams are different vector spaces (and <code>transformers</code> exposes no slicing API).</p>
  <div class="mermaid">
flowchart TB
  subgraph E4B["E4B-it · parent"]
    A4["d_model 2560 · 42 layers · 2 KV heads · 4.5B eff"]
  end
  subgraph E2B["E2B-it · MatFormer sub-model"]
    A2["d_model 1536 · 35 layers · 1 KV head · 2.3B eff"]
  end
  E4B -- "Mix-n-Match: slice FFN + skip layers<br/>(+ compress d_model & depth — beyond canonical MatFormer)" --> E2B
  PLE["Per-Layer Embeddings (256-d)"] -.-> E4B
  PLE -.-> E2B
  NOTE["⇒ residual streams are DIFFERENT spaces<br/>⇒ literal direction transfer is ill-posed<br/>⇒ E2 reframed: CKA + held-out decodability"]
  E2B --> NOTE
  style E4B fill:#11202e,stroke:#58a6ff,color:#e6edf3
  style E2B fill:#1c2230,stroke:#bc8cff,color:#e6edf3
  style NOTE fill:#23161a,stroke:#f85149,color:#e6edf3
  style PLE fill:#161b22,stroke:#2a3340,color:#9aa7b4
  </div>

  <h2>3 · E1 — Entity-knowledge direction</h2>
  <p><b>Question.</b> Is there a single linear "known-entity" direction <code>d_know</code> that the model
  <i>uses</i> to gate factual recall vs. confabulation? <b>Anchor:</b> Ferrando et al. 2024/ICLR 2025,
  <i>Do I Know This Entity?</i> — we extract the axis with diff-of-means (the SAE-free, M2-feasible approximation).</p>

  <div class="mermaid">
flowchart TB
  C["Contrast corpus<br/>40 known clozes (single-token answer)<br/>40 unknown/fictional"]
  RO{"Readout on -it"}
  C --> RO
  RO -->|"raw cloze"| ECHO["model ECHOES context<br/>gold rank ~600 · INVALID"]
  RO -->|"assistant-prefill ✓"| REC["model recalls · gold rank 0<br/>(stem prefills model turn)"]
  REC --> SPLIT["stratified TRAIN / VAL split"]
  SPLIT --> DIR["fit d_know = diff-of-means<br/>+ pick layer on TRAIN"]
  DIR --> VAL["held-out separation AUC on VAL"]
  DIR --> NEC["H1 necessity (VAL): ablate d_know<br/>across all layers → gold-logit drop"]
  NEC --> GATE{"SPECIFICITY GATE<br/>d_know vs 5 random + orthogonal"}
  GATE -->|"ratio > 2"| SPEC["specific to d_know ✓"]
  GATE -->|"ratio ≤ 2"| ART["ABLATION ARTIFACT — stop, report"]
  DIR --> SUF["H2 sufficiency (VAL): steer on unknown<br/>→ entropy change"]
  style ECHO fill:#23161a,stroke:#f85149,color:#e6edf3
  style REC fill:#13231a,stroke:#3fb950,color:#e6edf3
  style GATE fill:#241f12,stroke:#d29922,color:#e6edf3
  style ART fill:#23161a,stroke:#f85149,color:#e6edf3
  style SPEC fill:#13231a,stroke:#3fb950,color:#e6edf3
  </div>

  <h3>3.1 · The readout matters more than the model</h3>
  <p>Fed a raw cloze the instruction-tuned model <b>echoes the context</b> ("The capital of France is"
  → " France"), so the gold token sits at median rank ~600 and any causal test is off-distribution. Staying
  on <code>-it</code> (the deployed model — we do <b>not</b> switch to base <code>-pt</code>), an
  <b>assistant-prefill</b> readout makes it emit the fact at <b>rank 0</b> (top-5 recall 100%).</p>

  <h3>3.2 · Separability is real and held-out</h3>
  <div class="panel"><div id="p_sweep" class="plot"></div></div>
  <div class="cap">Known/unknown projection onto <code>d_know</code> separates at high AUC across a <b>broad band</b> of
  layers — the argmax "peak" is not stably localized (layer 8 at n=16, layer 26 at n=40), so the "≈ Ferrando
  layer-9 in Gemma 2 2B" comparison is at best loose. Held-out VAL AUC = <span id="valauc"></span> (vs in-sample 1.00).</div>

  <h3>3.3 · …but the causal "necessity" is an ablation artifact</h3>
  <p>Naïve necessity (ablate <code>d_know</code> across all 35 layers) gave a huge gold-logit drop — which
  <i>looks</i> like strong necessity. The specificity control kills it: ablating, with the <b>same</b> protocol,
  random unit directions and a direction orthogonal to <code>d_know</code> hurts <b>just as much</b>.</p>
  <div class="two">
    <div class="panel"><div id="p_spec" class="plot"></div></div>
    <div class="panel"><div id="p_items" class="plot"></div></div>
  </div>
  <div class="cap">Left: mean held-out gold-logit drop by ablated direction. Right: per-item — the strongest of 5
  random directions (orange) frequently matches or beats <code>d_know</code> (blue).</div>
  <div class="callout bad">
    <b>Specificity gate: FAIL.</b> <span class="mono">specificity_ratio = drop(d_know) / max(random_max, orth) =
    <span id="ratio"></span></span> — well under the 2× bar. Removing <i>any</i> 1-D direction across all layers
    damages the residual stream comparably, so the necessity signal is <b>not specific to <code>d_know</code></b>.
    The earlier "+32.9, 40/40 confirmed" was a generic aggressive-ablation artifact. <b>No d_know-specific
    necessity is claimed, and it was not tuned to flip.</b>
  </div>

  <h3>3.4 · H2 sufficiency — a clean null, for an interesting reason</h3>
  <p>Steering <code>d_know</code> up on unknown prompts does <b>not</b> sharpen them (entropy drop
  <span id="suff"></span>, sharpened <span id="sufffrac"></span>). Unknown prompts already have <i>low</i> clean
  entropy — the <code>-it</code> model <b>confidently confabulates</b> on fictional entities — so there is nothing
  to sharpen. H2's premise (unknowns are high-entropy) is itself false on <code>-it</code>.</p>

  <h2>4 · E2 — Elastic interpretability across MatFormer granularities</h2>
  <p><b>Reframed question</b> (the original "transfer" is dimensionally ill-posed): despite Gemma 4 compressing
  <code>d_model</code> and depth, is the known/unknown geometry still <b>aligned</b> across E2B and E4B? Measured
  with dimension-agnostic CKA + held-out decodability, with the statistics fixed so the numbers are citable.</p>

  <div class="mermaid">
flowchart LR
  CO["same contrast<br/>80 prompts"] --> CAP2["load E2B · capture ALL token<br/>positions · all layers · FREE"]
  CAP2 --> CAP4["load E4B · capture ALL token<br/>positions · all layers · FREE"]
  CAP4 --> AL["align rows by (prompt, position)<br/>shared tokenizer → n_repr = 2054"]
  AL --> CKA["linear CKA<br/>per matched layer"]
  AL --> NEST["nesting: head vs TAIL vs RANDOM slice"]
  AL --> DEC["held-out k-fold decodability<br/>(diff-of-means, per model)"]
  style CAP2 fill:#1c2230,stroke:#bc8cff,color:#e6edf3
  style CAP4 fill:#11202e,stroke:#58a6ff,color:#e6edf3
  style NEST fill:#241f12,stroke:#d29922,color:#e6edf3
  </div>

  <div class="callout warn"><b>Validity fixes</b> (without these the numbers are not citable): multi-position
  capture lifts CKA out of the n≪d regime (<span class="mono">n_repr = <span id="nrepr"></span> &gt; 1000</span>);
  decodability is <b>k-fold held-out</b> (in-sample is trivially ~1.0 at d≫n); nesting must beat a <b>random-slice</b>
  baseline, not just the tail.</div>

  <h3>4.1 · Cross-granularity similarity & the nesting test</h3>
  <div class="two">
    <div class="panel"><div id="p_cka" class="plot"></div></div>
    <div class="panel"><div id="p_nest" class="plot"></div></div>
  </div>
  <div class="cap">Left: linear CKA between E2B and E4B at depth-matched layers (mean <span id="meancka"></span>).
  Right: the nesting test — E2B vs E4B's first-1536 "head" dims (the Matryoshka inner-slice hypothesis) vs tail
  vs random slice. Head &gt; random holds but the mean margin is small (+0.02), <b>concentrating in deep layers</b>.
  CKA includes shared chat-template positions, which inflate the absolute value — the head−random <i>margin</i> is
  the template-robust signal.</div>

  <h3>4.2 · The geometry survives compression to E2B</h3>
  <div class="panel"><div id="p_dec" class="plot"></div></div>
  <div class="cap">Held-out (k-fold) diff-of-means AUC for known-vs-unknown, by depth, in each granularity. Both
  reach ≈0.99 in the upper half — the entity-knowledge geometry is present and decodable in <b>both</b> E2B and E4B.</div>

  <h2>5 · What we can and cannot claim</h2>
  <table>
    <tr><th>Claim</th><th>Evidence</th><th>Verdict</th></tr>
    <tr><td>A linear known/unknown direction exists in E2B-it</td><td>held-out VAL AUC 0.96; broad-band sweep</td><td><span class="tag ok">SUPPORTED · rung 2</span></td></tr>
    <tr><td><code>d_know</code> is causally <i>necessary</i> for recall</td><td>specificity_ratio 0.98 (= random ≈ orth)</td><td><span class="tag fail">FAIL · artifact</span></td></tr>
    <tr><td>Adding <code>d_know</code> induces confident confabulation (sufficiency)</td><td>entropy ↑ not ↓; unknowns already low-entropy</td><td><span class="tag fail">NULL</span></td></tr>
    <tr><td>Entity geometry is aligned across E2B/E4B granularities</td><td>mean CKA 0.76; both decodable ≈0.99 held-out</td><td><span class="tag ok">SUPPORTED (moderate)</span></td></tr>
    <tr><td>E2B is the Matryoshka inner-slice of E4B</td><td>head 0.77 &gt; random 0.75, +0.18 deep only</td><td><span class="tag warn">WEAK · deep layers</span></td></tr>
    <tr><td>Direct probe/vector transfer E4B→E2B</td><td>1536 vs 2560 dims</td><td><span class="tag warn">N/A · ill-posed</span></td></tr>
  </table>
  <div class="callout good"><b>Net.</b> Both experiments are honest <b>separation / decodability</b> results — the
  entity-knowledge direction is real, held-out, and present across both MatFormer granularities. <b>No causal
  claim survives.</b> A careful negative beats a polished false positive.</div>

  <h3>Methods rigor bar applied throughout</h3>
  <div>
    <span class="pill">specificity controls (random + orthogonal)</span>
    <span class="pill">held-out / no selection circularity</span>
    <span class="pill">n ≫ d for CKA</span>
    <span class="pill">random-slice baseline for nesting</span>
    <span class="pill">nulls reported, never tuned</span>
    <span class="pill">torch + stdlib only (no sklearn)</span>
    <span class="pill">Logfire span on every capture/intervention</span>
  </div>

  <div class="foot">
    Generated from <code>data/eval/results/entity_knowledge_*.json</code> and
    <code>matformer_elastic_*.json</code>. Code: <code>src/gemma4_lab/interp/</code> · design:
    <code>docs/research/00–02</code>. Plots are interactive (hover / zoom / drag). Mermaid + Plotly via CDN.
  </div>
</div>

<script>
const D = __DATA__;
mermaid.initialize({startOnLoad:true, theme:'dark', themeVariables:{fontSize:'14px'}});

const INK='#e6edf3', MUT='#9aa7b4', LINE='#2a3340', PANEL='#161b22';
const BLUE='#58a6ff', GOOD='#3fb950', BAD='#f85149', WARN='#d29922', PUR='#bc8cff', ORANGE='#e3934d';
const LAYOUT = (title, xt, yt, extra={}) => Object.assign({
  title:{text:title, font:{color:INK, size:15}},
  paper_bgcolor:PANEL, plot_bgcolor:PANEL,
  font:{color:MUT, size:12},
  margin:{l:58,r:18,t:42,b:46},
  xaxis:{title:xt, gridcolor:LINE, zerolinecolor:LINE, color:MUT},
  yaxis:{title:yt, gridcolor:LINE, zerolinecolor:LINE, color:MUT},
  legend:{orientation:'h', y:-0.22, font:{color:MUT}},
}, extra);
const CFG = {displayModeBar:false, responsive:true};

// ---- model id ----
document.getElementById('modelid').textContent = D.e1.model_id;
document.getElementById('valauc').textContent = D.e1.val_auc;
document.getElementById('ratio').textContent = D.e1.specificity_ratio;
document.getElementById('suff').textContent = D.e1.suff_drop;
document.getElementById('sufffrac').textContent = (D.e1.suff_frac*100).toFixed(0)+'%';
document.getElementById('nrepr').textContent = D.e2.n_repr;
document.getElementById('meancka').textContent = D.e2.mean_cka;

// ---- TL;DR cards ----
const cards = [
  {k:'E1 separation (held-out)', v:D.e1.val_auc, cls:'good', d:'VAL AUC · linear known/unknown direction is real'},
  {k:'E1 specificity gate', v:D.e1.specificity_ratio+'×', cls:'bad', d:'d_know ≈ random ≈ orth → necessity is an artifact'},
  {k:'E2 cross-granularity CKA', v:D.e2.mean_cka, cls:'acc', d:'mean over matched layers · n_repr '+D.e2.n_repr},
  {k:'E2 decodability (both)', v:'≈0.99', cls:'good', d:'E2B '+D.e2.decode_e2b_max+' · E4B '+D.e2.decode_e4b_max+' held-out'},
];
document.getElementById('cards').innerHTML = cards.map(c=>
  `<div class="card"><div class="k">${c.k}</div><div class="v ${c.cls}">${c.v}</div><div class="d">${c.d}</div></div>`).join('');

// ---- E1 layer sweep ----
Plotly.newPlot('p_sweep', [
  {x:D.e1.sweep_layers, y:D.e1.sweep_auc, type:'scatter', mode:'lines+markers',
   line:{color:BLUE,width:2}, marker:{size:5,color:BLUE}, name:'eff-AUC',
   hovertemplate:'layer %{x}<br>AUC %{y:.3f}<extra></extra>'},
  {x:[D.e1.extraction_layer], y:[D.e1.sweep_auc[D.e1.extraction_layer]], type:'scatter', mode:'markers',
   marker:{size:13,color:PUR,symbol:'star'}, name:'chosen layer ('+D.e1.extraction_layer+')',
   hovertemplate:'chosen layer %{x}<extra></extra>'},
], LAYOUT('Known/unknown separability by layer (full-data sweep)','decoder layer','direction-blind eff-AUC',
   {yaxis:{range:[0.5,1.03], gridcolor:LINE, color:MUT, title:'direction-blind eff-AUC'}}), CFG);

// ---- E1 specificity bars ----
const specColors=[BLUE,WARN,ORANGE,PUR];
Plotly.newPlot('p_spec', [{
  x:D.e1.spec_labels, y:D.e1.spec_values, type:'bar',
  marker:{color:specColors}, text:D.e1.spec_values.map(v=>'+'+v), textposition:'outside',
  hovertemplate:'%{x}<br>drop %{y:.2f}<extra></extra>'
}], LAYOUT('H1 specificity — gold-logit drop under ablation (held-out)','ablated direction','mean gold-logit drop',
   {showlegend:false, yaxis:{title:'mean gold-logit drop', gridcolor:LINE, color:MUT, range:[0,35]}}), CFG);

// ---- E1 per-item grouped ----
Plotly.newPlot('p_items', [
  {x:D.e1.item_answers, y:D.e1.item_dknow, type:'bar', name:'d_know', marker:{color:BLUE}},
  {x:D.e1.item_answers, y:D.e1.item_randmax, type:'bar', name:'random (max of 5)', marker:{color:ORANGE}},
  {x:D.e1.item_answers, y:D.e1.item_orth, type:'bar', name:'orthogonal', marker:{color:PUR}},
], LAYOUT('Per-item (held-out VAL): d_know vs random vs orthogonal','known item','gold-logit drop',
   {barmode:'group', xaxis:{tickangle:-40, color:MUT, gridcolor:LINE}}), CFG);

// ---- E2 CKA by layer ----
Plotly.newPlot('p_cka', [
  {x:D.e2.e2b_layer, y:D.e2.cka, type:'scatter', mode:'lines+markers', line:{color:BLUE,width:2},
   marker:{size:4}, name:'CKA', hovertemplate:'E2B L%{x}<br>CKA %{y:.3f}<extra></extra>'},
], LAYOUT('Cross-granularity CKA by matched layer','E2B layer (depth-matched to E4B)','linear CKA',
   {yaxis:{range:[0.4,1.0], gridcolor:LINE, color:MUT, title:'linear CKA'}}), CFG);

// ---- E2 nesting ----
Plotly.newPlot('p_nest', [
  {x:D.e2.e2b_layer, y:D.e2.head, type:'scatter', mode:'lines', line:{color:GOOD,width:2}, name:'head slice (inner 1536)'},
  {x:D.e2.e2b_layer, y:D.e2.rand, type:'scatter', mode:'lines', line:{color:MUT,width:1.5,dash:'dot'}, name:'random slice'},
  {x:D.e2.e2b_layer, y:D.e2.tail, type:'scatter', mode:'lines', line:{color:WARN,width:1.5}, name:'tail slice'},
], LAYOUT('Nesting test: E2B vs E4B head / random / tail slice','E2B layer','linear CKA',
   {yaxis:{range:[0.4,1.0], gridcolor:LINE, color:MUT, title:'linear CKA'}}), CFG);

// ---- E2 decodability ----
Plotly.newPlot('p_dec', [
  {x:D.e2.e2b_layer, y:D.e2.dec_e2b, type:'scatter', mode:'lines+markers', line:{color:PUR,width:2}, marker:{size:4}, name:'E2B (1536d)'},
  {x:D.e2.e4b_layer, y:D.e2.dec_e4b, type:'scatter', mode:'lines+markers', line:{color:BLUE,width:2}, marker:{size:4}, name:'E4B (2560d)'},
], LAYOUT('Held-out known/unknown decodability by depth','layer','diff-of-means k-fold AUC',
   {yaxis:{range:[0.6,1.02], gridcolor:LINE, color:MUT, title:'held-out AUC'}}), CFG);
</script>
</body>
</html>'''

out = TEMPLATE.replace("__DATA__", json.dumps(DATA))
path = f"{ROOT}/docs/research/phase6-report.html"
open(path, "w", encoding="utf-8").write(out)
print("wrote", path, len(out), "bytes")
print("E1 ratio", DATA["e1"]["specificity_ratio"], "| E2 mean_cka", DATA["e2"]["mean_cka"], "| n_repr", DATA["e2"]["n_repr"])
