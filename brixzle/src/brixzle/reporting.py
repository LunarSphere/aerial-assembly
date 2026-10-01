"""Standalone result tables, interactive preference ranking, and plots."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import html
import json

from .models import Evaluation, write_json
from .optimization import pareto_indices


def build_report(run_dir: Path, output_dir: Path | None = None) -> Path:
    output_dir = output_dir or run_dir / "report"
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluations = []
    for path in sorted(run_dir.rglob("evaluation.json")):
        e = Evaluation(**json.loads(path.read_text()))
        evaluations.append((path, e))
    if not evaluations:
        raise ValueError("No evaluations found under run directory")
    rows = [{"path": str(path.relative_to(run_dir)), **e.__dict__} for path, e in evaluations]
    # Escape '<' to prevent a malicious target/error string from ending script.
    payload = json.dumps(rows, allow_nan=False).replace("<", "\\u003c")
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Brixzle experiment report</title><style>
body{font:15px system-ui;margin:2rem;color:#223;max-width:1600px}h1{margin-bottom:.3rem}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:.5rem;border-bottom:1px solid #ddd;text-align:left}
.controls{display:flex;gap:1.3rem;flex-wrap:wrap;padding:1rem;background:#eef3f5;margin:1rem 0}
label{display:block}pre{white-space:pre-wrap;background:#eef3f5;padding:1rem}.bad{color:#9a2929}
button{padding:.3rem}.chart{max-width:850px;width:100%}circle{cursor:pointer}svg text{font-size:12px}
</style><h1>Brixzle experiments</h1>
<p>Results describe the configured contact simulation. Physical calibration and bending tests are pending.
Success means the complete benchmark passed, including intermediate stability.</p>
<div class="controls" id="controls"></div>
<label><input type="checkbox" id="feasible" checked> Show only geometrically and numerically feasible evaluations</label>
<p id="status"></p><svg class="chart" viewBox="0 0 850 340" id="chart"></svg>
<table><thead><tr><th>Design / phase</th><th>Worst success</th><th>Worst lower bound</th><th>Peak force p95 (N)</th>
<th>Mean capped time (s)</th><th>Mass (g)</th><th>Collision events</th><th>Preference score</th><th>Details</th></tr></thead><tbody id="rows"></tbody></table>
<pre id="details">Select a row to inspect its benchmark failures and parameters.</pre>
<script>
const data=PAYLOAD;
const labels=['Failure probability','Force','Time','Mass','Collisions'];
const weights=[1,1,1,1,1];
const el=id=>document.getElementById(id);
function text(tag,value){const node=document.createElement(tag);node.textContent=value;return node;}
labels.forEach((label,i)=>{const l=text('label',label+' ');const input=document.createElement('input');
input.type='range';input.min=0;input.max=5;input.step=.1;input.value=1;input.oninput=()=>{weights[i]=+input.value;render();};
l.append(input);el('controls').append(l);});
function render(){let candidates=data.filter(e=>!el('feasible').checked||e.feasible);
const lo=labels.map((_,j)=>Math.min(...candidates.map(e=>e.objectives[j])));
const hi=labels.map((_,j)=>Math.max(...candidates.map(e=>e.objectives[j])));
candidates=candidates.map(e=>({...e,score:e.objectives.reduce((s,x,j)=>s+weights[j]*(x-lo[j])/(hi[j]-lo[j]||1),0)/(weights.reduce((a,b)=>a+b,0)||1)})).sort((a,b)=>a.score-b.score);
el('status').textContent=`${candidates.length} evaluations shown of ${data.length}. Lower bounds are one-sided exact confidence bounds; search trials alone do not establish reliability.`;
el('rows').replaceChildren();el('chart').replaceChildren();
const ns='http://www.w3.org/2000/svg';
function svg(tag,attrs){const n=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));el('chart').append(n);return n;}
svg('path',{d:'M60 20 V290 H820',fill:'none',stroke:'#445'});
svg('text',{x:350,y:325}).textContent='Brick mass (g)';svg('text',{x:5,y:15}).textContent='Worst success (%)';
candidates.forEach(e=>{const benchmarks=Object.values(e.benchmarks).filter(b=>b.mandatory);
const lower=benchmarks.length?Math.min(...benchmarks.map(b=>b.success_lower_bound)):0;
const tr=document.createElement('tr');
const phase=['validate','reevaluate','numerical','numerical-standard','diagnostic'].find(p=>e.path.split('/').includes(p))||'search/evaluation';
const mode=benchmarks[0]?.manipulation_mode||'unknown';
const vals=[e.parameters.family+'/'+e.parameters.reinforcement+' · '+phase+' · '+mode,
((1-e.objectives[0])*100).toFixed(1)+'%',(lower*100).toFixed(1)+'%',e.objectives[1].toFixed(3),
e.objectives[2].toFixed(1),e.objectives[3].toFixed(2),e.objectives[4].toFixed(2),e.score.toFixed(3)];
vals.forEach(v=>tr.append(text('td',v)));const td=document.createElement('td');const b=text('button','Inspect');
const show=()=>{el('details').textContent=JSON.stringify(e,null,2);};b.onclick=show;td.append(b);tr.append(td);el('rows').append(tr);
const dot=svg('circle',{cx:60+Math.min(e.objectives[3],40)/40*760,cy:290-(1-e.objectives[0])*260,r:4,
fill:e.feasible?'#126c85':'#bb5555',opacity:.65});dot.onclick=show;});}
el('feasible').onchange=render;render();
</script></html>""".replace("PAYLOAD", payload)
    report = output_dir / "index.html"
    report.write_text(page)
    write_json(output_dir / "evaluations.json", rows)
    markdown = ["# Brixzle experiment results", "", "Simulation evidence; physical calibration is pending.", "",
                "| Family | Feasible | Worst success | Mass (g) | Force p95 (N) |",
                "| --- | --- | --- | --- | --- |"]
    for _, e in evaluations:
        markdown.append(f"| {e.parameters['family']}/{e.parameters['reinforcement']} | {e.feasible} | "
                        f"{1-e.objectives[0]:.1%} | {e.objectives[3]:.2f} | {e.objectives[1]:.3f} |")
    (output_dir / "summary.md").write_text("\n".join(markdown)+"\n")
    # Publication/export artifacts use a standard plotting library, without a
    # GUI dependency. Color distinguishes feasible and infeasible evaluations.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for feasible, color in ((True, "#126c85"), (False, "#bb5555")):
        subset = [e for _, e in evaluations if e.feasible == feasible]
        if subset:
            axes[0].scatter([e.objectives[3] for e in subset], [1-e.objectives[0] for e in subset],
                            c=color, label="Feasible" if feasible else "Infeasible", alpha=0.7)
            axes[1].scatter([e.objectives[1] for e in subset], [e.objectives[2] for e in subset], c=color, alpha=0.7)
    axes[0].set(xlabel="Brick mass (g)", ylabel="Worst benchmark success rate", ylim=(-0.03, 1.03))
    axes[0].legend()
    axes[1].set(xlabel="Worst benchmark force p95 (N)", ylabel="Mean capped assembly time (s)")
    fig.tight_layout()
    fig.savefig(output_dir / "objectives.svg")
    fig.savefig(output_dir / "objectives.png", dpi=160)
    plt.close(fig)
    return report
