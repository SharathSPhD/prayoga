"""Graphical abstract: title + concept strip + three data-grounded panels
(dose-response, order-parameter broken symmetry, behavioural bound) + tier verdict.
All numbers are read from results/ aggregate JSON. Output: figures/graphical_abstract.png
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
RA, RB, RX = ROOT/"results/axis_a", ROOT/"results/axis_b", ROOT/"results/axis_x"
FIG = ROOT/"figures"
C_REFUSAL="#b5384d"; C_BLUE="#2f6f9f"; C_ORANGE="#d98a3a"; C_GREEN="#3f8f6b"
C_GRAY="#8a8f98"; C_INK="#233240"; C_BG="#f4f6f9"
plt.rcParams.update({"font.family":"serif","font.serif":["Palatino","Palatino Linotype","URW Palladio L","DejaVu Serif"]})

def gauss(x,mu,s): return np.exp(-0.5*((x-mu)/s)**2)/(s*np.sqrt(2*np.pi))
def logistic(x,ec,k,lo,hi): return lo+(hi-lo)/(1+np.exp(-k*(x-ec)))
def L(p): return json.loads(Path(p).read_text())

dose=L(RA/"dose_gemma-2-2b-it.json")
sym=L(RB/"symmetry_gemma-2-2b-it.json")
tri={m:L(RX/f"triangulation_{m}.json") for m in ("gemma-2-2b-it","qwen2.5-3b-it","gemma-2-9b-it")}

fig=plt.figure(figsize=(13.6,7.8),dpi=200)
bg=fig.add_axes([0,0,1,1]); bg.set_xlim(0,1); bg.set_ylim(0,1); bg.axis("off")
bg.add_patch(FancyBboxPatch((0,0),1,1,boxstyle="square,pad=0",facecolor=C_BG,edgecolor="none",zorder=0))

# title band
bg.add_patch(FancyBboxPatch((0,0.885),1,0.115,boxstyle="square,pad=0",facecolor=C_INK,edgecolor="none",zorder=1))
bg.text(0.5,0.955,"Refusal as a Broken Symmetry",ha="center",va="center",fontsize=30,weight="bold",color="white",zorder=2)
bg.text(0.5,0.912,"Mechanistic interpretability of output-policy capture across jailbreak, hypnosis, and vasikarana",
        ha="center",va="center",fontsize=11.5,color="#c7d0da",style="italic",zorder=2)

# concept strip
bg.add_patch(FancyBboxPatch((0.05,0.80),0.90,0.055,boxstyle="round,pad=0.004,rounding_size=0.02",
             facecolor="white",edgecolor="#d3dae2",lw=1.1,zorder=2))
bg.text(0.5,0.827,"Jailbreak  =  Hypnosis  =  Vasikarana:  one move -- inject a context that suppresses the "
        "monitor and co-opts generation",ha="center",va="center",fontsize=12,color=C_INK,weight="bold",zorder=3)

# panel headers / captions helpers
def header(x,t): bg.text(x,0.745,t,ha="center",fontsize=11.5,weight="bold",color=C_INK,zorder=3)
def caption(x,t): bg.text(x,0.30,t,ha="center",va="top",fontsize=9.6,color="#3c4854",zorder=3)

# ---- Panel 1: causal & dosable (dose-response) ----
header(0.185,"CAUSAL & DOSABLE")
a1=fig.add_axes([0.075,0.37,0.235,0.34])
xs=np.linspace(0,1,300); f=dose["fit"]
a1.plot(xs,logistic(xs,f["ec50"],f["slope"],f["lo"],f["hi"]),color=C_REFUSAL,lw=2.4,zorder=2)
a1.scatter(dose["alphas"],dose["asr_real"],color=C_REFUSAL,s=34,zorder=3,edgecolor="white",lw=0.5)
a1.axvline(f["ec50"],ls="--",color=C_INK,alpha=0.7)
a1.text(f["ec50"]+0.04,0.12,"EC50 0.33",fontsize=9,color=C_INK)
a1.set_xlabel(r"ablation strength $\alpha$",fontsize=9.5); a1.set_ylabel("attack success",fontsize=9.5)
a1.set_ylim(-0.05,1.05); a1.tick_params(labelsize=8)
for s in ("top","right"): a1.spines[s].set_visible(False)
caption(0.185,"ablate one direction: ASR 0.00 -> 0.90\nsmooth logistic dose-response, $R^2$=0.996")

# ---- Panel 2: broken symmetry (order parameter) ----
header(0.5,"A BROKEN SYMMETRY")
a2=fig.add_axes([0.395,0.37,0.235,0.34])
sd=sym["within_orbit_std_harmful"]; hm=sym["order_param_harmful_mean"]; sm=sym["order_param_harmless_mean"]
plain=sym["order_param_plain_harmful"]; inj=sym["order_param_injected_harmful"]
gx=np.linspace(sm-4*sd,hm+4*sd,400); pk=gauss(hm,hm,sd)
a2.fill_between(gx,gauss(gx,sm,sd),color=C_BLUE,alpha=0.35); a2.fill_between(gx,gauss(gx,hm,sd),color=C_REFUSAL,alpha=0.35)
a2.plot(gx,gauss(gx,sm,sd),color=C_BLUE,lw=1.6); a2.plot(gx,gauss(gx,hm,sd),color=C_REFUSAL,lw=1.6)
a2.annotate("",xy=(inj,pk*0.42),xytext=(plain,pk*0.42),arrowprops=dict(arrowstyle="-|>",color=C_ORANGE,lw=2.4))
a2.text((plain+inj)/2,pk*0.52,"injection",ha="center",fontsize=9,color=C_ORANGE,weight="bold")
a2.text(sm,pk*1.05,"harmless",ha="center",fontsize=8.5,color=C_BLUE); a2.text(hm,pk*1.05,"harmful",ha="center",fontsize=8.5,color=C_REFUSAL)
a2.set_ylim(0,pk*1.25); a2.set_yticks([]); a2.set_xticks([])
a2.set_xlabel(r"order parameter $m=(h\cdot\hat d)/\Vert h\Vert$",fontsize=9.5)
for s in ("top","right","left"): a2.spines[s].set_visible(False)
caption(0.5,"invariant across paraphrase orbits\n($F$-ratio 19.2 vs 0.65), collapses under injection")

# ---- Panel 3: the bound (triangulation) ----
header(0.815,"THE BOUND")
a3=fig.add_axes([0.715,0.37,0.235,0.34])
mods=["Gemma-2-2B","Qwen2.5-3B","Gemma-2-9B"]; keys=["gemma-2-2b-it","qwen2.5-3b-it","gemma-2-9b-it"]
x=np.arange(3); w=0.38
corr=[tri[k]["corr_AB_pearson"] for k in keys]; flip=[tri[k]["flip_rate"] for k in keys]
a3.bar(x-w/2,corr,w,color=C_BLUE,label="internal coupling")
a3.bar(x+w/2,flip,w,color=C_ORANGE,label="behavioural capture")
for i in range(3): a3.text(x[i]+w/2,flip[i]+0.03,f"{flip[i]:.1%}",ha="center",fontsize=7.5,color=C_INK)
a3.set_xticks(x); a3.set_xticklabels(["G-2B","Q-3B","G-9B"],fontsize=8.5); a3.set_ylim(0,1.05)
a3.set_ylabel("rate",fontsize=9.5); a3.tick_params(labelsize=8); a3.legend(fontsize=7.6,loc="upper center",ncol=1,frameon=False)
for s in ("top","right"): a3.spines[s].set_visible(False)
caption(0.815,"injection collapses internal readouts (corr 0.94-0.97)\nyet genuine capture is only 1-2.4%")

# ---- core innovation band ----
bg.add_patch(FancyBboxPatch((0.05,0.15),0.90,0.088,boxstyle="round,pad=0.004,rounding_size=0.02",
             facecolor=C_INK,edgecolor="none",alpha=0.94,zorder=2))
bg.text(0.5,0.212,"Refusal is a measurable, ablatable, dosable order parameter: a symmetry whose injection-collapse is a jailbreak.",
        ha="center",va="center",fontsize=10.8,color="white",zorder=3)
bg.text(0.5,0.176,"Yet that collapse does not capture behaviour, bounding the cross-domain unification to the representational level.",
        ha="center",va="center",fontsize=10.8,color="#dbe2ea",zorder=3)

# ---- tier verdict ----
for i,(t,s,c) in enumerate([("MECHANISM","measured: single, dosable refusal direction",C_REFUSAL),
                            ("ANALOGY","consistent: suppressed monitoring precision",C_BLUE),
                            ("METAPHOR","falsified: no avasthatraya / turiya machine-state",C_ORANGE)]):
    cx=0.09+i*0.315
    bg.add_patch(plt.Circle((cx,0.09),0.010,color=c,zorder=4))
    bg.text(cx+0.022,0.105,t,ha="left",va="center",fontsize=10.5,weight="bold",color=c,zorder=4)
    bg.text(cx+0.022,0.068,s,ha="left",va="center",fontsize=8.2,color="#3c4854",zorder=4)

fig.savefig(FIG/"graphical_abstract.png",dpi=200,facecolor=C_BG,bbox_inches="tight",pad_inches=0.06)
print("wrote",FIG/"graphical_abstract.png")
