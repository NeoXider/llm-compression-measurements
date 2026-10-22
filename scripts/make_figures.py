# Figures v2. REFERENCES (F16, Q8_0, Q4_0) are drawn as lines; OTHER built-in llama.cpp types are grey markers;
# OUR formats/prototypes are green. All numbers copied from RESULTS.md (R-numbers in captions).
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150})
REF={"F16":"#222222","Q8_0":"#1f78b4","Q4_0":"#e66101"}; OTHER="#8c8c8c"; OURS="#1b9e77"
os.makedirs("figures",exist_ok=True)
def refline(ax,y,name,xmax=None):
    ax.axhline(y,color=REF[name],lw=1.6,ls="--"); 
BETTER={"fig1_decode_speed":["up"],"fig2_vram":["down"],"fig3_kl_vs_bits":["downright","downright"],
        "fig4_lossless":["down"],"fig5_roofline_prediction":["up"],"fig6_read_microbench":["up"]}
def mark_better(ax,kind):
    # arrow in axes coordinates pointing toward the BETTER direction, green label
    c="#2e7d32"
    if kind=="up": ax.annotate("",xy=(0.96,0.92),xytext=(0.96,0.72),xycoords="axes fraction",arrowprops=dict(arrowstyle="-|>",color=c,lw=2)); ax.text(0.94,0.82,"better",transform=ax.transAxes,ha="right",va="center",fontsize=8,color=c,fontweight="bold")
    if kind=="down": ax.annotate("",xy=(0.96,0.10),xytext=(0.96,0.30),xycoords="axes fraction",arrowprops=dict(arrowstyle="-|>",color=c,lw=2)); ax.text(0.94,0.20,"better",transform=ax.transAxes,ha="right",va="center",fontsize=8,color=c,fontweight="bold")
    if kind=="downright":
        ax.annotate("",xy=(0.95,0.06),xytext=(0.78,0.22),xycoords="axes fraction",arrowprops=dict(arrowstyle="-|>",color=c,lw=2))
        ax.text(0.76,0.14,"better\n(smaller, less loss)",transform=ax.transAxes,ha="right",va="center",fontsize=7,color=c,fontweight="bold")
def save(fig,n):
    for ax,k in zip(fig.axes,BETTER.get(n,[])): mark_better(ax,k)
    fig.tight_layout(); fig.savefig(f"figures/{n}.png"); plt.close(fig)

# Fig1: decode speed vs F16 (R21, llama-bench) -- references are lines; ours not implemented in llama.cpp (stated)
# R25 (near-clean window): depths 4K/32K/64K/120K; absolute tok/s on the right panel
d=[4096,32768,65536,122880]; dl=["4K","32K","64K","120K"]
fig,axs=plt.subplots(1,2,figsize=(9.2,3.5))
ax=axs[0]
ax.axhline(1,color=REF["F16"],lw=1.8,ls="--",label="F16 KV (reference = 1.0)")
ax.plot(d,[0.883,0.732,0.684,0.629],"o-",color=REF["Q8_0"],lw=1.8,label="Q8_0 KV (reference)")
ax.plot(d,[0.889,0.725,0.668,0.622],"s-",color=REF["Q4_0"],lw=1.8,label="Q4_0 KV (reference)")
ax.text(0.99,0.04,"Our lossless formats: no llama.cpp kernel\n(read cost in Fig. 6)",transform=ax.transAxes,ha="right",fontsize=7,color=OURS)
ax.set_xscale("log",base=2); ax.set_xticks(d); ax.set_xticklabels(dl); ax.set_ylim(0.5,1.1)
ax.set_xlabel("Filled context depth (tokens)"); ax.set_ylabel("Decode speed vs F16 KV (x)"); ax.set_title("Decode speed ratio (R25)",fontsize=9)
ax.legend(frameon=False,fontsize=7,loc="center left")
ax=axs[1]
ax.plot(d,[162.2,88.2,59.1,37.6],"o--",color=REF["F16"],lw=1.8,label="F16 KV")
ax.plot(d,[143.2,64.6,40.4,23.6],"o-",color=REF["Q8_0"],lw=1.8,label="Q8_0 KV")
ax.plot(d,[144.2,63.9,39.5,23.3],"s-",color=REF["Q4_0"],lw=1.8,label="Q4_0 KV")
ax.set_xscale("log",base=2); ax.set_xticks(d); ax.set_xticklabels(dl)
ax.set_xlabel("Filled context depth (tokens)"); ax.set_ylabel("Decode tok/s (median)"); ax.set_title("Absolute decode speed, RTX 3090 (R25)",fontsize=9)
ax.legend(frameon=False,fontsize=7)
BETTER["fig1_decode_speed"]=["up","up"]
save(fig,"fig1_decode_speed")

# Fig2: KV memory vs filled tokens at -c 32768. References = flat reserved buffers (lines); ours = lazy pool (PoC-A, F16)
tok=[1024,4096,16384,32768]
fig,axs=plt.subplots(1,2,figsize=(9.2,3.5))
ax=axs[1]
cx=[4096,32768,65536,122880]
ax.plot(cx,[3246,7286,11902,19980],"o--",color=REF["F16"],lw=1.8,label="F16 KV")
ax.plot(cx,[2984,5246,7830,12352],"o-",color=REF["Q8_0"],lw=1.8,label="Q8_0 KV")
ax.plot(cx,[2840,4094,5526,8032],"s-",color=REF["Q4_0"],lw=1.8,label="Q4_0 KV")
ax.axhline(24576,color="#999999",lw=1,ls=":"); ax.text(122880,23300,"24 GiB card",fontsize=7,color="#666666",ha="right")
ax.set_xscale("log",base=2); ax.set_xticks(cx); ax.set_xticklabels(["4K","32K","64K","120K"])
ax.set_xlabel("Filled context (tokens; 64K/120K with YaRN)"); ax.set_ylabel("Process VRAM peak (MiB)"); ax.set_title("Whole process, filled context (R30)",fontsize=9)
ax.set_ylim(0,26000); ax.legend(frameon=False,fontsize=7,loc="center left")
ax=axs[0]
BETTER["fig2_vram"]=["down","down"]
ax.axhline(4608,color=REF["F16"],lw=1.8,ls="--",label="F16 KV, reserved at start (reference)")
ax.axhline(2448,color=REF["Q8_0"],lw=1.8,ls="--",label="Q8_0 KV, reserved (reference)")
ax.axhline(1296,color=REF["Q4_0"],lw=1.8,ls="--",label="Q4_0 KV, reserved (reference)")
ax.plot(tok,[144,576,2304,4608],"o-",color=OURS,lw=2,label="Ours: lazy VMM pool (F16, grows with tokens)")
ax.set_xscale("log",base=2); ax.set_xticks(tok); ax.set_xticklabels(["1K","4K","16K","32K"])
ax.set_xlabel("Tokens in context (-c 32768)"); ax.set_ylabel("KV memory (MiB)"); ax.set_title("KV memory: reserved vs lazily mapped",fontsize=9)
ax.set_ylim(0,5600); ax.legend(frameon=False,fontsize=7,loc="upper left",bbox_to_anchor=(0,0.79)); save(fig,"fig2_vram")

# Fig3: size vs KL. Vertical reference lines F16/Q8_0/Q4_0; grey = other llama.cpp types; green = ours (lossless, KL = 0)
kv_other=[("q5_1",6.0,0.0135),("q5_0",5.5,0.0168),("q4_1",5.0,0.0543),("iq4_nl",4.5,0.0983)]
fig,axs=plt.subplots(1,2,figsize=(9.2,3.6))
ax=axs[0]
for n,bpe,c in (("F16 (ref)",16,REF["F16"]),("Q8_0 (ref)",8.5,REF["Q8_0"]),("Q4_0 (ref)",4.5,REF["Q4_0"])):
    ax.axvline(bpe,color=c,lw=1.6,ls="--"); ax.text(bpe,1.3e-1 if bpe!=4.5 else 2e-1,n,rotation=90,va="top",ha="right",fontsize=7,color=c)
ax.plot([8.5,4.5],[0.0042,0.0595],"o",color="none")
ax.plot([8.5],[0.0042],"o",color=REF["Q8_0"]); ax.plot([4.5],[0.0595],"o",color=REF["Q4_0"]); ax.plot([16],[2e-4],"o",color=REF["F16"])
for n,b,k in kv_other: ax.plot(b,k,"s",color=OTHER); ax.annotate(n,(b,k),textcoords="offset points",xytext=(4,4),fontsize=7,color=OTHER)
ax.plot([13.8,13.5,14.5],[2e-4]*3,"^",color=OURS); ax.annotate("ours lossless\n(zlib/LZMA/EXP3)",(13.8,2e-4),textcoords="offset points",xytext=(-30,8),fontsize=7,color=OURS)
ax.set_yscale("log"); ax.set_ylim(1e-4,3e-1); ax.set_xlabel("bits per element"); ax.set_ylabel("Mean KL vs F16 KV (floor = bit-exact)")
ax.invert_xaxis(); ax.set_title("KV cache (R12; lossless = 0)",fontsize=9)
ax=axs[1]
w_ref=[("Q8_0 (ref)",8.51,1e-4,REF["Q8_0"]),("Q4_K_M (ref)",5.36,0.0602,REF["Q4_0"])]
w_other=[("Q6_K",6.57,0.0106),("Q5_K_M",5.95,0.0206),("Q3_K_M",4.60,0.2405),("Q2_K",4.03,1.849)]
for n,b,k,c in w_ref: ax.axvline(b,color=c,lw=1.6,ls="--"); ax.plot(b,k,"o",color=c); ax.text(b,2.5,n,rotation=90,va="top",ha="right",fontsize=7,color=c)
for n,b,k in w_other: ax.plot(b,k,"s",color=OTHER); ax.annotate(n,(b,k),textcoords="offset points",xytext=(4,4),fontsize=7,color=OTHER)
ax.plot([5.36*1.0061],[1e-4],"^",color=OURS); ax.annotate("ours CQW2 (lossless,\n+0.61% on Q4_K)",(5.4,1e-4),textcoords="offset points",xytext=(6,10),fontsize=7,color=OURS)
ax.set_yscale("log"); ax.set_ylim(1e-4,3); ax.set_xlabel("bits per weight"); ax.set_ylabel("Mean KL vs Q8_0 (floor = lossless)")
ax.invert_xaxis(); ax.set_title("Weights (R7, Qwen3.5-2B; lossless = 0)",fontsize=9)
save(fig,"fig3_kl_vs_bits")

# Fig4: lossless sizes vs F16. References as lines (F16 1.0, Q8_0 0.53 lossy, Q4_0 0.28 lossy); ours = bars
names=["zlib planes\n(CPU)","LZMA planes\n(CPU)","EXP3 fixed-rate\n(CPU est.)","EXP3 on GPU\nlayout (real)","Entropy ceiling\n(fields)"]
vals=[0.864,0.847,0.907,0.945,0.865]
fig,ax=plt.subplots(figsize=(6.4,3.6))
ax.bar(names,vals,color=[OURS]*4+["#bbbbbb"])
for i,v in enumerate(vals): ax.text(i,v+0.012,f"{v:.2f}",ha="center",fontsize=8)
for y,n,c in ((1.0,"F16 = 1.00 (reference)",REF["F16"]),(0.53,"Q8_0 = 0.53 (lossy reference)",REF["Q8_0"]),(0.28,"Q4_0 = 0.28 (lossy reference)",REF["Q4_0"])):
    ax.axhline(y,color=c,lw=1.6,ls="--"); ax.text(4.45,y+0.015,n,ha="right",fontsize=7,color=c)
ax.set_ylim(0,1.12); ax.set_ylabel("Size relative to F16 KV"); ax.set_title("Lossless F16 KV (ours, green) vs reference formats",fontsize=9)
plt.xticks(fontsize=7); save(fig,"fig4_lossless")

# Fig5: analytic roofline (*): references = Q8-class, Q4-class predictions as dashed lines; ours lossless solid green
ctxs=[4096,16384,32768,131072]
fig,ax=plt.subplots(figsize=(5.6,3.5))
ax.axhline(1,color=REF["F16"],lw=1.8,ls="--",label="F16 KV (reference = 1.0)")
ax.plot(ctxs,[1.10,1.30,1.45,1.71],"--",color=REF["Q8_0"],lw=1.8,label="Q8-class (reference, predicted)")
ax.plot(ctxs,[1.16,1.55,1.90,2.75],"--",color=REF["Q4_0"],lw=1.8,label="Q4-class (reference, predicted)")
ax.plot(ctxs,[1.03,1.07,1.10,1.14],"o-",color=OURS,lw=2,label="Ours: lossless 0.86 (predicted)")
ax.plot([4096,32768,65536,122880],[0.883,0.732,0.684,0.629],"x",color=REF["Q8_0"],ms=8,mew=2,label="Q8_0 measured (R25)")
ax.plot([4096,32768,65536,122880],[0.889,0.725,0.668,0.622],"+",color=REF["Q4_0"],ms=9,mew=2,label="Q4_0 measured (R25)")
ax.set_xscale("log",base=2); ax.set_xticks(ctxs); ax.set_xticklabels(["4K","16K","32K","128K"]); ax.set_ylim(0.5,3.0)
ax.set_xlabel("Context"); ax.set_ylabel("Decode speedup vs F16 (x)"); ax.set_title("* Bandwidth-only prediction vs measured (RTX 3090)",fontsize=9)
ax.legend(frameon=False,fontsize=7,loc="upper left"); save(fig,"fig5_roofline_prediction")

# Fig6: read microbenchmark (R22): references F16 = 1, Q8_0, Q4_0 lines; ours EXP3
T=[4096,32768,65536,122880]
fig,ax=plt.subplots(figsize=(5.6,3.5))
ax.axhline(1,color=REF["F16"],lw=1.8,ls="--",label="F16 KV read (reference = 1.0)")
ax.plot(T,[1.07,1.01,1.00,0.99],"-",color=REF["Q8_0"],lw=1.8,label="Q8_0 (reference)")
ax.plot(T,[1.12,1.02,1.08,1.01],"-",color=REF["Q4_0"],lw=1.8,label="Q4_0 (reference)")
ax.plot(T,[0.73,0.68,0.65,0.65],"o-",color=OURS,lw=2,label="Ours: lossless EXP3 (bit-exact)")
ax.set_xscale("log",base=2); ax.set_xticks(T); ax.set_xticklabels(["4K","32K","64K","120K"]); ax.set_ylim(0.5,1.4)
ax.set_xlabel("Tokens read (one layer)"); ax.set_ylabel("Read speed vs F16 (x, >1 faster)"); ax.set_title("PoC-B CUDA microbenchmark (R26, near-clean GPU)",fontsize=9)
ax.legend(frameon=False,fontsize=7,loc="lower left"); save(fig,"fig6_read_microbench")
# Fig7/Fig8: BASE vs BASE + OURS. Each panel: baseline = dashed lines at size 1.0 / speed 1.0;
# ours = green points (filled = lossless, hollow = lossy); "better" = smaller (left) and faster (up).
def base_panel(ax, title, pts, xlim=(0.7, 1.3), ylim=(0.6, 1.3)):
    ax.axvline(1.0, color=REF["F16"], lw=1.4, ls="--"); ax.axhline(1.0, color=REF["F16"], lw=1.4, ls="--")
    ax.plot([1.0], [1.0], "o", color=REF["F16"], ms=8); ax.annotate("baseline", (1.0, 1.0), textcoords="offset points", xytext=(5, -12), fontsize=7)
    nm = 0
    for name, x, y, lossless in pts:
        if y is None:  # size measured, speed not measured: staggered tick marks along the bottom
            yy = ylim[0] + (0.03 + 0.06 * nm) * (ylim[1] - ylim[0])
            ax.plot([x], [yy], "|", color=OURS, ms=10, mew=2)
            ax.annotate(name + " (speed n/m)", (x, yy), textcoords="offset points", xytext=(4, -3), fontsize=6, color=OURS)
            nm += 1
            continue
        ax.plot([x], [y], "o" if lossless else "D", color=OURS, mfc=OURS if lossless else "white", mew=1.8, ms=7)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=6.5, color=OURS)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_title(title, fontsize=9)
    ax.set_xlabel("size vs baseline (x)"); ax.set_ylabel("speed vs baseline (x)")
    c = "#2e7d32"
    ax.annotate("", xy=(0.05, 0.95), xytext=(0.25, 0.75), xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", color=c, lw=2))
    ax.text(0.06, 0.70, "better", transform=ax.transAxes, fontsize=8, color=c, fontweight="bold")

fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.0))
base_panel(axs[0], "KV base F16 (read microbenchmark, 32K)", [
    ("EXP3 lossless", 0.931, 0.68, True), ("zlib planes lossless", 0.864, None, True), ("LZMA planes lossless", 0.847, None, True),
    ("drop4 lossy", 0.797, 0.98, False)], xlim=(0.7, 1.1))
base_panel(axs[1], "KV base Q8_0 (32K)", [
    ("7-bit codes lossy", 0.912, 0.98, False), ("LZMA planes lossless (K)", 0.892, None, True),
    ("fixed 7b+esc lossless (K)", 1.063, None, True), ("6b+1% outliers lossy", 0.802, None, False)], xlim=(0.7, 1.15))
base_panel(axs[2], "KV base Q4_0 (32K)", [
    ("fixed 3b+esc lossless", 1.240, 0.75, True), ("LZMA planes lossless (K)", 0.790, None, True),
    ("3b+1% outliers lossy", 0.849, None, False)], xlim=(0.7, 1.3))
fig.text(0.5, 0.005, "Filled = lossless, hollow = lossy. Speed = GPU read microbenchmark of one layer (R33). KL of our KV variants not measured.",
         ha="center", fontsize=7)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig("figures/fig7_kv_base_vs_ours.png"); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.0))
base_panel(axs[0], "Weights base BF16 (decode, depth 0)", [
    ("FFN->Q8_0 lossy\nKL 0.0013, tasks 28 vs 27", 0.770, 1.195, False), ("exponent Huffman lossless", 0.666, None, True)], xlim=(0.55, 1.1), ylim=(0.8, 1.3))
base_panel(axs[1], "Weights base Q8_0", [
    ("FFN->Q6_K lossy\nKL 0.0046, tasks 29 vs 28", 0.889, 1.055, False), ("split zlib lossless", 0.944, None, True)],
    xlim=(0.8, 1.1), ylim=(0.8, 1.3))
base_panel(axs[2], "Weights base Q4_K_M", [
    ("emb->Q4_K lossy\nKL x1.27, tasks 28 vs 27", 0.899, 1.110, False), ("FFN up/gate->Q3_K lossy\nKL x2.1, tasks 25", 0.936, 0.982, False),
    ("CQW2 lossless", 1.006, None, True), ("split zlib lossless", 0.978, None, True)], xlim=(0.8, 1.1), ylim=(0.8, 1.3))
fig.text(0.5, 0.005, "Qwen3.5-2B, llama.cpp decode on RTX 3090 (R34). Lossless weight formats have no GPU decoder (speed not measured).",
         ha="center", fontsize=7)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig("figures/fig8_weights_base_vs_ours.png"); plt.close(fig)
print("ok")
