# Three-type comparison figures. Colors are FIXED everywhere:
#   gray  = base (no compression)      green  = ours LOSSLESS      orange = ours LOSSY
# Every number is from COMPARISON_TABLE.md / RESULTS.md (R33-R35). "n/m" = not measured.
import numpy as np, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150})
BASE,LL,LY="#8c8c8c","#1b9e77","#e66101"
os.makedirs("figures",exist_ok=True)
G="#2e7d32"
def arrow(ax,kind):
    if kind=="up":   ax.annotate("",xy=(0.97,0.95),xytext=(0.97,0.78),xycoords="axes fraction",arrowprops=dict(arrowstyle="-|>",color=G,lw=2)); ax.text(0.94,0.865,"better",transform=ax.transAxes,ha="right",va="center",fontsize=8,color=G,fontweight="bold")
    else:            ax.annotate("",xy=(0.97,0.74),xytext=(0.97,0.92),xycoords="axes fraction",arrowprops=dict(arrowstyle="-|>",color=G,lw=2)); ax.text(0.94,0.83,"better",transform=ax.transAxes,ha="right",va="center",fontsize=8,color=G,fontweight="bold")
def groups(ax,labels,series,ylabel,title,better,ylim,hline=1.0,fmt="{:.2f}",proto=None):
    """series: list of (values, color); a value None = not measured; value tuple (lo,hi) = range"""
    x=np.arange(len(labels)); w=0.26
    for si,(vals,color,name) in enumerate(series):
        for i,v in enumerate(vals):
            xp=x[i]+(si-1)*w
            if v is None:
                ax.text(xp,ylim[0]+(ylim[1]-ylim[0])*0.03,"n/m",ha="center",va="bottom",fontsize=7,color="#666",rotation=90); continue
            lo=hi=v
            if isinstance(v,tuple): lo,hi=v; v=(lo+hi)/2
            hatch="//" if (proto and (si,i) in proto) else None
            ax.bar(xp,v,w,color=color,edgecolor="white",hatch=hatch)
            if hi>lo: ax.errorbar(xp,v,yerr=[[v-lo],[hi-v]],color="black",capsize=2,lw=1)
            ax.text(xp,hi+(ylim[1]-ylim[0])*0.015,fmt.format(v),ha="center",va="bottom",fontsize=7)
    if hline is not None: ax.axhline(hline,color="#444",lw=1,ls="--")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(*ylim); ax.set_ylabel(ylabel); ax.set_title(title,fontsize=9); arrow(ax,better)
def legend(fig):
    fig.legend(handles=[Patch(color=BASE,label="Base (no compression)"),Patch(color=LL,label="Ours, LOSSLESS"),Patch(color=LY,label="Ours, LOSSY"),Patch(facecolor="white",edgecolor="k",hatch="//",label="hatched = CPU prototype only")],loc="lower center",ncol=4,frameon=False,fontsize=8)

# ---------- KV cache ----------
bases=["Base F16","Base Q8_0","Base Q4_0"]
size_ll=[0.847,0.91,0.83]       # F16: LZMA planes; Q8: LZMA (K .892 / V .931 avg); Q4: LZMA (K .790 / V .869 avg)
size_ly=[0.80,0.80,0.85]        # F16 -4 mantissa bits; Q8 6 bit + 1% outliers; Q4 3 bit + 1% outliers
speed_ll=[(0.65,0.73),None,(0.73,0.80)]   # GPU-readable formats (F16 EXP3; Q4 3bit+exc); Q8 GPU format is larger than base
speed_ly=[(0.85,0.98),(0.94,1.00),None]
err_ll=[1.0,1.0,1.0]            # identical (bit-exact)
err_ly=[None,(2.1,3.3),(1.5,2.1)]
fig,axs=plt.subplots(1,3,figsize=(12.5,4.0))
groups(axs[0],bases,[([1,1,1],BASE,"b"),(size_ll,LL,"l"),(size_ly,LY,"y")],"Size relative to own base","KV cache: SIZE","down",(0,1.5),proto={(1,0),(1,1),(1,2)})
groups(axs[1],bases,[([1,1,1],BASE,"b"),(speed_ll,LL,"l"),(speed_ly,LY,"y")],"Read speed relative to own base","KV cache: SPEED (GPU read microbenchmark)","up",(0,1.5))
groups(axs[2],bases,[([1,1,1],BASE,"b"),(err_ll,LL,"l"),(err_ly,LY,"y")],"Error relative to the base's own error","KV cache: QUALITY (1.0 = same as base)","down",(0,4.0),fmt="{:.1f}x")
axs[2].text(0.02,0.92,"F16 lossy: abs. error 0.6% (base error = 0, ratio undefined)",transform=axs[2].transAxes,fontsize=6.5,color="#555")
fig.tight_layout(rect=(0,0.07,1,1)); legend(fig); fig.savefig("figures/fig9_kv_three_types.png"); plt.close(fig)

# ---------- Weights (Qwen3.5-2B) ----------
wb=["Base BF16","Base Q8_0","Base Q4_K_M"]
w_size_ll=[0.666,0.944,0.978]; w_size_ly=[0.77,0.89,0.90]
w_speed_ly=[(1.167,1.195),(1.036,1.055),(1.048,1.110)]
w_kl_base=[0.0013,1e-4,0.0602]; w_kl_ly=[0.0013,0.0046,0.077]
fig,axs=plt.subplots(1,3,figsize=(12.5,4.0))
groups(axs[0],wb,[([1,1,1],BASE,"b"),(w_size_ll,LL,"l"),(w_size_ly,LY,"y")],"Size relative to own base","WEIGHTS: SIZE","down",(0,1.5),proto={(1,0),(1,1),(1,2)})
groups(axs[1],wb,[([1,1,1],BASE,"b"),([None,None,None],LL,"l"),(w_speed_ly,LY,"y")],"Decode speed relative to own base","WEIGHTS: SPEED (decode, llama.cpp; lossless has no GPU decoder)","up",(0,1.7))
ax=axs[2]; x=np.arange(3); w=0.26
ax.bar(x-w,w_kl_base,w,color=BASE); ax.bar(x,w_kl_base,w,color=LL); ax.bar(x+w,w_kl_ly,w,color=LY)
for xi,v in zip(x-w,w_kl_base): ax.text(xi,v+0.002,"0" if v<1e-3 and False else f"{v:.4f}",ha="center",fontsize=7)
for xi,v in zip(x,w_kl_base): ax.text(xi,v+0.002,"same",ha="center",fontsize=7)
for xi,v in zip(x+w,w_kl_ly): ax.text(xi,v+0.002,f"{v:.4f}",ha="center",fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(wb); ax.set_ylabel("Mean KL vs Q8_0 (Q8_0 base = 0)"); ax.set_title("WEIGHTS: QUALITY (lossless = identical to base)",fontsize=9); ax.set_ylim(0,0.1); arrow(ax,"down")
fig.tight_layout(rect=(0,0.07,1,1)); legend(fig); fig.savefig("figures/fig10_weights_three_types.png"); plt.close(fig)
print("ok")
