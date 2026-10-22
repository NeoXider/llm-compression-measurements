# Server-class roofline: Llama-70B-like (80 layers, 8 KV heads, hd128), weights 4-bit (35 GB read/token step), batch B sequences.
# Per step: weights read once, KV read B times. t = max(t_mem, t_compute). Specs approximate public figures.
GB=1e9
GPUS={ # bw GB/s, dense fp16 TFLOPS, vram GB
 "RTX 3090":(936,71,24), "RTX 5090":(1792,210,32), "L40S":(864,362,48),
 "RTX PRO 6000 Blackwell":(1792,250,96), "A100 80GB":(2039,312,80),
 "H100 SXM":(3350,989,80), "H200":(4800,989,141), "B200":(8000,2250,192)}
KV_TOK=2*80*8*128*2     # 327680 B per token
W=35*GB; PARAMS=70e9; EFF=0.6
def step(g,B,ctx,rw=1,rkv=1,kvbits=16):
    bw,tf,_=GPUS[g]; kv=KV_TOK*ctx*kvbits/16*B
    tm=(W*rw+kv*rkv)/(bw*GB*EFF); tc=2*PARAMS*B/(tf*1e12*0.5)
    return B/max(tm,tc), tm>=tc          # tokens/s total, memory-bound?
print("Llama-70B-class, 4-bit weights 35GB. total tok/s (all sequences) and speedup vs plain; M=memory-bound C=compute-bound")
for g in ("H100 SXM","H200","B200","A100 80GB"):
    print("\n==",g, "VRAM",GPUS[g][2],"GB")
    for B,ctx in ((1,8192),(16,8192),(64,8192),(64,32768),(256,8192)):
        base,mb=step(g,B,ctx); row=f"B={B:>3} ctx={ctx:>5} base={base:8.0f}{'M' if mb else 'C'}"
        for name,rw,rkv,kb in (("KV lossless .86",1,.86,16),("W+KV lossless",.934,.86,16),("KV Q8-class",1,1,8.5),("KV Q4-class",1,1,4.5),("W.75+KV Q4",.75,1,4.5)):
            s,_=step(g,B,ctx,rw,rkv,kb); row+=f" | {name}: {s/base:4.2f}x"
        print(row)
print("\nKV memory for batch (GB): 70B-class F16, ctx 8K x B=64 =", round(KV_TOK*8192*64/GB,0),"GB ; 32K x 64 =",round(KV_TOK*32768*64/GB,0),"GB")
for g in ("H100 SXM","H200","B200"):
    free=GPUS[g][2]*GB-W-5*GB
    for name,kb,r in (("F16",16,1),("F16 lossless .86",16,.86),("Q8",8.5,1),("Q4",4.5,1)):
        print(g,name,"max resident tokens", int(free/(KV_TOK*kb/16*r)))
