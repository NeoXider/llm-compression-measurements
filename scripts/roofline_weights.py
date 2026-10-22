# Weights-side model (batch-1 decode is memory-bound: tok/s ~ eff*BW / bytes_read_per_token)
GB=1e9; BW={"RTX 3090":936,"RTX 5090":1792}; VRAM={"RTX 3090":24,"RTX 5090":32}
MODELS={"Qwen3-4B":4.0,"8B":8.0,"14B":14.0,"32B":32.0,"70B":70.0}   # billions of params
BPW={"Q8_0":8.5,"Q6_K":6.6,"Q5_K_M":5.7,"Q4_K_M":4.85,"Q3_K_M":3.9,"IQ2-class":2.5}  # approximate bits per weight
EFF=0.5; OVERHEAD=1.5   # GB for KV(short ctx)+buffers
def size(p,bpw,r=1.0): return p*1e9*bpw/8*r/GB
print("Weights size (GB) and decode tok/s on RTX 3090 (eff 0.5), '-' = does not fit 24GB with 1.5GB overhead")
print(f"{'model/quant':18} {'GB':>6} {'plain':>8} {'lossless x0.934':>16} {'lossy x0.75':>12}")
for m,p in MODELS.items():
    for q,b in BPW.items():
        if q in("Q4_K_M","Q5_K_M","Q8_0","Q3_K_M") or m=="Qwen3-4B":
            row=f"{m+' '+q:18} {size(p,b):6.1f}"
            for r in (1.0,0.934,0.75):
                s=size(p,b,r)
                fit=s+OVERHEAD<=24
                row+=f" {EFF*936*GB/(s*GB):8.1f}" if fit else f" {'-':>8}"
                if r==0.934: row+="       "
            print(row)
print("\nWhat compression ratio of Q4_K_M is needed to fit a model in 24GB (1.5GB overhead):")
for m,p in MODELS.items():
    s=size(p,4.85); need=(24-OVERHEAD)/s
    print(f"{m:10} Q4_K_M {s:5.1f} GB -> ratio needed {need:4.2f}", "(fits)" if need>=1 else "")
print("\nPCIe-offloaded 70B Q4_K_M on 3090 (24GB): fraction streamed =", round(1-(24-OVERHEAD)/size(70,4.85),2))
s=size(70,4.85); f=1-(24-OVERHEAD)/s
for r in (1.0,0.934,0.75,0.5):
    t=f*s*r/25 + (1-f)*s*r/(0.5*936)
    print(f"  ratio {r}: ~{1/t:.1f} tok/s (PCIe 25GB/s eff, GPU decode ALU hidden)")
