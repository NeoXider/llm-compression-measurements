# Roofline model for decode with compressed weights/KV. Card specs are approximate public figures.
GB=1e9
CARDS={ # bw GB/s, int32 Tops (decode ALU budget), pcie eff GB/s, vram GB
 "RTX 3090":   dict(bw=936,  intops=17.8e12, pcie=25, vram=24),
 "RTX 5090":   dict(bw=1792, intops=104e12,  pcie=50, vram=32),
 "B200 (dc)":  dict(bw=8000, intops=40e12,   pcie=50, vram=192),
}
# Qwen3-4B Q4_K_M: 36 layers, 8 KV heads, head_dim 128
KV_PER_TOK_F16=2*36*8*128*2           # 147456 B
W=2.5*GB                               # Q4_K_M weights read per token (approx)
PARAMS=4.0e9
def decode(card,ctx,r_w=1.0,r_kv=1.0,kv_bits=16,d_w=0,d_kv=0,eff=0.5,overlap=True):
    c=CARDS[card]
    kv=KV_PER_TOK_F16*ctx*kv_bits/16
    bytes_=W*r_w+kv*r_kv
    t_mem=bytes_/(c["bw"]*GB*eff)
    # decode ALU cost: elements decoded * ops each
    el_w=PARAMS; el_kv=KV_PER_TOK_F16/2*ctx*kv_bits/16
    t_dec=(el_w*d_w+el_kv*d_kv)/c["intops"]
    t=max(t_mem,t_dec) if overlap else t_mem+t_dec
    return 1/t, t_mem*1e3, t_dec*1e3
if __name__=="__main__":
    print("KV bytes/token F16:",KV_PER_TOK_F16," KV@16K F16 GiB:",round(KV_PER_TOK_F16*16384/2**30,2))
    for card in CARDS:
        c=CARDS[card]
        print("\n==",card,"balance FLOP/B irrelevant; mem-bound decode. ctx tok/s (baseline F16 KV, eff 0.5)")
        for ctx in (4096,16384,32768,131072):
            base=decode(card,ctx)[0]
            rows=[("F16 KV",1,1,16,0,0)]
            out=[f"ctx={ctx:>6} base={base:7.1f} tok/s"]
            for name,rw,rkv,kb,dw,dk in [
              ("lossless KV 0.86 (10 ops/el)",1,0.86,16,0,10),
              ("lossless KV+W 0.86/0.934",0.934,0.86,16,6,10),
              ("Q8 KV (ref)",1,1.0,8.5,0,0),
              ("Q8 KV lossless-coded 0.83",1,0.83,8.5,0,8),
              ("Q4 KV (ref)",1,1.0,4.5,0,0),
              ("Q4 KV lossless-coded 0.85",1,0.85,4.5,0,8)]:
                s=decode(card,ctx,rw,rkv,kb,dw,dk)[0]
                out.append(f"  {name}: {s:7.1f} ({s/base:5.2f}x)")
            print("\n".join(out))
        # not-overlapped decode (naive): decompress full KV each token with 40 ops/el
        s=decode(card,16384,1,0.86,16,0,40,overlap=False)[0]; print("  naive no-overlap 40 ops/el @16K:",round(s,1),"tok/s", round(s/decode(card,16384)[0],2),"x")
    print("\n== PCIe offload: fraction f of weights streamed each token")
    for card in ("RTX 3090","RTX 5090"):
        c=CARDS[card]
        for f in (0.1,0.3,0.5):
            tp=f*W/(c["pcie"]*GB); tg=(1-f)*W/(c["bw"]*GB*0.5)
            for r in (1.0,0.934,0.7,0.5):
                print(card,f"f={f}",f"ratio={r}", f"tok/s={1/(tp*r+tg+f*W*r*4/c['intops']):.1f}") if r in (1.0,0.934,0.5) else None
    print("\n== VRAM capacity: max context with 24GB-class, W+1.0GB overhead")
    for card in CARDS:
        c=CARDS[card]; free=c["vram"]*GB-W-1.0*GB
        for name,kb,r in [("F16",16,1),("F16 lossless 0.86",16,0.86),("Q8",8.5,1),("Q4",4.5,1)]:
            print(card,name,int(free/(KV_PER_TOK_F16*kb/16*r)),"tokens")
    print("\n== CPU-side decode cannot feed PCIe: zlib 240MB/s vs PCIe 25GB/s ->",round(25000/240),"x too slow")
