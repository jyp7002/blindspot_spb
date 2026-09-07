"""Y (reframed, local): base<->instruct robustness of the family bias signature.
The whole registry is INSTRUCT; 'base-only' invariant is not met anywhere. Test
whether instruct tuning distorts the geometry on gemma small models."""
import sys, os, json; sys.path.insert(0,"/home/jovyan/Blind_spot_spb/src")
import numpy as np, torch, gc
from transformers import AutoModelForCausalLM, AutoTokenizer
import geometry
AXES=["occ_gender","crows_socioeconomic","crows_race-color","crows_gender","crows_religion","crows_age"]
PROF=os.path.abspath("results/profiles")
def load(hf):
    tok=AutoTokenizer.from_pretrained(hf); 
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    tok.padding_side="left"
    m=AutoModelForCausalLM.from_pretrained(hf,dtype=torch.bfloat16,low_cpu_mem_usage=True).to("cuda").eval()
    return m,tok
def corr(a,b):
    a,b=a-a.mean(),b-b.mean();n=np.linalg.norm(a)*np.linalg.norm(b);return float(a@b/n) if n>0 else float('nan')
BASE={"gemma2b":"google/gemma-2-2b","gemma9b":"google/gemma-2-9b"}
os.makedirs("results/profiles_base",exist_ok=True)
prof_base={}
for name,hf in BASE.items():
    m,t=load(hf)
    try:
        for ax in AXES:
            p,_=geometry.bias_profile(m,t,ax,batch_size=8)
            np.save(f"results/profiles_base/{name}|{ax}.npy",p)
            prof_base[(name,ax)]=p
    finally:
        try:m.to("meta")
        except:pass
        del m; gc.collect(); torch.cuda.empty_cache()
    print(f"[y] profiled {name} base",flush=True)
# compare base vs -it (existing) per model per axis; and same-family Δd base vs it
print("\n=== base<->it profile correlation (per model, per axis) ===")
rows=[]
for name in BASE:
    cs=[]
    for ax in AXES:
        it=np.load(f"{PROF}/{name}|{ax}.npy")
        b=prof_base[(name,ax)]
        c=corr(it,b); cs.append(c)
        print(f"  {name:8s} {ax:22s} base<->it r={c:+.3f}")
    print(f"  {name} mean base<->it r = {np.nanmean(cs):+.3f}")
    rows.append((name,float(np.nanmean(cs))))
# same-family d: gemma2b<->gemma9b, base-only vs it-only
def famd(getter):
    cs=[corr(getter("gemma2b",ax),getter("gemma9b",ax)) for ax in AXES]
    return float(np.nanmean([c for c in cs if not np.isnan(c)]))
d_base=famd(lambda n,ax:prof_base[(n,ax)])
d_it=famd(lambda n,ax:np.load(f"{PROF}/{n}|{ax}.npy"))
print(f"\n=== same-family gemma d (2b<->9b): base={d_base:+.3f}  it={d_it:+.3f} ===")
json.dump(dict(base_vs_it=rows,same_family_d_base=d_base,same_family_d_it=d_it),
          open("results/y_base_check.json","w"),indent=2)
print("wrote results/y_base_check.json")
