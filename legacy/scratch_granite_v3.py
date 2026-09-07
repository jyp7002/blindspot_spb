"""V3: granite T2 all-nan diagnosis — per-alpha collateral breakdown."""
import sys; sys.path.insert(0,"/home/jovyan/Blind_spot_spb")
import colab_t2t4 as C
import numpy as np, torch, json

tid="ibm-granite/granite-3.1-8b-instruct"
mmlu=C.load_mmlu(n=200,seed=0); wt=C.load_wikitext(n_chunks=20,seed=0)
m,t=C.load_model(tid, dispatch=False)
diag={}
try:
    for axis in ["occ_gender","crows_socioeconomic"]:
        pre=C.fast_eval(m,t,axis,mmlu,wt,12)
        el=C.elicit_selfdebias(m,t,axis,0,12)
        dwb,_=C.train_task_vector(m,t,el["biased"],rank=16,steps=250,lr=1e-4,seed=0,bs=6,targets=C.ATTN,grad_checkpoint=True)
        dwd,_=C.train_task_vector(m,t,el["debiased"],rank=16,steps=250,lr=1e-4,seed=0,bs=6,targets=C.ATTN,grad_checkpoint=True)
        E,_=C.binarize(C.contrast(dwb,dwd),"per_tensor",0.0,0)
        print(f"\n=== {axis} ===  pre: skew={pre['skew']:+.3f} mmlu={pre['mmlu_acc']:.3f} ppl={pre['ppl']:.2f}",flush=True)
        rows=[]
        for a in (2,4,8,16):
            undo=C.apply_edit(m,E,alpha=a,sign=-1.0)
            try: post=C.fast_eval(m,t,axis,mmlu,wt,12)
            finally: undo()
            dmmlu=pre['mmlu_acc']-post['mmlu_acc']; pplr=post['ppl']/pre['ppl']
            rem=abs(pre['skew'])-abs(post['skew'])
            ok=(dmmlu<=0.02) and (pplr<=1.10)
            binds=[]
            if dmmlu>0.02: binds.append(f"MMLU drop {dmmlu:+.3f}")
            if pplr>1.10: binds.append(f"ppl ratio {pplr:.3f}")
            print(f"  a={a:2d}: skew={post['skew']:+.3f} rem={rem:+.3f} mmlu={post['mmlu_acc']:.3f}(d{dmmlu:+.3f}) ppl={post['ppl']:.2f}(x{pplr:.3f}) {'IN-BUDGET' if ok else 'FAIL: '+', '.join(binds)}",flush=True)
            rows.append(dict(alpha=a,rem=rem,dmmlu=dmmlu,pplr=pplr,ok=ok))
        diag[axis]=dict(pre=pre,rows=rows)
finally:
    C.load_model  # noop
    import gc; 
    try: m.to("meta")
    except: pass
    del m; gc.collect(); torch.cuda.empty_cache()
json.dump(diag,open("results/granite_v3_diag.json","w"),indent=2,default=float)
print("\nwrote results/granite_v3_diag.json")
