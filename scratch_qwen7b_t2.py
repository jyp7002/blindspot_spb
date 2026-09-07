import sys, traceback; sys.path.insert(0, "/home/jovyan/Blind_spot_spb")
import colab_t2t4 as C
mmlu = C.load_mmlu(n=64, seed=0); wt = C.load_wikitext(n_chunks=8, seed=0)
try:
    rows = C.run_t2("qwen", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-3B-Instruct",
                    mmlu, wt, seeds=(0,), batch_size=12, train_bs=6)
    print("ROWS:", rows)
    assert rows, "no rows!"
    print("QWEN7B T2 OK — hardened code works at real 7B scale")
except Exception as e:
    print("QWEN7B T2 FAILED:", type(e).__name__, e); traceback.print_exc()
