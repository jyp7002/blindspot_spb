from huggingface_hub import snapshot_download
for m in ["meta-llama/Llama-3.2-1B-Instruct","meta-llama/Llama-3.2-3B-Instruct","google/gemma-2-2b-it"]:
    print("=== "+m,flush=True)
    snapshot_download(m, allow_patterns=["*.json","*.safetensors","*.txt","*.model"], max_workers=8)
    print("done "+m,flush=True)
