from huggingface_hub import snapshot_download
for m in ["Qwen/Qwen2.5-7B-Instruct","meta-llama/Llama-3.1-8B-Instruct"]:
    print("=== "+m,flush=True)
    snapshot_download(m, allow_patterns=["*.json","*.safetensors","*.txt","*.model","tokenizer*"], max_workers=8)
    print("done "+m,flush=True)
