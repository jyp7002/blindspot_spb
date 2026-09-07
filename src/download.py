from huggingface_hub import snapshot_download
MODELS=["Qwen/Qwen2.5-0.5B-Instruct","Qwen/Qwen2.5-1.5B-Instruct","Qwen/Qwen2.5-3B-Instruct",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct","microsoft/Phi-3.5-mini-instruct"]
for m in MODELS:
    print("=== "+m,flush=True)
    snapshot_download(m, allow_patterns=["*.json","*.safetensors","*.txt","*.model"], max_workers=8)
    print("done "+m,flush=True)
