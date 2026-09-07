from huggingface_hub import snapshot_download
for m in ["HuggingFaceTB/SmolLM2-360M-Instruct","microsoft/Phi-3-mini-4k-instruct","google/gemma-2-9b-it"]:
    print("=== "+m,flush=True)
    snapshot_download(m, allow_patterns=["*.json","*.safetensors","*.txt","*.model"], max_workers=8)
    print("done "+m,flush=True)
