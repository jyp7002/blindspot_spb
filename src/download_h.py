from huggingface_hub import snapshot_download
MODELS=[
 "allenai/OLMo-2-0425-1B-Instruct","allenai/OLMo-2-1124-7B-Instruct",
 "tiiuae/Falcon3-1B-Instruct","tiiuae/Falcon3-3B-Instruct",
 "ibm-granite/granite-3.1-2b-instruct","ibm-granite/granite-3.1-8b-instruct",
]
for m in MODELS:
    print("=== "+m,flush=True)
    snapshot_download(m, allow_patterns=["*.json","*.safetensors","*.txt","*.model","tokenizer*"], max_workers=8)
    print("done "+m,flush=True)
