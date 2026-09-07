from huggingface_hub import snapshot_download
# 4th Qwen sibling tightens the within-family d_axis estimate
snapshot_download("Qwen/Qwen2.5-7B-Instruct",
                  allow_patterns=["*.json","*.safetensors","*.txt","*.model"], max_workers=8)
print("done qwen7b", flush=True)
