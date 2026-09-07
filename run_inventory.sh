export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
cd /home/jovyan/Blind_spot_spb/src
python inventory.py --n-bbq 600 --n-ss 500 --n-crows 300 --batch-size 32 \
  --models qwen0.5b qwen1.5b qwen3b smol1.7b phi3.5 llama1b llama3b gemma2b
