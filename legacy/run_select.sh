export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
cd /home/jovyan/Blind_spot_spb/src
python select_acquired.py && python verify_axes.py
