#!/home/jerzy/ov_monitor/.venv/bin/python3
"""
Run this standalone (server stopped, or on a different model slot) to see
exactly what pipe.generate() returns and which calling convention works.

Usage:
    python debug_generate.py
"""
import openvino_genai as ov_genai
import os, inspect, textwrap

MODELS_DIR = os.path.expanduser("~/ov_models")
MODEL_ID   = "qwen2.5-3b-int4"          # use the small one for speed
MODEL_PATH = f"{MODELS_DIR}/{MODEL_ID}"
DEVICE     = "GPU.1"
CONFIG     = {"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": "/tmp/ov_cache_b60"}

PROMPT = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nSay hello.<|im_end|>\n<|im_start|>assistant\n"

print(f"openvino_genai version : {ov_genai.__version__}")
print(f"Loading {MODEL_ID} on {DEVICE} …")
pipe = ov_genai.LLMPipeline(MODEL_PATH, DEVICE, **CONFIG)
print("Loaded.\n")

# --- inspect generate() signature ---
try:
    sig = inspect.signature(pipe.generate)
    print(f"pipe.generate signature: {sig}\n")
except Exception as e:
    print(f"Could not inspect signature: {e}\n")

# --- attempt 1: positional GenerationConfig ---
print("=== Attempt 1: pipe.generate(prompt, GenerationConfig) ===")
try:
    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = 30
    cfg.temperature    = 0.7
    cfg.do_sample      = True
    raw = pipe.generate(PROMPT, cfg)
    print(f"  type   : {type(raw)}")
    print(f"  repr   : {repr(raw)[:200]}")
    print(f"  str()  : {str(raw)[:200]}")
    if hasattr(raw, "texts"):
        print(f"  .texts : {raw.texts}")
    if hasattr(raw, "__dict__"):
        print(f"  __dict__: {raw.__dict__}")
except Exception as e:
    print(f"  FAILED : {e}")

print()

# --- attempt 2: keyword args ---
print("=== Attempt 2: pipe.generate(prompt, max_new_tokens=30, temperature=0.7, do_sample=True) ===")
try:
    raw = pipe.generate(PROMPT, max_new_tokens=30, temperature=0.7, do_sample=True)
    print(f"  type   : {type(raw)}")
    print(f"  repr   : {repr(raw)[:200]}")
    print(f"  str()  : {str(raw)[:200]}")
    if hasattr(raw, "texts"):
        print(f"  .texts : {raw.texts}")
except Exception as e:
    print(f"  FAILED : {e}")

print()

# --- attempt 3: greedy (no sampling) ---
print("=== Attempt 3: pipe.generate(prompt, max_new_tokens=30) greedy ===")
try:
    raw = pipe.generate(PROMPT, max_new_tokens=30)
    print(f"  type   : {type(raw)}")
    print(f"  repr   : {repr(raw)[:200]}")
    print(f"  str()  : {str(raw)[:200]}")
    if hasattr(raw, "texts"):
        print(f"  .texts : {raw.texts}")
except Exception as e:
    print(f"  FAILED : {e}")

print()

# --- attempt 4: GenerationConfig via keyword ---
print("=== Attempt 4: pipe.generate(prompt, generation_config=cfg) ===")
try:
    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = 30
    raw = pipe.generate(PROMPT, generation_config=cfg)
    print(f"  type   : {type(raw)}")
    print(f"  repr   : {repr(raw)[:200]}")
    if hasattr(raw, "texts"):
        print(f"  .texts : {raw.texts}")
except Exception as e:
    print(f"  FAILED : {e}")

print("\nDone.")
