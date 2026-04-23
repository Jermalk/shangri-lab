# DECISIONS.md — ov_server

> Reconstructed from Claude Code session history.
> Authored collaboratively by Jerzy and Claude (Code + chat).
> This document captures *why* things are the way they are — decisions, dead ends, and non-obvious choices that don't appear in the code itself.

---

## VLM Integration

### Why `AutoTokenizer` instead of `AutoProcessor`

**Decision:** VLM loading uses `AutoTokenizer`, not `AutoProcessor`.

**Reason:** `AutoProcessor` for Qwen2.5-VL unconditionally loads a video processor subclass that imports `torchvision`. This is an unconditional import — it fires even if video is never used. Since `VLMPipeline` handles all image preprocessing internally, the processor's image/video components are never needed. Only `apply_chat_template` is needed for building the text prompt, and `AutoTokenizer` does that equally well without the dependency.

**Dead end:** Initially tried `AutoProcessor` — server failed to start due to missing torchvision.

---

### Why images are not passed as PIL Images to `VLMPipeline`

**Decision:** Images are converted to `ov.Tensor` before being passed to `VLMPipeline.generate()`.

**Reason:** `VLMPipeline.generate()` does not accept PIL Images directly — it requires `ov.Tensor` objects. Passing PIL Images causes a runtime error. Conversion step added in `_chat_vlm()` after `_decode_image()`.

**Dead end:** First implementation passed PIL Images directly — failed with a clear type error at inference time.

---

### Why image requests are routed to the VLM regardless of the model name in the request

**Decision:** `_has_images()` fires before any model lookup. If a request contains images, it always routes to `VISION_MODEL` — the requested model name is ignored.

**Reason:** Clients like AnythingLLM may send image requests with a text model name selected (user forgot to switch). Silent correct routing is better than a confusing failure. Users don't need to manually select the VLM when attaching images.

---

### Why `vlm_max_image_turns` and `vlm_max_image_side_px` exist

**Problem:** Multi-turn conversations with images caused severe VRAM growth and slowdowns. Two causes:
1. Full message history is sent to `VLMPipeline` each request — a 10-turn conversation with 3 images per turn re-encodes all 30 images on the final turn.
2. Full-resolution images (e.g. 1920×1080 screenshot) produce ~2600 vision tokens each via the patch encoding scheme (~1920/28 × 1080/28).

**Decision:** Two config-tunable mitigations:
- `vlm_max_image_turns: 1` — only images from the most recent user turn are encoded; earlier turns have images stripped to text only.
- `vlm_max_image_side_px: 1280` — images are downscaled before encoding. A 1920px image goes from ~2600 to ~1260 tokens. Text in documents remains legible at this resolution.

**Tradeoff:** Setting `vlm_max_image_turns: 1` means the model can't compare images across turns. Raise to 2–3 if that workflow is needed.

---

### Config key `vision_model`

**Decision:** `config.json` contains a `vision_model` key pointing to the VLM directory name.

**Reason:** Allows switching the active VLM without code changes. Server logs a warning at startup if the configured model directory doesn't exist (e.g. model not yet converted).

---

### Model aliases and VLM

**Investigated:** Whether model aliases could be used to route image requests to the VLM.

**Conclusion:** Not needed. Since routing is content-driven (`_has_images()`), not name-driven, aliases don't affect VLM routing. An image request sent as `gpt-4o` still goes to the VLM correctly.

---

### VLM prompt construction

**Decision:** `build_vlm_prompt()` uses `AutoTokenizer.apply_chat_template()` with `{"type": "image"}` placeholders inserted at image positions.

**Reason:** `VLMPipeline` expects a specific interleaved format — text prompt with image token placeholders in the right positions. The Jinja chat template handles the Qwen-specific formatting.

---

## config.json

### Initial typo: double `q` in alias

**Bug found and fixed:** `"qqwen3-14b-int4-ov"` (double q) in the aliases section. The model directory was `qwen3-14b-int4-ov`. Silent failure — requests to the alias would not resolve.

### Alias key correction

**Bug found and fixed:** The alias `"qwen3-14b-int4-ov" → "qwen3-14b-int4-ov"` was a self-referential no-op. Corrected to `"qwen3-14b-int4" → "qwen3-14b-int4-ov"` — maps the non-OV variant name to the installed OV model.

---

## Prompt token counting with VLMs

**Known limitation:** `prompt_tokens` in the response reflects only text tokens — vision tokens processed internally by `VLMPipeline` are not exposed and not counted. This is a VLMPipeline limitation, not a server bug.

---

## Shell / curl quirks encountered

**Problem:** `curl` with base64-encoded images in `-d` argument fails with "argument too long" (OS limit on argument length).

**Solution:** Write payload to a file, use `curl -d @/tmp/payload.json`. The `@filename` syntax bypasses the argument length limit entirely.

---

## Model Loading & Caching

### Why `MAX_LOADED_MODELS = 2` and LRU eviction

**Decision:** At most 2 models resident in VRAM simultaneously. LRU eviction when a 3rd would load.

**Reason:** With a 24 GB GPU, qwen3-14b-int4 (~9 GB) + qwen3-8b-int4 (~4.6 GB) fit comfortably together. A hard cap of 2 prevents uncontrolled growth if more models are added later.

**Two eviction triggers:**
- Hard cap: count ≥ `MAX_LOADED_MODELS` → evict LRU regardless of VRAM state
- Soft cap: free VRAM − model_size_estimate < 1.5 GB → evict LRU even under count limit
- VRAM query via `GPU_MEMORY_STATISTICS` — silently skipped if unavailable (older OV build)

**LRU tracking:** every cache hit updates `model_last_used[model_id]` so the most recently used model is always last to be evicted.

---

### Early mistake: unconditional eviction on every load

**Dead end:** First implementation cleared `loaded_models` entirely before loading any model. This meant agent calls (3b) always flushed the 14b, causing ~84s reload on every subsequent chat turn.

**Fix:** Removed the unconditional eviction. Both models now load once and stay resident. The LRU cap is the only guard.

---

### The three-model-identity thrashing problem

**Problem discovered in production:** AnythingLLM sends `"model": "qwen3-14b-int4"` hardcoded in workspace settings. After changing `default_model` to `qwen2.5-coder-14b-int4`, the cache was seeing three distinct identities: `qwen2.5-coder-14b-int4` (default), `qwen3-14b-int4` (AnythingLLM), `qwen3-8b-int4` (agent). Three identities against a 2-slot cache → constant eviction → 84s reloads.

**Fix:** Added `"qwen3-14b-int4": "qwen2.5-coder-14b-int4"` alias. Collapses 3 identities to 2. Both fit simultaneously (9 + 4.6 = 13.6 GB), so after first warm-up there are zero model-swap reloads.

**Lesson:** Model aliases are not just convenience — they're essential for cache efficiency when clients send fixed model names.

---

### Why `MAX_NEW_TOKENS_AGENT = 200`

**Problem found in logs:** The 3b agent model hit the 2048-token cap on tool-selection calls — it was rambling instead of outputting a short JSON blob. Tool selection only needs ~50–100 tokens.

**Fix:** Separate token cap for agent calls: `MAX_NEW_TOKENS_AGENT = 200`. Applied when the system prompt heuristic fires.

---

### Agent model routing: why a string heuristic, not `req.tools`

**Decision:** Agent routing uses `"picks the most optimal function" in system_prompt` as the detection signal, not `bool(req.tools)`.

**Reason:** AnythingLLM never sends the OpenAI `tools` parameter — it uses a system prompt injection approach exclusively. Using only `req.tools` would route all AnythingLLM agent calls to the 14b model, defeating the purpose entirely.

**Rejected suggestion:** A code review proposed switching to `bool(req.tools)` only. Rejected — would silently break all agent routing for AnythingLLM.

---

### Why thinking is disabled for agent/tool calls

**Problem:** On the first working agent implementation, the 14b model was being used for tool selection. The `<think>` block was consuming ~500 of the 512 available tokens, leaving ~12 tokens for the actual JSON response — producing "4 characters" of output.

**Fix:** When agent routing fires, append `/no_think` to the prompt and cap at `MAX_NEW_TOKENS_AGENT`.

---

## Concurrency

### Per-model `asyncio.Lock` for parallel inference safety

**Decision:** Each loaded model has its own `asyncio.Lock`. Acquiring it wraps the `generate()` call.

**Reason:** `LLMPipeline` is not thread-safe. Two requests for the same model calling `generate()` on the same instance concurrently produces undefined behaviour. The lock serialises them; a second request for the same model waits rather than corrupting.

**Parallel across models is fine:** Different models use different pipeline instances, so no lock contention between a 3b agent call and a 14b chat response.

---

### `active_requests` counter replaces `stats.busy` boolean

**Reason:** A boolean `busy` flag breaks under any concurrency — the first request to finish sets it `False` while the second is still running. Replaced with an integer counter: incremented on entry, decremented in `finally`.

---

## Model Discovery

### Why `generation_config.json` is the LLM/embedding discriminator

**Problem:** Original heuristic used presence of `openvino_detokenizer.xml` to distinguish LLMs from embedding models. The embedding model (`multilingual-e5-large-int8`) also had a detokenizer.

**Fix:** Use `generation_config.json` as the signal — LLMs have it, embedding models don't. Clean and reliable.

---

### Why embedding model is excluded from `/v1/models`

**Decision:** `/v1/models` lists only chat/LLM models. The embedding model is never included.

**Reason:** `/v1/models` is a chat-model catalogue consumed by clients like AnythingLLM for model selection. Embedding clients hit `/v1/embeddings` directly and don't need the model listed. Including it caused AnythingLLM to try to use it as a chat model.

**Dead end:** An explicit `+ [EMBEDDING_MODEL_ID]` append was left over from the hardcoded era. Once auto-discovery was wired in, the embedding model was already correctly excluded from `AVAILABLE_MODELS`, but still being bolted on manually in the endpoint — removed.

---

## Bugs Found and Fixed

### `model_id` NameError in `/v1/embeddings`

`model_id` was used inside the embeddings handler but not defined in that function scope. Would crash any embeddings call. Fixed to `req.model`.

### `stats.busy` stuck on streaming error

`stats.busy = False` was last in `finally`, after lines that could theoretically throw. Moved to first line of `finally` block — ensures it always resets.

### `datetime.UTC` AttributeError

A fix for the `utcnow()` deprecation warning introduced `datetime.datetime.UTC` — which doesn't exist. `UTC` is on `timezone`, not `datetime`. Fixed to `timezone.utc` (works Python 3.6+).

### `fix_mistral_regex=True` missing on LLM tokenizer

The flag was set on the embedding model tokenizer load but missing from the LLM tokenizer load, causing tokenizer warnings on `qwen2.5-3b-int4`. Added to both.

### Hardcoded `"chatcmpl-x"` in streaming finish chunk

The finish chunk had a hardcoded id instead of using the `chunk_id` generated at the start of the stream. Broke clients that correlate chunks to the opening response. `chunk_id` hoisted to shared scope.

### `get_event_loop()` deprecated in Python 3.10+

`asyncio.get_event_loop()` returns the wrong loop when called from a thread (e.g. inside `run_in_executor`). Replaced with `asyncio.get_running_loop()` in all model loader paths.

---

## Deployment & Operations

### Network accessibility — no changes needed

`ov_server` was already bound to `0.0.0.0:11435` from the start. Avahi (mDNS) runs on the machine, so `EnvyStorm.local` resolves automatically. No server config changes were needed to make it accessible from the local network.

### systemd `ExecStart` override pattern

When overriding `ExecStart` in a systemd drop-in, the first line must be a blank `ExecStart=` to clear the inherited value before setting the new one. Without the blank line, systemd appends rather than replaces.

```ini
[Service]
ExecStart=
ExecStart=/usr/bin/python3 /opt/ov_server/ov_server.py --debug
```

### SIGUSR1 debug toggle

Debug logging (POST body, rendered prompts) is toggled at runtime via `SIGUSR1` without restarting the service. State is in-memory only — resets to the startup flag value on restart.

```bash
kill -USR1 $(pgrep -f ov_server.py)
```

### Process name for journald filtering

`setproctitle` sets the process name to `ov_server`. Use `_COMM=ov_server` in journald filters, not `_COMM=python3`.

```bash
journalctl -f _COMM=ov_server
```

### Directory layout rationale

- App code: `/opt/ov_server/` — FHS convention for self-contained third-party software
- Models: `/opt/ov_server/models/` — co-located, excluded from git via `.gitignore`
- Config: `/opt/ov_server/config.json` — tracked in git, all tunables in one place, no hardcoded values in source

`models/` is on the same filesystem as `/opt/`, so `mv` from `~/ov_models/` is an inode rename — instant regardless of total model size (~16 GB).

---

## Model Selection Rationale

### Why `qwen2.5-coder-14b-int4` as default over `qwen3-14b-int4`

`qwen3-14b` is a strong general-purpose model. `qwen2.5-coder-14b` is purpose-trained on SQL, schemas, code generation, and structured data reasoning — the primary workload of this setup. Same VRAM budget (~9 GB INT4), same architecture family, direct drop-in swap.

### Why INT4 not INT8 for the 8B agent model

Quality bottleneck for tool selection is model size, not quantization precision. The jump from 3B→8B INT4 is enormous (JSON reliability, instruction following). The jump from 8B INT4→8B INT8 is modest — ~1–2% perplexity recovery on a task the model is already capable enough for. INT8 costs ~9 GB vs ~4.6 GB for INT4, consuming headroom for no meaningful gain on this task.

INT8 makes sense for the main reasoning model where precision compounds over long outputs. Not for agent tool selection.

---

## Streaming — The `AsyncTokenStreamer` Crash

### The most critical bug: `put` → `write` in openvino_genai 2026.1

**Symptom:** Every streaming request returned `200 OK` then crashed the entire server process with `SIGABRT`. The `200 OK` appeared in logs before the crash because HTTP headers were already flushed before generation started.

**Root cause:** `openvino_genai 2026.1` renamed the `StreamerBase` pure virtual method from `put(token_id: int) → bool` to `write(token: int | Sequence[int]) → StreamingStatus`. The old `put` override was silently ignored by the C++ binding — every streaming call fell through to the unimplemented `write`, hitting the pure virtual → `std::terminate` → `SIGABRT`.

**Fix:** Override `write` instead of `put`, handle both `int` and `Sequence[int]` token argument, return `StreamingStatus.RUNNING` instead of `False`.

**Lesson:** C++ pure virtual method renames in Python bindings produce no import-time error and no warning — they fail silently at call time with a process-killing SIGABRT. The `200 OK` in logs is a red herring; the real signal is the `status=6/ABRT` in the systemd journal.

---

## Repository Notes

### ov_server originally at `/home/jerzy/ov_server.py` — single file, no git

The server started as a single Python file in the home directory before being moved to `/ov_server/` and later `/opt/ov_server/`. Git was initialized at the `/ov_server/` stage. The `PYTHONPATH=/home/jerzy/.local/lib/python3.12/site-packages` environment variable in the systemd service unit is the artifact of the pre-venv era.

### `.gitignore` essentials

Excludes: `__pycache__/`, `.env`, `models/`, `.venv/`, and raw probe output files (`*_output.txt`, `vram_probe.txt`, `debug_output.txt`).

---

*Document complete.*
