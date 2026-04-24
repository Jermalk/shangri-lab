# Shangri-Lab 

> *The pursuit of the perfect local LLM pipeline. The journey is the point.*

A personal lab for experimenting with local LLM pipelines, agentic workflows, and AI-assisted tooling - built by a .NET developer learning Python, running entirely on local hardware with **Intel Arc B60 24GB and OpenVINO**.

No cloud APIs. No GPU rental. No magic frameworks yet.

---

## What this is

A collection of experiments, working prototypes, and honest field notes from someone who:

- Comes from .NET, systems architecture, and automation backgrounds
- Is **not** a Python specialist code here is explicit over clever
- Runs everything locally on Intel Arc B60 24GB with OpenVINO
- Values **observability** over abstraction if I can't see what's happening, it doesn't count as working

---

## Hardware

| Machine | Specs | Role |
|---|---|---|
| HP EnvyStorm | i7-12700K, 64GB RAM, Arc B60 24GB VRAM, Linux | Primary inference server |
| NAS | Ryzen, 16GB RAM, Linux | Docker farm (planned) |

**Inference stack:** Intel OpenVINO not Ollama, not CUDA. If you're running Arc and struggling to find practical examples which works, you're in the right place.

---

## Models in use

| Model | Purpose | Throughput |
|---|---|---|
| `qwen3-8b-int4-ov` | Fast pre-selection, filtering | ~105 t/s |
| `qwen3-14b-int4-ov` | Main reasoning, summarization | ~40t/s |
| `qwen2.5-coder-14b-int4` | Code-related tasks | ~40t/s |
| `qwen2.5-vl-7b-int4-ov` | Multimodal / image input - custom conversion to ov format| not benchmarked yet |
| `multilingual-e5-large-int8` | Embeddings, RAG, deduplication | not benchmarked yet |

---

## Projects

---

### [ov-server](./ov-server)

An OpenAI-compatible REST API server backed by `openvino_genai`. Exposes `/v1/chat/completions`, `/v1/embeddings`, and `/v1/models` on port `11435` drop-in replacement for OpenAI API in local tooling (AnythingLLM, LangChain, etc.).

**What it does:**
- Serves multiple Qwen INT4 models from Intel Arc via OpenVINO
- Full streaming support via `AsyncTokenStreamer` (real token-by-token, not buffered)
- Vision/multimodal support routes image-containing requests to `VLMPipeline` automatically
- Tool call support  parses Qwen `<tool_call>` blocks, returns OpenAI-compatible `tool_calls` response
- LRU model eviction with VRAM headroom checks keeps up to 2 models loaded simultaneously
- Thinking block extraction separates `<think>...</think>` from answer, formats for display
- `/health` endpoint with live stats: throughput, token counts, loaded models, RAM

**Notable engineering decisions:**
- Direct `sysfs`/`fdinfo` VRAM queries no `nvidia-smi`, no `intel_gpu_top`
- Event loop captured at streamer construction time fixes `get_event_loop()` deprecation in threaded streaming
- Per-model `asyncio.Lock` concurrent requests on different models run in parallel; same model is serialised
- Single-file server by design keeps deployment simple

**Status:** Working in production on EnvyStorm

---

### [ov-monitor](./ov-monitor)

A rich terminal monitor for the OpenVINO server and Intel Arc GPU. Reads hardware metrics directly from kernel interfaces no root required, no `intel_gpu_top` dependency.

**What it shows:**
- Server status: busy/online, loaded models, last throughput, total request/token counts
- GPU engine utilisation by type: Render (rcs), Compute (ccs), Video (vcs), VideoEnh (vecs), Blitter (bcs) from `fdinfo` cycle counter deltas
- VRAM usage: global from `vram0_mm` + per-process breakdown via `/proc/<pid>/fdinfo`
- GPU temperature (GT + VRAM), fan RPM, instantaneous power from energy counter delta
- CPU per-core utilisation, frequency, load averages, temperatures
- System RAM and swap

**What makes it interesting:**
- Engine utilisation computed from `drm-cycles-*` / `drm-total-cycles-*` counter deltas in `/proc/<pid>/fdinfo` \u2014 works on xe driver where `intel_gpu_top -J` is broken
- VRAM per process without root reads `drm-total-vram0` from fdinfo, no debugfs clients required
- Instantaneous power derived from `energy1_input` hwmon delta between polls

**Status:** Working on xe driver (Arc B580)

---

### [scraper-pipeline](./scraper-pipeline) *(in progress)*

A two-stage local LLM pipeline for scraping and analyzing offers from the web.

**Architecture:**
```
URL list Scraper - Qwen3-8b (pre-selector) \ Qwen3-14b (summarizer) \ structured output
```

**Design goals:**
- Full observability - every stage logged, every model decision visible
- No black boxes - raw model output always captured before parsing
- Built step by step - baseline first, frameworks later if at all

**Status:**  In progress

---

## Philosophy

> Build the simplest thing that gives full visibility first.  
> Tune quality only after you can observe it.  
> Shangri-Lab is not a framework. It is a small, well-understood pipeline where every step is visible and trustworthy.

*This lab exists because most local LLM content assumes Nvidia hardware, Python expertise, and comfort with heavyweight frameworks. These experiments assume none of those things.*

If something works here, it's because it actually works - not because a framework hid the failure.

---

## What you might find useful here

- **Intel Arc + OpenVINO** practical setup and model conversion notes
- **Qwen model recipes** for INT4 quantized inference on OpenVINO including VLM and coder variants
- **OpenAI-compatible local server** that actually works with AnythingLLM, LangChain, and similar tools
- **xe driver GPU monitoring** without intel_gpu_top - sysfs/fdinfo approach for Arc cards
- **Two-stage pipeline patterns** - fast filter model + heavy reasoning model
- **Honest failure notes** - what didn't work and why

---

## Status

This is a personal lab, not a polished product. Things break. Approaches get abandoned. Notes are sometimes incomplete. That's the point (at least for now).

---

## Author

IT manager and software architect from Silesia, Poland.  
Background in .NET, Kubernetes, Kafka, automation, electronics, and even geography.  
Learning Python one pipeline at a time.

---

*Shangri-Lab - because the perfect pipeline is always just over the next mountain.*
