# PROGRESS.md — Shangri-Lab

Running log of sessions, findings, and state. Most recent entry first.

---

## 2026-04-23 — Documentation pass + Claude Code audit

### What was done

Full documentation pass over both components, conducted as a collaborative session between Jerzy and Claude (chat). Source material: three Claude Code session JSONL files extracted from `~/.claude/projects/` on EnvyStorm, filtered to assistant-only turns and fed to Claude for structured extraction.

This is documented in `docs/how-to-dig-claude-code-history.md` — the process of recovering decisions and rationale from Claude Code session history is itself a reusable technique worth sharing.

**Commits this session:**
- `ov-server/DECISIONS.md` — architectural decisions, dead ends, bugs found and fixed, deployment rationale. Reconstructed entirely from Claude Code session history.
- `ov-monitor/DECISIONS.md` — fdinfo approach, xe driver workarounds, Rich UI decisions, environment notes.
- `ov-monitor/README.md` — written from scratch based on session history.
- `ov-server/requirements.txt` — generated from live user pip install, CUDA/nvidia leftovers absent (were never in user pip — the dirty requirements.txt from the RTX3080 era was a local artefact, not committed).
- `ov-monitor/pics/ov_monitor.png` — screenshot of the monitor running on EnvyStorm with both models loaded.
- `ov-server/MODELS.md` — B&B experiment note added by Jermalk.

### Claude Code audit — initiated

Claude Code briefed with full session context (`claude-code-briefing.md`) and given access to the repo. Task: review consistency, cold readability, CLAUDE.md accuracy against actual code state, config.json vs code alignment.

**Audit findings:** _[fill in after audit completes]_

**Issues found:** _[fill in]_

**Fixed in this session:** _[fill in]_

**Deferred:** _[fill in]_

---

## 2026-04 — ov_server development (Claude Code sessions)

Three Claude Code sessions covering the full ov_server build. Reconstructed summary — see `ov-server/DECISIONS.md` for full detail.

### Major milestones
- Initial server: FastAPI + LLMPipeline, streaming via AsyncTokenStreamer
- VLM integration: VLMPipeline, image routing, VRAM growth mitigations
- Tool call support: Qwen `<tool_call>` parsing, `finish_reason: "tool_calls"`, agent routing heuristic
- Concurrency: per-model asyncio.Lock, active_requests counter, LRU cache
- Model discovery: auto-scan via `generation_config.json` presence
- Config externalised to `config.json`
- Migration: `/home/jerzy/ov_server.py` → `/ov_server/` → `/opt/ov_server/`
- Models migrated to `/opt/ov_server/models/`
- Git repo initialised

### Critical bugs encountered and fixed
- `StreamerBase.put()` → `write()` rename in openvino_genai 2026.1 — silent SIGABRT on every streaming request
- Three-model-identity cache thrashing — 84s reloads on every agent call
- `vram0_mm` debugfs requires root — VRAM gauge showed 0% despite active inference
- xe driver DRI path mismatch — ov_server invisible to fdinfo scan

### B&B experiment
Converted qwen3-14b, qwen3-8b, and qwen2.5-vl-7b to OpenVINO format manually using optimum-cli. All successful. Conclusion: pre-built models from HuggingFace OpenVINO releases load faster into VRAM; inference speed is equivalent. Custom conversions not worth the effort for models that have official releases. Exception: `qwen2.5-vl-7b-int4-ov` remains a custom conversion — official `OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov` exists but no comparison done yet.

---

## The bumpy path — experiments that didn't make it

A honest record of what was tried, failed, or abandoned. Reconstructed from bash history on EnvyStorm.

### The Nvidia era (before Arc)

The machine started life with an RTX3080. Ollama was the inference stack. `nvidia-smi` appears three times in the first 100 commands. At some point the decision was made to go all-in on Intel Arc + OpenVINO. The great purge:

```bash
sudo apt-get purge '^nvidia-.*'
```

One command. No going back.

### Arc B580 bringup — not plug and play

The Battlemage (B580, `0xe211`) arrived in late 2024 and required significant work to get stable. Reconstructed from bash history:

- Tried `linux-oem-26.04`, then `linux-oem-24.04d` — kernel hunting for xe driver support
- Intel GPU repo setup failed first attempt — wrong URL (`noble main` vs `lts-24.04`), had to remove and redo
- `intel_gpu_top` tried repeatedly — never worked reliably on xe (this became the reason ov_monitor exists)
- Three different `modprobe.d/xe.conf` attempts: `force_probe=e0b0` → `force_probe=e211` → `max_vfs=0`
- Manually downloaded `bmg_guc_70.bin` firmware directly from kernel.org because the packaged version was wrong
- udev rules attempted for debugfs access (`/etc/udev/rules.d/99-xe-debugfs.rules`)

Getting OpenCL and Level Zero enumerated cleanly took multiple sessions.

### Qwen3-30B-A3B-NF4 — tested, not adopted

```python
pipe = ov_genai.LLMPipeline('/home/jerzy/ov_models/qwen3-30b-a3b-nf4', 'GPU.1', **config)
result = pipe.generate("Hello", max_new_tokens=20)
```

The Qwen3 30B MoE model in NF4 was tested directly. No further history — presumably VRAM pressure or speed made it impractical for the use case. Never made it into ov_server config.

### Gemma4 conversion — transformers version hell

Attempted to convert both `google/gemma-4-12b-it` and `google/gemma-4-E4B-it` to OpenVINO format. Hit a wall with transformers compatibility — Gemma4 required a version of transformers that hadn't been released yet on PyPI. Tried four different approaches in sequence:

```bash
optimum-cli export openvino -m google/gemma-4-12b-it ...     # failed
pip install --upgrade transformers optimum-intel openvino    # didn't help
pip install git+https://github.com/huggingface/transformers.git --force-reinstall  # too unstable
pip install "transformers>=4.57,<4.58" --force-reinstall     # nope
pip install --pre "transformers>=4.58.0.dev" --force-reinstall  # still broken
pip install "transformers==4.57.6" --force-reinstall         # back to stable
```

Eventually abandoned. Cache cleaned:
```bash
rm -rf ~/.cache/huggingface/hub/models--google--gemma-4-E4B-it/
```

Lesson: cutting-edge models require cutting-edge transformers which may break optimum-intel. Wait for the dust to settle before converting brand new architectures.

### The B&B experiment — confirmed from bash history

The full sequence is visible in history:

1. Downloaded pre-built `OpenVINO/Qwen3-8B-int4-ov` and `OpenVINO/Qwen3-14B-int4-ov` from HuggingFace
2. At some point also converted both manually using optimum-cli (the no-suffix versions)
3. Tested both — pre-built loads faster into VRAM, inference speed equivalent
4. Switched to pre-built, kept the `-ov` naming convention
5. VLM was converted manually: `optimum-cli export openvino --model Qwen/Qwen2.5-VL-7B-Instruct ... /opt/ov_server/models/qwen2.5-vl-7b-int4-ov` — the `-ov` suffix added manually to match convention

---

## Current state

### ov-server
- Running at `/opt/ov_server/` as a systemd service
- Models: `qwen3-8b-int4-ov` (agent), `qwen3-14b-int4-ov` (default), `qwen2.5-vl-7b-int4-ov` (vision), `multilingual-e5-large-int8` (embeddings)
- Both LLMs resident in VRAM simultaneously, no eviction during normal use
- AnythingLLM connected and working — agent calls, vision, embeddings all functional

### ov-monitor
- Running at `/home/jerzy/ov_monitor/`
- Displays: engine utilisation, per-process VRAM, power, temperature, fan speed
- No systemd service — run manually

### Shangri-Lab repo
- Public on GitHub: https://github.com/Jermalk/shangri-lab
- Both components fully documented
- Claude Code audit in progress

---

## Next actions

- [ ] Complete Claude Code audit — fill in findings above
- [ ] Fix anything the audit surfaces
- [ ] Write `ov-monitor/CLAUDE.md`
- [ ] Consider replacing `qwen2.5-vl-7b-int4-ov` with official pre-built and benchmark load speed
- [ ] `scraper-pipeline/README.md` — not yet written (CLAUDE.md exists)
