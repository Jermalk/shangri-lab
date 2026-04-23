# DECISIONS.md — ov_monitor

> Reconstructed from Claude Code session history.
> Authored collaboratively by Jerzy and Claude (Code + chat).
> This document captures *why* things are the way they are — decisions, dead ends, and non-obvious choices that don't appear in the code itself.

---

## ov_monitor — GPU Visibility and fdinfo

### Why `intel_gpu_top -J` is not used

`intel_gpu_top -J` is broken on the xe driver on this kernel. This is the entire reason `ov_monitor.py` exists — the standard tool doesn't work.

### Why `/proc` scan instead of debugfs clients list

**Problem:** The xe driver registers different processes under different DRI paths. `ov_server` (python3) was registered under `dri/0000:00:02.0`, while the monitor was reading the clients list exclusively from `DRI_B60` (`0000:03:00.0`). Result: ov_server was invisible to both VRAM and engine utilisation polling — `vram_by_proc` returned empty every poll, server_pid was always `None`.

**Fix:** Both `_read_fdinfo_vram` and the server PID lookup in `_poll` now scan `/proc` directly for processes with `drm-driver: xe` fdinfo entries. No DRI path assumptions. Live test confirmed python3 detected at 9589 MiB after fix.

**Dead end:** Debugfs clients list (`DRI_B60/clients`) approach — looked correct in probe output, failed in practice because the inference process happened to register under a different DRI path.

### Why VRAM gauge showed 0% despite active inference

**Problem:** `vram0_mm` is a debugfs file requiring root. Always returned `None`, parsed as 0 used / 0 total. Gauge showed `0/24480 MiB = 0%` even with 10+ GB actively allocated.

**Fix:** Per-process fdinfo scan runs first. `used_mib` falls back to the sum of all xe VRAM consumers from fdinfo when `vram0_mm` is unreadable. Total stays at hardcoded 24480 MiB (GPU spec). Confirmed 10403 MiB shown correctly after fix.

### hwmon path for power readings

`hwmon5` is the xe driver hwmon on this specific machine — confirmed by probe. Power derived from `energy1_input` delta between polls. Not a stable path across machines or reboots; the probe script exists to confirm it.

---

## ov_monitor — Display / Rich UI

### Layout → Group for natural panel height

**Problem:** `rich.Layout` with `ratio` fills stretches panels to fill the terminal vertically — empty padding appears between rows.

**Fix:** Replaced with `rich.console.Group` — panels render at natural content height, stacked with a single blank line between them. `Layout` import removed.

### Minimum terminal width guard

`MIN_WIDTH = 70` constant checked on every refresh cycle (reacts to live window resize). Below threshold: all panels replaced with a centred error message showing current and required width. Prevents garbled display on narrow terminals.

### Fan row always rendered

Fan row was conditionally added only when `fan_rpm is not None` — caused the row to vanish when sysfs read fails, shifting all subsequent rows and breaking alignment. Fixed to always render, substituting `---` in dim grey when value is absent.

---

## Environment & Dependency Management

### OpenVINO installed in user pip, not venv — known fragility

OpenVINO packages live in `~/.local/lib/python3.12/site-packages` (user pip). The `ov_monitor/.venv` originally had `include-system-site-packages = false` and zero OpenVINO packages — scripts run inside the venv couldn't import OpenVINO. Scripts had to be run with system `python3`, which worked but was fragile.

**Fix:** OpenVINO 2026.1.0 + genai + tokenizers + optimum-intel installed into `.venv` explicitly. All three devices enumerate inside the venv. `pip freeze` → `requirements.txt` (59 packages). Shebangs updated to venv Python.

### Wrong OpenVINO import in CLAUDE.md

`from openvino.runtime import Core` — the `openvino.runtime` namespace was deprecated in 2023.x and removed by 2026.x. Would raise `ModuleNotFoundError` on the installed version. Correct import:

```python
import openvino as ov
core = ov.Core()
assert "CPU" in core.available_devices
```

### Arc B580 (Battlemage) driver requirements

Device ID `0xe211` = Arc B580, released Nov 2024. Requires kernel ≥ 6.11 and compute runtime ≥ 25.x. Running kernel 6.17 and OpenCL ICD 25.18 — satisfied. Keeping drivers current is non-optional for this GPU; it is not yet in any LTS kernel's stable support window.

### GPU device enumeration: GPU.0 vs GPU.1

- `GPU.0` = Intel UHD Graphics 770 (iGPU, Alder Lake GT1)
- `GPU.1` = Intel Arc B580 (dGPU, Battlemage)

All inference code must target `GPU.1` explicitly. The mandatory device check in CLAUDE.md guards against accidentally running on CPU or iGPU.

---

## Repository Notes

### `.gitignore` essentials

Excludes: `__pycache__/`, `.env`, `.venv/`, and raw probe output files (`*_output.txt`, `vram_probe.txt`, `debug_output.txt`).

---

*Document complete.*
