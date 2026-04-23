#!/home/jerzy/ov_monitor/.venv/bin/python3
"""
ov_monitor.py — Terminal monitor for OpenVino server + Intel Arc B60 + system
Usage: python ov_monitor.py [--url http://localhost:11435] [--interval 2]
Requires: pip install psutil requests rich
For Intel GPU metrics: requires intel_gpu_top (part of intel-gpu-tools)
  sudo apt install intel-gpu-tools
"""

import argparse
import os
import re
import time
import threading
from datetime import datetime

import psutil
import requests
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://localhost:11435")
parser.add_argument("--interval", type=float, default=2.0)
args = parser.parse_args()

console = Console()
MIN_WIDTH = 70

# ---------------------------------------------------------------------------
# GPU metrics — reads directly from sysfs/debugfs (xe driver, Arc B60)
# No intel_gpu_top -J needed (broken on this driver version)
#
# Sources confirmed by probe:
#   hwmon5 (xe)  — temp, power, energy
#   debugfs gt0/hw_engines — rcs0 RING_HEAD/TAIL (render busy)
#   debugfs gt1/hw_engines — vcs0/vcs2 (video busy)
#   debugfs vram0_mm       — VRAM allocator stats
# ---------------------------------------------------------------------------

DRI_B60  = "/sys/kernel/debug/dri/0000:03:00.0"
HWMON_XE = "/sys/class/hwmon/hwmon5"   # xe driver hwmon — confirmed by probe
VRAM_TOTAL_MiB = 24480                 # B60 visible_size from vram0_mm

def _read(path: str, default=None) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return default

def _read_int(path: str, default=0) -> int:
    v = _read(path)
    try:
        return int(v) if v is not None else default
    except ValueError:
        return default

# ---------------------------------------------------------------------------
# VRAM — from vram0_mm: "usage: <bytes>" line (confirmed working)
# ---------------------------------------------------------------------------
def _read_vram_mm() -> tuple[int, int]:
    """Returns (used_mib, total_mib) from vram0_mm."""
    text = _read(f"{DRI_B60}/vram0_mm", "")
    used = total = 0
    for line in text.splitlines():
        m = re.match(r'\s*usage:\s*(\d+)', line)
        if m:
            used = int(m.group(1)) // (1024 * 1024)
        m = re.match(r'\s*size:\s*(\d+)', line)
        if m:
            total = int(m.group(1)) // (1024 * 1024)
    return used, total or VRAM_TOTAL_MiB

# ---------------------------------------------------------------------------
# Per-process VRAM — reads /proc/<pid>/fdinfo for drm-total-vram0
# No sudo needed. Returns {proc_name: total_vram_mib} for xe clients only.
# ---------------------------------------------------------------------------
def _read_fdinfo_vram() -> dict[str, int]:
    """Scan all /proc PIDs for xe DRM VRAM — no debugfs clients dependency."""
    procs: dict[str, int] = {}
    try:
        for pid_str in os.listdir("/proc"):
            if not pid_str.isdigit():
                continue
            fdinfo_dir = f"/proc/{pid_str}/fdinfo"
            try:
                fds = os.listdir(fdinfo_dir)
            except OSError:
                continue
            total_vram = 0
            for fd in fds:
                content = _read(f"{fdinfo_dir}/{fd}", "")
                if "drm-driver:\txe" not in content:
                    continue
                for fline in content.splitlines():
                    m = re.match(r'drm-total-vram0:\s+(\d+)\s+KiB', fline)
                    if m:
                        total_vram = max(total_vram, int(m.group(1)) // 1024)
            if total_vram > 0:
                name = _read(f"/proc/{pid_str}/comm", pid_str).strip()
                procs[name] = procs.get(name, 0) + total_vram
    except Exception:
        pass
    return procs

# ---------------------------------------------------------------------------
# Engine utilisation — from /proc/<pid>/fdinfo cycle counters (xe)
# Computes delta between polls: used_cycles / total_cycles * 100
# Confirmed fields: drm-cycles-rcs, drm-cycles-ccs, drm-cycles-bcs, etc.
# ---------------------------------------------------------------------------
ENGINE_CYCLE_KEYS = ["rcs", "ccs", "bcs", "vcs", "vecs"]

def _read_fdinfo_cycles(pid: int) -> dict[str, int]:
    """Sum drm-cycles-<eng> and drm-total-cycles-<eng> across all xe fds."""
    result: dict[str, int] = {}
    try:
        fdinfo_dir = f"/proc/{pid}/fdinfo"
        for fd in os.listdir(fdinfo_dir):
            content = _read(f"{fdinfo_dir}/{fd}", "")
            if "drm-driver:	xe" not in content:
                continue
            for line in content.splitlines():
                m = re.match(r'(drm-(?:cycles|total-cycles)-\w+):\s*(\d+)', line)
                if m:
                    k, v = m.group(1), int(m.group(2))
                    result[k] = result.get(k, 0) + v
    except Exception:
        pass
    return result

class GpuPoller:
    def __init__(self):
        self.data: dict = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._prev_cycles: dict[str, int] = {}
        self._prev_energy = 0
        self._prev_time   = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _poll(self) -> dict:
        d = {}

        # --- temperatures (mdeg → °C) ---
        t2 = _read_int(f"{HWMON_XE}/temp2_input")
        t3 = _read_int(f"{HWMON_XE}/temp3_input")
        d["temp_gt_c"]  = round(t2 / 1000, 1) if t2 else None
        d["temp_mem_c"] = round(t3 / 1000, 1) if t3 else None

        # --- fan (RPM) ---
        fan = _read_int(f"{HWMON_XE}/fan1_input")
        d["fan_rpm"] = fan if fan else None

        # --- power cap (microwatts → W) ---
        p1 = _read_int(f"{HWMON_XE}/power1_cap")
        d["power_cap_w"] = round(p1 / 1_000_000, 1) if p1 else None
        d["_energy1_uj"] = _read_int(f"{HWMON_XE}/energy1_input")

        # --- VRAM per process (fdinfo) — must run before global VRAM ---
        d["vram_by_proc"] = _read_fdinfo_vram()

        # --- VRAM global: prefer vram0_mm; fall back to fdinfo sum ---
        used_mib, total_mib = _read_vram_mm()
        if used_mib == 0 and d["vram_by_proc"]:
            used_mib = sum(d["vram_by_proc"].values())
        d["vram_used_mib"]  = used_mib
        d["vram_total_mib"] = total_mib

        # --- engine utilisation via cycle counters ---
        # find ov_server pid by scanning /proc for python3 with xe fdinfo
        server_pid = None
        try:
            for pid_str in os.listdir("/proc"):
                if not pid_str.isdigit():
                    continue
                if _read(f"/proc/{pid_str}/comm", "").strip() != "python3":
                    continue
                fdinfo_dir = f"/proc/{pid_str}/fdinfo"
                try:
                    for fd in os.listdir(fdinfo_dir):
                        if "drm-driver:\txe" in _read(f"{fdinfo_dir}/{fd}", ""):
                            server_pid = int(pid_str)
                            break
                except OSError:
                    pass
                if server_pid:
                    break
        except Exception:
            pass
        d["_cycles"] = _read_fdinfo_cycles(server_pid) if server_pid else {}

        return d

    def _run(self):
        while not self._stop.is_set():
            try:
                d = self._poll()
                now = time.time()
                dt  = now - self._prev_time

                # instantaneous power from energy delta
                e1 = d.pop("_energy1_uj", 0)
                if self._prev_energy and dt > 0 and e1 >= self._prev_energy:
                    d["power_w"] = round((e1 - self._prev_energy) / 1_000_000 / dt, 1)
                else:
                    d["power_w"] = None
                self._prev_energy = e1

                # engine utilisation % from cycle counter delta
                cur = d.pop("_cycles", {})
                eng_pct: dict[str, float] = {}
                if self._prev_cycles and dt > 0:
                    for eng in ENGINE_CYCLE_KEYS:
                        used_key  = f"drm-cycles-{eng}"
                        total_key = f"drm-total-cycles-{eng}"
                        delta_used  = cur.get(used_key,  0) - self._prev_cycles.get(used_key,  0)
                        delta_total = cur.get(total_key, 0) - self._prev_cycles.get(total_key, 0)
                        if delta_total > 0:
                            pct = min(100.0, max(0.0, delta_used / delta_total * 100))
                            eng_pct[eng] = round(pct, 1)
                self._prev_cycles = cur
                d["engine_pct"] = eng_pct

                self._prev_time = now
                with self._lock:
                    self.data = d
            except Exception as e:
                with self._lock:
                    self.data = {"error": str(e)}
            time.sleep(1.5)

    def get(self) -> dict:
        with self._lock:
            return dict(self.data)

    def stop(self):
        self._stop.set()


def parse_gpu(raw: dict) -> dict:
    return raw


# ---------------------------------------------------------------------------
# Server health
# ---------------------------------------------------------------------------
def fetch_health(base_url: str) -> dict:
    try:
        r = requests.get(f"{base_url}/health", timeout=2)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "connection refused"}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Bar helper
# ---------------------------------------------------------------------------
def bar(pct: float, width: int = 20, color_ok="green", color_warn="yellow", color_crit="red") -> Text:
    pct = max(0.0, min(100.0, pct))
    filled = round(pct / 100 * width)
    color = color_ok if pct < 60 else (color_warn if pct < 85 else color_crit)
    b = Text()
    b.append("█" * filled, style=color)
    b.append("░" * (width - filled), style="bright_black")
    b.append(f" {pct:5.1f}%", style=color)
    return b


def val_or_na(v, fmt="{:.0f}") -> str:
    return fmt.format(v) if v is not None else "n/a"


# ---------------------------------------------------------------------------
# Build panels
# ---------------------------------------------------------------------------
def make_server_panel(health: dict) -> Panel:
    t = Table(box=None, show_header=False, padding=(0, 1))
    t.add_column(style="bright_black", no_wrap=True)
    t.add_column()

    if "error" in health:
        t.add_row("Status", Text("OFFLINE", style="bold red"))
        t.add_row("Error", health["error"])
    else:
        busy = health.get("busy", False)
        if busy:
            busy_sec = health.get("busy_for_sec", 0)
            status_txt = Text(f"BUSY  ({busy_sec:.0f}s)", style="bold yellow")
        else:
            status_txt = Text("ONLINE", style="bold green")
        t.add_row("Status",    status_txt)
        t.add_row("RAM used",  bar(health.get("ram_used_pct", 0)))
        t.add_row("RAM avail", f"{health.get('ram_available_gb', 0):.1f} GB")

        loaded = health.get("loaded_models", [])
        t.add_row("LLM loaded",  ", ".join(loaded) if loaded else Text("none", style="bright_black"))
        t.add_row("Embeddings",  Text("yes", style="green") if health.get("embedding_loaded") else Text("no", style="bright_black"))

        t.add_row("", "")  # spacer

        last_model = health.get("last_model", "")
        if last_model:
            t.add_row("Last model",  last_model)
            t.add_row("Throughput",  f"{health.get('last_tok_per_sec', 0):.1f} tok/s  ({health.get('last_tokens', 0)} tokens  {health.get('last_elapsed_sec', 0):.1f}s)")
            t.add_row("Last req at", health.get("last_request_at", ""))
        t.add_row("Total reqs",  str(health.get("total_requests", 0)))
        t.add_row("Total tokens",str(health.get("total_tokens", 0)))

    return Panel(t, title="[bold]OpenVino Server[/bold]", border_style="blue")


def make_gpu_panel(gpu: dict) -> Panel:
    t = Table(box=None, show_header=False, padding=(0, 1))
    t.add_column(style="bright_black", no_wrap=True)
    t.add_column()

    if "error" in gpu:
        t.add_row("GPU", Text(gpu["error"], style="yellow"))
    elif not gpu:
        t.add_row("GPU", Text("waiting for data...", style="bright_black"))
    else:
        # --- engine utilisation % (cycle counter deltas) ---
        eng_pct = gpu.get("engine_pct", {})
        ENGINE_LABELS = {
            "rcs":  "Render  (rcs)",
            "ccs":  "Compute (ccs)",
            "vcs":  "Video   (vcs)",
            "vecs": "VideoEnh(vecs)",
            "bcs":  "Blitter (bcs)",
        }
        for eng, label in ENGINE_LABELS.items():
            if eng in eng_pct:
                t.add_row(label, bar(eng_pct[eng]))

        t.add_row("", "")

        # --- VRAM global ---
        used  = gpu.get("vram_used_mib", 0)
        total = gpu.get("vram_total_mib", VRAM_TOTAL_MiB)
        pct   = used / total * 100 if total else 0
        t.add_row("VRAM total", bar(pct, color_ok="cyan", color_warn="yellow", color_crit="red"))
        t.add_row("VRAM used",  f"{used:,} / {total:,} MiB  ({pct:.1f}%)")

        # --- VRAM per process ---
        by_proc = gpu.get("vram_by_proc", {})
        for pname, pmib in sorted(by_proc.items(), key=lambda x: -x[1]):
            ppct = pmib / total * 100 if total else 0
            label = pname[:12]
            t.add_row(f"  {label}", f"{pmib:,} MiB  ({ppct:.1f}%)")

        t.add_row("", "")

        # --- temperatures ---
        gt_t  = gpu.get("temp_gt_c")
        mem_t = gpu.get("temp_mem_c")
        if gt_t is not None:
            clr = "green" if gt_t < 70 else ("yellow" if gt_t < 85 else "red")
            t.add_row("GT temp",   Text(f"{gt_t:.1f} °C", style=clr))
        if mem_t is not None:
            clr = "green" if mem_t < 80 else ("yellow" if mem_t < 95 else "red")
            t.add_row("VRAM temp", Text(f"{mem_t:.1f} °C", style=clr))

        # --- fan ---
        fan = gpu.get("fan_rpm")
        if fan is not None:
            clr = "green" if fan < 1500 else ("yellow" if fan < 2500 else "red")
            t.add_row("Fan", Text(f"{fan} RPM", style=clr))
        else:
            t.add_row("Fan", Text("---", style="bright_black"))

        # --- power ---
        pw     = gpu.get("power_w")
        pw_cap = gpu.get("power_cap_w")
        if pw is not None:
            cap_str = f"  / {pw_cap} W cap" if pw_cap else ""
            t.add_row("Power", f"{pw} W{cap_str}")

    return Panel(t, title="[bold]Intel Arc B60  [xe][/bold]", border_style="cyan")


def make_cpu_panel() -> Panel:
    t = Table(box=None, show_header=False, padding=(0, 1))
    t.add_column(style="bright_black", no_wrap=True)
    t.add_column()

    cpu_pct = psutil.cpu_percent(interval=None)
    per_core = psutil.cpu_percent(interval=None, percpu=True)
    freq = psutil.cpu_freq()
    load = psutil.getloadavg()
    temps = {}
    try:
        for name, entries in psutil.sensors_temperatures().items():
            for e in entries:
                if e.current and e.current > 0:
                    temps[e.label or name] = e.current
    except Exception:
        pass

    t.add_row("Overall", bar(cpu_pct))
    # Show per-core in compact rows of 4
    for i in range(0, len(per_core), 4):
        chunk = per_core[i:i+4]
        label = f"Core {i}-{i+len(chunk)-1}" if len(chunk) > 1 else f"Core {i}"
        core_bar = Text()
        for c in chunk:
            clr = "green" if c < 60 else ("yellow" if c < 85 else "red")
            core_bar.append(f"{c:4.0f}% ", style=clr)
        t.add_row(label, core_bar)

    if freq:
        t.add_row("Freq", f"{freq.current/1000:.2f} GHz  (max {freq.max/1000:.1f} GHz)")
    t.add_row("Load avg", f"{load[0]:.2f}  {load[1]:.2f}  {load[2]:.2f}  (1/5/15 min)")

    if temps:
        best = sorted(temps.items(), key=lambda x: -x[1])[:4]
        for label, temp in best:
            clr = "green" if temp < 70 else ("yellow" if temp < 90 else "red")
            t.add_row(label[:14], Text(f"{temp:.0f} °C", style=clr))

    return Panel(t, title="[bold]CPU[/bold]", border_style="magenta")


def make_mem_panel() -> Panel:
    t = Table(box=None, show_header=False, padding=(0, 1))
    t.add_column(style="bright_black", no_wrap=True)
    t.add_column()

    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()

    t.add_row("RAM used", bar(vm.percent))
    t.add_row("RAM",      f"{vm.used/1024**3:.1f} / {vm.total/1024**3:.1f} GB")
    t.add_row("Available",f"{vm.available/1024**3:.1f} GB")
    t.add_row("Swap used",bar(sw.percent, color_ok="cyan", color_warn="yellow", color_crit="red"))
    t.add_row("Swap",     f"{sw.used/1024**3:.1f} / {sw.total/1024**3:.1f} GB")

    return Panel(t, title="[bold]Memory[/bold]", border_style="green")


def make_footer(ts: str) -> Text:
    t = Text(justify="center")
    t.append(f" {args.url}  •  refreshed {ts}  •  Ctrl+C to quit ", style="bright_black")
    return t


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def build_display(health: dict, gpu_raw: dict) -> Group:
    if console.width < MIN_WIDTH:
        msg = Text(
            f"Terminal too narrow  ({console.width} cols) — resize to at least {MIN_WIDTH} columns",
            style="bold red",
            justify="center",
        )
        return Group(msg)

    gpu = parse_gpu(gpu_raw)
    ts = datetime.now().strftime("%H:%M:%S")
    gap = Text("")
    return Group(
        make_server_panel(health),
        gap,
        make_gpu_panel(gpu),
        gap,
        make_cpu_panel(),
        gap,
        make_mem_panel(),
        gap,
        make_footer(ts),
    )


def main():
    gpu_poller = GpuPoller()
    # Prime cpu_percent (first call is always 0.0)
    psutil.cpu_percent(interval=None)
    psutil.cpu_percent(interval=None, percpu=True)

    console.print(f"[bold]ov_monitor[/bold] — connecting to [cyan]{args.url}[/cyan] …")

    with Live(console=console, refresh_per_second=1, screen=True) as live:
        while True:
            health = fetch_health(args.url)
            gpu_raw = gpu_poller.get()
            live.update(build_display(health, gpu_raw))
            time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bright_black]bye.[/bright_black]")
