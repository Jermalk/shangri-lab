#!/bin/bash
# Run with: sudo bash probe_b60.sh
# Probes all known xe-driver debugfs/sysfs paths for Arc B60
# Paste the output so we can build the correct monitor

PCI="0000:03:00.0"
DRI_BASE="/sys/kernel/debug/dri/$PCI"
SYSFS_BASE="/sys/class/drm"

echo "=== kernel / driver ==="
uname -r
lspci -nn | grep -i "vga\|display\|3d\|intel arc\|$PCI" 2>/dev/null
echo ""

echo "=== xe driver loaded? ==="
lsmod | grep -E "^xe |^i915 " || echo "neither xe nor i915 found in lsmod"
echo ""

echo "=== DRI debugfs root ==="
ls /sys/kernel/debug/dri/ 2>/dev/null || echo "no debugfs/dri (need sudo or debugfs not mounted)"
echo ""

echo "=== gt dirs under $PCI ==="
ls "$DRI_BASE/" 2>/dev/null || echo "path not found: $DRI_BASE"
echo ""

for gt in gt0 gt1; do
    echo "--- $gt/hw_engines ---"
    cat "$DRI_BASE/$gt/hw_engines" 2>/dev/null | head -40 || echo "not found"
    echo ""

    echo "--- $gt/freq0/cur_freq (MHz) ---"
    cat "$DRI_BASE/$gt/freq0/cur_freq" 2>/dev/null || echo "not found"

    echo "--- $gt/freq0/min_freq ---"
    cat "$DRI_BASE/$gt/freq0/min_freq" 2>/dev/null || echo "not found"

    echo "--- $gt/freq0/max_freq ---"
    cat "$DRI_BASE/$gt/freq0/max_freq" 2>/dev/null || echo "not found"
    echo ""

    echo "--- $gt/guc_info (active jobs) ---"
    cat "$DRI_BASE/$gt/uc/guc_info" 2>/dev/null | grep -E "finished|active|pending" | head -20 || echo "not found"
    echo ""

    echo "--- $gt/engines (sysfs) ---"
    ls "$DRI_BASE/$gt/engines/" 2>/dev/null || echo "not found"
    echo ""
done

echo "=== VRAM via sysfs clients ==="
for f in /sys/kernel/debug/dri/$PCI/clients \
          /sys/kernel/debug/dri/*/clients; do
    echo "--- $f ---"
    cat "$f" 2>/dev/null | head -20 || echo "not found"
done
echo ""

echo "=== memory info ==="
for f in "$DRI_BASE/vram_mm" \
          "$DRI_BASE/gem_names" \
          "$DRI_BASE/i915_gem_objects" \
          "$DRI_BASE/xe_gem_objects" \
          "$DRI_BASE/clients"; do
    echo "--- $f ---"
    cat "$f" 2>/dev/null | head -10 || echo "not found"
done
echo ""

echo "=== intel_gpu_top available? ==="
which intel_gpu_top 2>/dev/null || echo "not in PATH"
intel_gpu_top --help 2>&1 | head -5 || true
echo ""

echo "=== intel_gpu_top JSON sample (3 sec) ==="
timeout 3 intel_gpu_top -J -d 1000 2>/dev/null || echo "failed or not available"
echo ""

echo "=== sysfs hwmon (temp/power) ==="
for hwmon in /sys/class/hwmon/hwmon*; do
    name=$(cat "$hwmon/name" 2>/dev/null)
    echo "--- $hwmon ($name) ---"
    cat "$hwmon/temp1_input" 2>/dev/null && echo " (temp1, mdeg)"
    cat "$hwmon/power1_input" 2>/dev/null && echo " (power1, microwatts)"
    ls "$hwmon/"  2>/dev/null | grep -E "temp|power|freq|energy"
done
echo ""

echo "Done."
