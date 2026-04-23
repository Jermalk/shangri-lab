#!/bin/bash
# Run this WHILE ov_server.py is active (ideally mid-inference)
# sudo bash probe_vram.sh 2>&1 | tee vram_probe.txt

PCI="0000:03:00.0"
DRI="/sys/kernel/debug/dri/$PCI"
PYTHON_PID=$(pgrep -f ov_server.py | head -1)

echo "=== ov_server.py pid: ${PYTHON_PID:-NOT FOUND} ==="
echo ""

echo "=== vram0_mm (full) ==="
cat "$DRI/vram0_mm" 2>/dev/null || echo "empty/not found"
echo ""

echo "=== gtt_mm (full) ==="
cat "$DRI/gtt_mm" 2>/dev/null || echo "empty/not found"
echo ""

echo "=== stolen_mm ==="
cat "$DRI/stolen_mm" 2>/dev/null || echo "empty/not found"
echo ""

echo "=== xe_gem_objects (full) ==="
cat "$DRI/xe_gem_objects" 2>/dev/null || echo "empty/not found"
echo ""

echo "=== gem_names (full) ==="
cat "$DRI/gem_names" 2>/dev/null || echo "empty/not found"
echo ""

echo "=== info ==="
cat "$DRI/info" 2>/dev/null || echo "not found"
echo ""

echo "=== xe_params ==="
cat "$DRI/xe_params" 2>/dev/null | head -30 || echo "not found"
echo ""

echo "=== sriov_info ==="
cat "$DRI/sriov_info" 2>/dev/null || echo "not found"
echo ""

echo "=== internal_clients ==="
cat "$DRI/internal_clients" 2>/dev/null || echo "not found"
echo ""

echo "=== ls gt0/ ==="
ls "$DRI/gt0/" 2>/dev/null || echo "not found"
echo ""

echo "=== ls gt1/ ==="
ls "$DRI/gt1/" 2>/dev/null || echo "not found"
echo ""

echo "=== gt0 subdirs (recursive ls 2 levels) ==="
find "$DRI/gt0" -maxdepth 2 2>/dev/null | head -60
echo ""

echo "=== gt1 subdirs ==="
find "$DRI/gt1" -maxdepth 2 2>/dev/null | head -60
echo ""

echo "=== hwmon5 full listing ==="
for f in /sys/class/hwmon/hwmon5/*; do
    val=$(cat "$f" 2>/dev/null)
    [ -n "$val" ] && echo "$f = $val"
done
echo ""

echo "=== /proc/$PYTHON_PID/status (VmRSS etc) ==="
if [ -n "$PYTHON_PID" ]; then
    grep -E "VmRSS|VmPeak|VmSize|VmSwap|Threads" /proc/$PYTHON_PID/status 2>/dev/null
fi
echo ""

echo "=== fdinfo VRAM for python3 pid $PYTHON_PID ==="
if [ -n "$PYTHON_PID" ]; then
    for fd in /proc/$PYTHON_PID/fdinfo/*; do
        content=$(cat "$fd" 2>/dev/null)
        if echo "$content" | grep -qi "vram\|drm-memory\|drm-engine"; then
            echo "--- $fd ---"
            echo "$content"
            echo ""
        fi
    done
fi
echo ""

echo "=== fdinfo scan all drm clients for VRAM ==="
for pid in $(cat "$DRI/clients" 2>/dev/null | awk 'NR>1 {print $2}' | sort -u); do
    name=$(cat /proc/$pid/comm 2>/dev/null || echo "?")
    found=0
    for fd in /proc/$pid/fdinfo/*; do
        content=$(cat "$fd" 2>/dev/null)
        if echo "$content" | grep -qi "vram\|drm-memory"; then
            if [ $found -eq 0 ]; then
                echo "--- pid $pid ($name) ---"
                found=1
            fi
            echo "$content" | grep -i "vram\|drm-memory\|drm-engine"
        fi
    done
done
echo ""

echo "Done."
