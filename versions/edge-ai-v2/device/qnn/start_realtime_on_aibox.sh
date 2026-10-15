#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "Usage: $0" >&2
  echo "Detector and depth are fixed to the validated single-engine HTP graph." >&2
  exit 64
fi

base="/data/local/tmp/qnn236-production"
app="$base/aibox_realtime_qnn.py"
combined="$base/yolo26s_detector_yolo26m_depth_qnn236/artifact/libyolo26s_detector_yolo26m_depth_qnn236.so"
log="$base/realtime-qnn.log"

for required in "$app" "$combined" /usr/lib/libQnnHtp.so; do
  test -e "$required" || { echo "Missing required file: $required" >&2; exit 66; }
done

if pgrep -f "^python3 $app " >/dev/null; then
  echo "Realtime service is already running"
  exit 0
fi

# The C270's UVC auto-framerate control (exposure_dynamic_framerate) can
# silently drop the delivered capture rate below 30 FPS in low light.
# Force a fixed exposure so camera_fps stays close to nominal; see
# qnn-artifacts/yolo26s_detector_yolo26m_depth_qnn236/production-manifest.json.
v4l2-ctl -d /dev/video2 --set-ctrl=exposure_dynamic_framerate=0 2>/dev/null || true

export LD_LIBRARY_PATH="/usr/lib:/usr/lib/rfsa/adsp${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
nohup python3 "$app" \
  --combined-model "$combined" \
  --confidence 0.45 \
  --host 0.0.0.0 \
  --port 8080 \
  >"$log" 2>&1 &
pid=$!

for _ in $(seq 1 90); do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "Realtime service exited during startup; log follows:" >&2
    tail -n 80 "$log" >&2
    exit 1
  fi
  if ss -ltnp 2>/dev/null | grep -Fq ':8080 '; then
    # jpegdec (software MJPEG decode; QCS8550 has no hardware MJPEG decode
    # element) is CPU-bound and was the actual bottleneck below ~17 FPS, not
    # HTP inference. Pin every thread of the service to the fastest cores
    # (3,4 at 2.8GHz, 5 at 3.19GHz) and prioritize the GStreamer capture
    # threads. See cpu_affinity_fix in the production manifest for the
    # measured before/after. This is not persisted anywhere else, so it is
    # re-applied here on every start.
    #
    # The GStreamer pipeline threads (queue*:src, v4l2src0:src) are spawned
    # a little after the HTTP port starts listening, so poll for them for
    # up to 15s rather than pinning only once right here.
    found_gst_threads=0
    for _ in $(seq 1 15); do
      for task in "/proc/$pid/task"/*; do
        tid="$(basename "$task")"
        taskset -pc 3,4,5 "$tid" >/dev/null 2>&1 || true
        comm="$(cat "$task/comm" 2>/dev/null || true)"
        case "$comm" in
          queue*|v4l2src*)
            renice -n -12 -p "$tid" >/dev/null 2>&1 || true
            found_gst_threads=1
            ;;
        esac
      done
      if [[ "$found_gst_threads" -eq 1 ]]; then
        break
      fi
      sleep 1
    done
    echo "STARTED pid=$pid url=http://127.0.0.1:8080/ log=$log (cpu_pinning_applied=$found_gst_threads)"
    exit 0
  fi
  sleep 1
done

echo "Timed out waiting for port 8080; process remains pid=$pid" >&2
exit 1
