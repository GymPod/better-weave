#!/bin/bash
# Run Better Weave server persistently (survives terminal disconnect)
# Usage: ./run.sh [start|stop|status|logs]

PID_FILE="/tmp/better_weave.pid"
LOG_FILE="/tmp/better_weave.log"
DIR="$(cd "$(dirname "$0")" && pwd)"

case "${1:-start}" in
  start)
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "Already running (PID $(cat "$PID_FILE"))"
      exit 0
    fi
    cd "$DIR"
    nohup python app.py > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    sleep 2
    if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "Started (PID $(cat "$PID_FILE"))"
      echo "URL: http://$(hostname -f):8421"
    else
      echo "Failed to start — check $LOG_FILE"
      exit 1
    fi
    ;;
  stop)
    if [ -f "$PID_FILE" ]; then
      kill "$(cat "$PID_FILE")" 2>/dev/null
      rm -f "$PID_FILE"
      echo "Stopped"
    else
      echo "Not running"
    fi
    ;;
  status)
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "Running (PID $(cat "$PID_FILE"))"
      echo "URL: http://$(hostname -f):8421"
    else
      echo "Not running"
    fi
    ;;
  logs)
    tail -f "$LOG_FILE"
    ;;
  *)
    echo "Usage: $0 {start|stop|status|logs}"
    ;;
esac
