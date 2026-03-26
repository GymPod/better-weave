#!/bin/bash
# Better Weave server daemon with auto-update from origin/main.
# Usage: ./run.sh [start|stop|status|logs|daemon-logs]

PID_FILE="/tmp/better_weave.pid"
DAEMON_PID_FILE="/tmp/better_weave_daemon.pid"
LOG_FILE="/tmp/better_weave.log"
DAEMON_LOG="/tmp/better_weave_daemon.log"
DIR="$(cd "$(dirname "$0")" && pwd)"
POLL_INTERVAL=30  # seconds between git checks

start_server() {
  cd "$DIR"
  nohup python app.py >> "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  sleep 2
  if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "[$(date)] Server started (PID $(cat "$PID_FILE"))" >> "$DAEMON_LOG"
    return 0
  else
    echo "[$(date)] Server failed to start" >> "$DAEMON_LOG"
    return 1
  fi
}

stop_server() {
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    kill "$(cat "$PID_FILE")" 2>/dev/null
    sleep 1
    kill -0 "$(cat "$PID_FILE")" 2>/dev/null && kill -9 "$(cat "$PID_FILE")" 2>/dev/null
    rm -f "$PID_FILE"
  fi
}

daemon_loop() {
  echo "[$(date)] Daemon started (PID $$), polling every ${POLL_INTERVAL}s" >> "$DAEMON_LOG"

  cd "$DIR"
  git fetch origin main --quiet 2>/dev/null
  git reset --hard origin/main --quiet 2>/dev/null
  LOCAL_SHA=$(git rev-parse HEAD)
  echo "[$(date)] Initial commit: ${LOCAL_SHA:0:8}" >> "$DAEMON_LOG"
  start_server

  while true; do
    sleep "$POLL_INTERVAL"

    cd "$DIR"
    git fetch origin main --quiet 2>/dev/null
    REMOTE_SHA=$(git rev-parse origin/main 2>/dev/null)

    if [ "$REMOTE_SHA" != "$LOCAL_SHA" ]; then
      echo "[$(date)] New commit: ${REMOTE_SHA:0:8} (was ${LOCAL_SHA:0:8}), updating..." >> "$DAEMON_LOG"
      stop_server
      git reset --hard origin/main --quiet 2>/dev/null
      LOCAL_SHA="$REMOTE_SHA"
      start_server
      echo "[$(date)] Restarted with ${LOCAL_SHA:0:8}" >> "$DAEMON_LOG"
    fi

    # Restart if server died
    if [ -f "$PID_FILE" ] && ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "[$(date)] Server died, restarting..." >> "$DAEMON_LOG"
      start_server
    fi
  done
}

case "${1:-start}" in
  start)
    if [ -f "$DAEMON_PID_FILE" ] && kill -0 "$(cat "$DAEMON_PID_FILE")" 2>/dev/null; then
      echo "Already running (daemon PID $(cat "$DAEMON_PID_FILE"), server PID $(cat "$PID_FILE" 2>/dev/null || echo '?'))"
      exit 0
    fi
    nohup bash "$0" _daemon >> "$DAEMON_LOG" 2>&1 &
    echo $! > "$DAEMON_PID_FILE"
    sleep 3
    if kill -0 "$(cat "$DAEMON_PID_FILE")" 2>/dev/null; then
      echo "Started (daemon PID $(cat "$DAEMON_PID_FILE"), server PID $(cat "$PID_FILE" 2>/dev/null || echo 'starting...'))"
      echo "URL: http://$(hostname -f):8421"
      echo "Auto-updates from origin/main every ${POLL_INTERVAL}s"
    else
      echo "Failed to start — check $DAEMON_LOG"
      exit 1
    fi
    ;;
  _daemon)
    daemon_loop
    ;;
  stop)
    stop_server
    if [ -f "$DAEMON_PID_FILE" ]; then
      kill "$(cat "$DAEMON_PID_FILE")" 2>/dev/null
      rm -f "$DAEMON_PID_FILE"
    fi
    echo "Stopped"
    ;;
  status)
    if [ -f "$DAEMON_PID_FILE" ] && kill -0 "$(cat "$DAEMON_PID_FILE")" 2>/dev/null; then
      echo "Daemon: running (PID $(cat "$DAEMON_PID_FILE"))"
    else
      echo "Daemon: not running"
    fi
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "Server: running (PID $(cat "$PID_FILE"))"
      echo "URL: http://$(hostname -f):8421"
    else
      echo "Server: not running"
    fi
    cd "$DIR" 2>/dev/null && echo "Commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
    ;;
  logs)
    tail -f "$LOG_FILE"
    ;;
  daemon-logs)
    tail -f "$DAEMON_LOG"
    ;;
  *)
    echo "Usage: $0 {start|stop|status|logs|daemon-logs}"
    ;;
esac
