#!/bin/bash
REVIEW_PORT=9000
LOG_CF=/tmp/cloudflared_9000.log
LOG_RV=/tmp/13_review.log
URL_FILE=/tmp/tunnel_url.txt
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

check_and_restart() {
    # 1. 检查审核服务
    if ! lsof -i :${REVIEW_PORT} -t >/dev/null 2>&1; then
        echo "[$(date '+%H:%M:%S')] 审核服务挂了，重启..."
        cd "$SCRIPT_DIR"
        nohup python3 13_review.py --port ${REVIEW_PORT} --host 0.0.0.0 > ${LOG_RV} 2>&1 &
        sleep 2
    fi

    # 2. 检查 cloudflared
    if ! pgrep -f "cloudflared.*${REVIEW_PORT}" >/dev/null 2>&1; then
        echo "[$(date '+%H:%M:%S')] 隧道挂了，重建..."
        nohup cloudflared tunnel --url http://localhost:${REVIEW_PORT} --no-tls-verify > ${LOG_CF} 2>&1 &
        sleep 8
        NEW_URL=$(grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' ${LOG_CF} | tail -1)
        if [ -n "$NEW_URL" ]; then
            echo "$NEW_URL" > ${URL_FILE}
            echo "[$(date '+%H:%M:%S')] 新地址: $NEW_URL"
        fi
    fi
}

check_and_restart

while true; do
    sleep 60
    check_and_restart
done
