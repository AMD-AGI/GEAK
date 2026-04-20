#!/bin/bash
# Enforces exclusive GPU access for geak_slot1 & geak_slot2.
# Requires passwordless sudo (sapmajum has it).
LOG="/data/sapmajum/monitor.log"

while true; do
    SLOT1_CID=$(docker inspect geak_slot1 --format '{{.Id}}' 2>/dev/null | head -c 12)
    SLOT2_CID=$(docker inspect geak_slot2 --format '{{.Id}}' 2>/dev/null | head -c 12)

    # 1) Stop non-whitelisted containers
    for c in $(docker ps --format "{{.Names}}" 2>/dev/null | grep -v -E "^geak_slot[12]$|^node-exporter"); do
        docker update --restart=no "$c" 2>/dev/null
        docker stop "$c" 2>/dev/null
        echo "[$(date +%H:%M:%S)] stopped container $c" >> "$LOG"
    done

    # 2) Kill rogue GPU processes (and their respawners)
    for pid in $(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/ {print $1}'); do
        [ -z "$pid" ] && continue
        [ ! -e "/proc/$pid" ] && continue
        cg=$(cat /proc/$pid/cgroup 2>/dev/null | head -1)
        if echo "$cg" | grep -qE "docker[-/]${SLOT1_CID:-zzzzzzzzzz}|docker[-/]${SLOT2_CID:-zzzzzzzzzz}"; then
            continue
        fi
        user=$(ps -p $pid -o user= 2>/dev/null)
        cmd=$(ps -p $pid -o comm= 2>/dev/null)
        args=$(ps -p $pid -o args= 2>/dev/null | head -c 100)
        # Kill the rogue process (use sudo because it may be other user)
        sudo -n kill -9 "$pid" 2>/dev/null
        echo "[$(date +%H:%M:%S)] killed rogue GPU pid=$pid user=$user cmd=$cmd" >> "$LOG"

        # Walk parent chain to kill respawners (tmux, bash scripts, shell wrappers)
        cur=$pid
        for _ in 1 2 3 4 5; do
            ppid=$(awk '/^PPid:/ {print $2}' /proc/$cur/status 2>/dev/null)
            [ -z "$ppid" ] || [ "$ppid" = "1" ] || [ "$ppid" = "0" ] && break
            pcomm=$(ps -p $ppid -o comm= 2>/dev/null)
            # Stop at login shell / systemd boundary
            [[ "$pcomm" =~ ^(sshd|systemd|init|login)$ ]] && break
            # Kill bash scripts, tmux: server, python wrappers
            if [[ "$pcomm" =~ ^(bash|tmux|tmux:|python|python3|run_lmeval.*|sh)$ ]]; then
                sudo -n kill -9 "$ppid" 2>/dev/null
                echo "[$(date +%H:%M:%S)]   + killed respawner pid=$ppid comm=$pcomm" >> "$LOG"
            fi
            cur=$ppid
        done
    done

    sleep 15
done
