#!/usr/bin/env bash
# Sample junction temp / memory temp / sclk once a second while a benchmark runs.
# rocm-smi --setperfdeterminism silently no-ops in this container, so the clock is
# free-running and has to be observed rather than pinned.
while true; do
  printf '%s ' "$(date +%H:%M:%S)"
  rocm-smi --showtemp --showclocks 2>/dev/null \
    | grep -Ei "Sensor junction|Sensor memory|sclk clock level" \
    | sed -E 's/.*: //' | tr '\n' ' '
  echo
  sleep 2
done
