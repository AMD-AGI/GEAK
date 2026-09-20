#!/bin/bash
# How many PMC counters can gfx1151 collect in ONE rocprofv3 pass?
# Bisect: n=1 already known OK (agent collected 5 counters one-at-a-time);
# the combined set already known to FAIL. So probe the middle only.
cd /scratch/jiangyon/geak-rdna
ALL="SQ_WAVES GRBM_GUI_ACTIVE FETCH_SIZE WRITE_SIZE"
for n in 2 4; do
  C=$(echo $ALL | cut -d" " -f1-$n)
  rm -rf /tmp/pmcout; mkdir -p /tmp/pmcout
  echo "pmc: $C" > /tmp/pmc_in.txt
  out=$(timeout 240 rocprofv3 -i /tmp/pmc_in.txt -d /tmp/pmcout -- python3 pmc_work.py 2>&1)
  if echo "$out" | grep -qiE "exceeds the capabilities|error code 38|aborted"; then
    echo "  n=$n  FAIL   [$C]"
  elif echo "$out" | grep -q "^ok"; then
    echo "  n=$n  OK     [$C]"
  else
    echo "  n=$n  ????   $(echo "$out" | tail -2 | tr "\n" " " | cut -c1-90)"
  fi
done
