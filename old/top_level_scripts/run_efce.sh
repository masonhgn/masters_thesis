#!/bin/bash

# experiment runner - EFCCE and EFCE, single run per algorithm
# usage: bash run_efce.sh

# kill all child processes on ctrl+c or exit
cleanup() {
  echo ""
  echo "caught interrupt, killing all running experiments..."
  kill $(jobs -p) 2>/dev/null
  wait 2>/dev/null
  echo "all processes killed."
  exit 1
}
trap cleanup INT TERM

# configuration
EXECUTABLE="./open_spiel-private/build.Release/bin/ltbr/run_corr_dist"
ITERATIONS=100
REPORT_INTERVAL=1
NUM_SAMPLES=1
SAMPLER="external"
RESULTS_DIR="./dealornodeal/cpp/experiment_results"

mkdir -p "$RESULTS_DIR"
rm -f "$RESULTS_DIR"/*.txt

# define experiments: name, algo, equilibrium
EXPERIMENTS=(
  "efcce_cfr|CFR|EFCCE"
  "efcce_efr_csps|EFR_CSPS|EFCCE"
  "efcce_efr_tips|EFR_TIPS|EFCCE"
  "efce_cfr|CFR|EFCE"
  "efce_cfr_in|CFR_in|EFCE"
  "efce_efr_tips_in|EFR_TIPS_in|EFCE"
  "efce_efr_csps|EFR_CSPS|EFCE"
)

TOTAL=${#EXPERIMENTS[@]}

echo "========================================"
echo "EFCCE + EFCE experiment runner"
echo "========================================"
echo ""
echo "configuration:"
echo "  iterations: $ITERATIONS"
echo "  report interval: $REPORT_INTERVAL"
echo "  num samples: $NUM_SAMPLES"
echo "  sampler: $SAMPLER"
echo "  total experiments: $TOTAL"
echo "  results dir: $RESULTS_DIR"
echo ""

# launch all experiments in parallel
declare -a PIDS=()
declare -a NAMES=()

for exp in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r name algo equilibrium <<< "$exp"
  outfile="$RESULTS_DIR/${name}.txt"
  echo "starting: $algo on $equilibrium..."
  $EXECUTABLE \
    --game bargaining_small \
    --algo "$algo" \
    --equilibrium "$equilibrium" \
    --output_file "$outfile" \
    --t $ITERATIONS \
    --report_interval $REPORT_INTERVAL \
    --num_samples $NUM_SAMPLES \
    --sampler $SAMPLER \
    --random_seed 0 2>/dev/null &
  PIDS+=($!)
  NAMES+=("$name")
done

echo ""
echo "all $TOTAL experiments launched, monitoring..."
echo ""

# monitor until done
while true; do
  COMPLETED=0
  for pid in "${PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      COMPLETED=$((COMPLETED + 1))
    fi
  done

  if [ $COMPLETED -eq $TOTAL ]; then
    break
  fi

  for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r name algo equilibrium <<< "$exp"
    file="$RESULTS_DIR/${name}.txt"
    if [ -f "$file" ]; then
      last_iter=$(tail -1 "$file" 2>/dev/null | awk '{print $1}')
      if [ -n "$last_iter" ] && [ "$last_iter" -eq "$last_iter" ] 2>/dev/null; then
        pct=$((last_iter * 100 / ITERATIONS))
        filled=$((pct / 5))
        bar=""
        for ((j=0; j<20; j++)); do
          if [ $j -lt $filled ]; then bar+="=";
          elif [ $j -eq $filled ]; then bar+=">";
          else bar+=" "; fi
        done
        printf "  %-25s [%s] %4d/%d (%3d%%)\n" "$name" "$bar" "$last_iter" "$ITERATIONS" "$pct"
      else
        printf "  %-25s starting...\n" "$name"
      fi
    else
      printf "  %-25s waiting...\n" "$name"
    fi
  done

  echo ""
  sleep 5
done

wait

echo "========================================"
echo "all experiments completed!"
echo "========================================"
echo ""
echo "results saved to: $RESULTS_DIR/"
echo ""
