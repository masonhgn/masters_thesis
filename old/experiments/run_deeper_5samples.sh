#!/bin/bash

# deeper game with moderate sampling to reduce noise
# max_turns=5, num_samples=5, 100 iterations

cleanup() {
  echo ""
  echo "caught interrupt, killing all running experiments..."
  kill $(jobs -p) 2>/dev/null
  wait 2>/dev/null
  echo "all processes killed."
  exit 1
}
trap cleanup INT TERM

EXECUTABLE="./open_spiel-private/build.Release/bin/ltbr/run_corr_dist"
GAME="bargaining_small(max_turns=5)"
ITERATIONS=100
REPORT_INTERVAL=10
NUM_SAMPLES=5
SAMPLER="external"
RESULTS_DIR="./experiments/results_deeper_5samples"

mkdir -p "$RESULTS_DIR"
rm -f "$RESULTS_DIR"/*.txt

EXPERIMENTS=(
  "efcce_cfr|CFR|EFCCE"
  "efcce_efr_csps|EFR_CSPS|EFCCE"
  "efcce_efr_tips|EFR_TIPS|EFCCE"
  "efce_cfr_in|CFR_in|EFCE"
  "efce_efr_csps|EFR_CSPS|EFCE"
  "efce_efr_tips_in|EFR_TIPS_in|EFCE"
)

TOTAL=${#EXPERIMENTS[@]}

echo "========================================"
echo "deeper + 5 samples: turns=5, samples=5, iter=$ITERATIONS"
echo "========================================"

declare -a PIDS=()
for exp in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r name algo equilibrium <<< "$exp"
  outfile="$RESULTS_DIR/${name}.txt"
  echo "  starting: $algo on $equilibrium..."
  $EXECUTABLE \
    --game "$GAME" --algo "$algo" --equilibrium "$equilibrium" \
    --output_file "$outfile" --t $ITERATIONS --report_interval $REPORT_INTERVAL \
    --num_samples $NUM_SAMPLES --sampler $SAMPLER --random_seed 0 2>/dev/null &
  PIDS+=($!)
done

echo ""
echo "waiting for $TOTAL experiments..."
wait
echo "done! generating plots..."
python3 plot_convergence.py "$RESULTS_DIR"
