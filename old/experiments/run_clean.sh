#!/bin/bash

# clean experiment with mc distance estimation
# mc_dist_samples=30 makes distance computation ~10x faster
# runs sequentially to avoid cpu contention

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
GAME="bargaining_small(max_turns=3)"
ITERATIONS=1000
REPORT_INTERVAL=20
NUM_SAMPLES=100
SAMPLER="external"
MC_DIST_SAMPLES=200
RESULTS_DIR="./experiments/results_clean"

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
echo "clean experiment (mc dist, sequential)"
echo "game: $GAME"
echo "iterations: $ITERATIONS, samples: $NUM_SAMPLES"
echo "mc dist samples: $MC_DIST_SAMPLES (0=exact)"
echo "total experiments: $TOTAL"
echo "========================================"
echo ""

TOTAL_START=$SECONDS
IDX=0
for exp in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r name algo equilibrium <<< "$exp"
  outfile="$RESULTS_DIR/${name}.txt"
  IDX=$((IDX + 1))
  REMAINING=$((TOTAL - IDX + 1))
  echo "========================================"
  echo "[$IDX/$TOTAL] $algo on $equilibrium"
  echo "  experiment: $name"
  echo "  remaining after this: $((REMAINING - 1)) experiments"
  echo "========================================"
  START_TIME=$SECONDS
  $EXECUTABLE \
    --game "$GAME" --algo "$algo" --equilibrium "$equilibrium" \
    --output_file "$outfile" --t $ITERATIONS --report_interval $REPORT_INTERVAL \
    --num_samples $NUM_SAMPLES --sampler $SAMPLER --random_seed 42 \
    --mc_dist_samples $MC_DIST_SAMPLES
  ELAPSED=$((SECONDS - START_TIME))
  TOTAL_ELAPSED=$((SECONDS - TOTAL_START))
  AVG_TIME=$((TOTAL_ELAPSED / IDX))
  EST_REMAINING=$(( AVG_TIME * (TOTAL - IDX) ))
  LAST_VAL=$(tail -1 "$outfile" 2>/dev/null)
  echo ""
  echo "  finished in ${ELAPSED}s | final: $LAST_VAL"
  echo "  total elapsed: ${TOTAL_ELAPSED}s | est. remaining: ${EST_REMAINING}s (~$((EST_REMAINING / 60))m)"
  echo ""
done

echo "========================================"
echo "all experiments completed!"
echo "========================================"
echo ""

echo "final equilibrium distances (lower = better):"
echo ""
for exp in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r name algo equilibrium <<< "$exp"
  file="$RESULTS_DIR/${name}.txt"
  last=$(tail -1 "$file" 2>/dev/null)
  printf "  %-25s %s\n" "$name" "$last"
done
echo ""

echo "generating plots..."
python3 plot_convergence.py "$RESULTS_DIR"
