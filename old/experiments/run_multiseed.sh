#!/bin/bash

# multi-seed experiment: runs each algorithm with 5 seeds
# plot script auto-detects _seed* files and averages with confidence intervals

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
GAME="bargaining_small(max_turns=4)"
ITERATIONS=500
REPORT_INTERVAL=10
NUM_SAMPLES=10
SAMPLER="external"
MC_DIST_SAMPLES=200
RESULTS_DIR="./experiments/results_multiseed"
SEEDS=(42 123 456 789 1024)

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

TOTAL_RUNS=$(( ${#EXPERIMENTS[@]} * ${#SEEDS[@]} ))
echo "========================================"
echo "multi-seed experiment"
echo "game: $GAME"
echo "iterations: $ITERATIONS, samples: $NUM_SAMPLES"
echo "mc dist samples: $MC_DIST_SAMPLES"
echo "seeds: ${SEEDS[*]}"
echo "total runs: $TOTAL_RUNS"
echo "========================================"
echo ""

TOTAL_START=$SECONDS
RUN=0
for exp in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r name algo equilibrium <<< "$exp"
  for seed in "${SEEDS[@]}"; do
    RUN=$((RUN + 1))
    outfile="$RESULTS_DIR/${name}_seed${seed}.txt"
    echo "========================================"
    echo "[$RUN/$TOTAL_RUNS] $algo on $equilibrium (seed=$seed)"
    echo "========================================"
    START_TIME=$SECONDS
    $EXECUTABLE \
      --game "$GAME" --algo "$algo" --equilibrium "$equilibrium" \
      --output_file "$outfile" --t $ITERATIONS --report_interval $REPORT_INTERVAL \
      --num_samples $NUM_SAMPLES --sampler $SAMPLER --random_seed $seed \
      --mc_dist_samples $MC_DIST_SAMPLES
    ELAPSED=$((SECONDS - START_TIME))
    TOTAL_ELAPSED=$((SECONDS - TOTAL_START))
    AVG_TIME=$((TOTAL_ELAPSED / RUN))
    EST_REMAINING=$(( AVG_TIME * (TOTAL_RUNS - RUN) ))
    LAST_VAL=$(tail -1 "$outfile" 2>/dev/null)
    echo ""
    echo "  finished in ${ELAPSED}s | final: $LAST_VAL"
    echo "  total elapsed: ${TOTAL_ELAPSED}s | est. remaining: ${EST_REMAINING}s (~$((EST_REMAINING / 60))m)"
    echo ""
  done
done

echo "========================================"
echo "all $TOTAL_RUNS runs completed!"
echo "========================================"
echo ""

echo "generating plots with confidence intervals..."
python3 plot_convergence.py "$RESULTS_DIR"
