#!/bin/bash

# depth comparison: run same clean config at max_turns=3,4,5
# shows how deeper game trees amplify tips > csps separation
# uses external sampler + high num_samples for clean measurements

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
ITERATIONS=1000
REPORT_INTERVAL=10
SAMPLER="external"
SEED=42

EXPERIMENTS=(
  "efcce_cfr|CFR|EFCCE"
  "efcce_efr_csps|EFR_CSPS|EFCCE"
  "efcce_efr_tips|EFR_TIPS|EFCCE"
  "efce_cfr_in|CFR_in|EFCE"
  "efce_efr_csps|EFR_CSPS|EFCE"
  "efce_efr_tips_in|EFR_TIPS_in|EFCE"
)

# depth 3 and 4 use 200 samples; depth 5 uses 100 (larger tree = slower)
declare -A DEPTH_SAMPLES
DEPTH_SAMPLES[3]=100
DEPTH_SAMPLES[4]=100
DEPTH_SAMPLES[5]=50

for DEPTH in 3 4 5; do
  NUM_SAMPLES=${DEPTH_SAMPLES[$DEPTH]}
  RESULTS_DIR="./experiments/results_depth_${DEPTH}"
  GAME="bargaining_small(max_turns=${DEPTH})"

  mkdir -p "$RESULTS_DIR"
  rm -f "$RESULTS_DIR"/*.txt

  echo "========================================"
  echo "depth ${DEPTH}: max_turns=${DEPTH}, ${NUM_SAMPLES} samples, ${ITERATIONS} iter"
  echo "========================================"

  declare -a PIDS=()
  declare -a NAMES=()

  for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r name algo equilibrium <<< "$exp"
    outfile="$RESULTS_DIR/${name}.txt"
    echo "  starting: $algo on $equilibrium..."
    $EXECUTABLE \
      --game "$GAME" --algo "$algo" --equilibrium "$equilibrium" \
      --output_file "$outfile" --t $ITERATIONS --report_interval $REPORT_INTERVAL \
      --num_samples $NUM_SAMPLES --sampler $SAMPLER --random_seed $SEED 2>/dev/null &
    PIDS+=($!)
    NAMES+=("$name")
  done

  echo ""
  echo "  waiting for depth ${DEPTH} experiments..."
  wait

  echo "  depth ${DEPTH} complete! final distances:"
  for name in "${NAMES[@]}"; do
    file="$RESULTS_DIR/${name}.txt"
    header=$(head -1 "$file" 2>/dev/null)
    last=$(tail -1 "$file" 2>/dev/null)
    printf "    %-25s %s\n" "$name" "$last"
  done
  echo ""

  echo "  generating plots for depth ${DEPTH}..."
  python3 plot_convergence.py "$RESULTS_DIR"
  echo ""

  unset PIDS
  unset NAMES
done

echo "========================================"
echo "all depth experiments completed!"
echo "========================================"
echo ""
echo "compare results across depths:"
for DEPTH in 3 4 5; do
  echo ""
  echo "--- max_turns=${DEPTH} ---"
  for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r name algo equilibrium <<< "$exp"
    file="./experiments/results_depth_${DEPTH}/${name}.txt"
    last=$(tail -1 "$file" 2>/dev/null)
    printf "  %-25s %s\n" "$name" "$last"
  done
done
