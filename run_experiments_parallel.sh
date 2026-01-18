#!/bin/bash

# parallel experiment runner - runs experiments in batches
# designed for multi-core machines (e.g., DigitalOcean c-32)
# usage: bash run_experiments_parallel_improved.sh [batch_number]
#   batch_number: 1 (experiments 1-7) or 2 (experiments 8-14) or "all" (default)

# configuration
EXECUTABLE="./open_spiel-private/build.Release/bin/ltbr/run_corr_dist"
ITERATIONS=10
REPORT_INTERVAL=20
NUM_SAMPLES=100  # reduced for memory efficiency
SAMPLER="external"  # use external sampling for speed
RANDOM_SEED=42  # seed for reproducibility
RESULTS_DIR="./experiment_results"

# get batch number from command line (default: all)
BATCH=${1:-all}

# create results directory
mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "parallel experiment runner - batch $BATCH"
echo "========================================"
echo ""
echo "configuration:"
echo "  iterations: $ITERATIONS"
echo "  report interval: $REPORT_INTERVAL"
echo "  num samples: $NUM_SAMPLES"
echo "  sampler: $SAMPLER"
echo "  results dir: $RESULTS_DIR"
echo "  cores available: $(nproc)"
echo ""

# track PIDs for monitoring
declare -a PIDS
declare -a NAMES

# ==============================================================================
# afcce experiments (2)
# ==============================================================================

if [ "$BATCH" = "1" ] || [ "$BATCH" = "all" ]; then
echo "[1/14] starting: cfr on afcce..."
$EXECUTABLE \
  --game bargaining_small\
  --algo CFR \
  --equilibrium AFCCE \
  --output_file "$RESULTS_DIR/afcce_cfr.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED 2>/dev/null &
PIDS+=($!)
NAMES+=("afcce_cfr")

echo "[2/14] starting: efr(act) on afcce..."
$EXECUTABLE \
  --game bargaining_small\
  --algo EFR_ACT \
  --equilibrium AFCCE \
  --output_file "$RESULTS_DIR/afcce_efr_act.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED 2>/dev/null &
PIDS+=($!)
NAMES+=("afcce_efr_act")

# ==============================================================================
# afce experiments (2)
# ==============================================================================

echo "[3/14] starting: cfr on afce..."
$EXECUTABLE \
  --game bargaining_small\
  --algo CFR \
  --equilibrium AFCE \
  --output_file "$RESULTS_DIR/afce_cfr.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED 2>/dev/null &
PIDS+=($!)
NAMES+=("afce_cfr")

echo "[4/14] starting: efr(act_in) on afce..."
$EXECUTABLE \
  --game bargaining_small\
  --algo EFR_ACT_in \
  --equilibrium AFCE \
  --output_file "$RESULTS_DIR/afce_efr_act_in.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED 2>/dev/null &
PIDS+=($!)
NAMES+=("afce_efr_act_in")

# ==============================================================================
# efcce experiments - external sampling (4)
# ==============================================================================

echo "[5/14] starting: cfr on efcce..."
$EXECUTABLE \
  --game bargaining_small\
  --algo CFR \
  --equilibrium EFCCE \
  --output_file "$RESULTS_DIR/efcce_cfr.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED 2>/dev/null &
PIDS+=($!)
NAMES+=("efcce_cfr")

echo "[6/14] starting: efr(csps) on efcce..."
$EXECUTABLE \
  --game bargaining_small\
  --algo EFR_CSPS \
  --equilibrium EFCCE \
  --output_file "$RESULTS_DIR/efcce_efr_csps.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED 2>/dev/null &
PIDS+=($!)
NAMES+=("efcce_efr_csps")

echo "[7/14] starting: efr(tips) on efcce..."
$EXECUTABLE \
  --game bargaining_small\
  --algo EFR_TIPS \
  --equilibrium EFCCE \
  --output_file "$RESULTS_DIR/efcce_efr_tips.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED 2>/dev/null &
PIDS+=($!)
NAMES+=("efcce_efr_tips")
fi

# ==============================================================================
# efcce/efce experiments - batch 2 (8-14)
# ==============================================================================

if [ "$BATCH" = "2" ] || [ "$BATCH" = "all" ]; then
echo "[8/14] starting: efr(bhv) on efcce..."
$EXECUTABLE \
  --game bargaining_small\
  --algo EFR_BHV \
  --equilibrium EFCCE \
  --output_file "$RESULTS_DIR/efcce_efr_bhv.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED 2>/dev/null &
PIDS+=($!)
NAMES+=("efcce_efr_bhv")

# ==============================================================================
# efce experiments - internal sampling (6)
# ==============================================================================

echo "[9/14] starting: cfr (external) on efce..."
$EXECUTABLE \
  --game bargaining_small\
  --algo CFR \
  --equilibrium EFCE \
  --output_file "$RESULTS_DIR/efce_cfr.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED 2>/dev/null &
PIDS+=($!)
NAMES+=("efce_cfr")

echo "[10/14] starting: cfr_in (internal) on efce..."
$EXECUTABLE \
  --game bargaining_small\
  --algo CFR_in \
  --equilibrium EFCE \
  --output_file "$RESULTS_DIR/efce_cfr_in.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED 2>/dev/null &
PIDS+=($!)
NAMES+=("efce_cfr_in")

echo "[11/14] starting: efr(cfps_in) on efce..."
$EXECUTABLE \
  --game bargaining_small\
  --algo EFR_CFPS_in \
  --equilibrium EFCE \
  --output_file "$RESULTS_DIR/efce_efr_cfps_in.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED 2>/dev/null &
PIDS+=($!)
NAMES+=("efce_efr_cfps_in")

echo "[12/14] starting: efr(tips_in) on efce..."
$EXECUTABLE \
  --game bargaining_small\
  --algo EFR_TIPS_in \
  --equilibrium EFCE \
  --output_file "$RESULTS_DIR/efce_efr_tips_in.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED 2>/dev/null &
PIDS+=($!)
NAMES+=("efce_efr_tips_in")

echo "[13/14] starting: efr(csps_in) on efce..."
$EXECUTABLE \
  --game bargaining_small\
  --algo EFR_CSPS_in \
  --equilibrium EFCE \
  --output_file "$RESULTS_DIR/efce_efr_csps_in.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED 2>/dev/null &
PIDS+=($!)
NAMES+=("efce_efr_csps_in")

echo "[14/14] starting: efr(bhv_in) on efce..."
$EXECUTABLE \
  --game bargaining_small\
  --algo EFR_BHV_in \
  --equilibrium EFCE \
  --output_file "$RESULTS_DIR/efce_efr_bhv_in.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED 2>/dev/null &
PIDS+=($!)
NAMES+=("efce_efr_bhv_in")
fi

echo ""
echo "========================================"
if [ "$BATCH" = "all" ]; then
  echo "all 14 experiments launched!"
elif [ "$BATCH" = "1" ]; then
  echo "batch 1: 7 experiments launched!"
else
  echo "batch 2: 7 experiments launched!"
fi
echo "========================================"
echo ""
echo "monitoring progress (updates every 10 seconds)..."
echo ""

# improved monitoring loop with detailed progress
TOTAL=${#PIDS[@]}

while true; do
    COMPLETED=0

    # count completed experiments
    for i in "${!PIDS[@]}"; do
        if ! kill -0 ${PIDS[$i]} 2>/dev/null; then
            COMPLETED=$((COMPLETED + 1))
        fi
    done

    # check if all done
    if [ $COMPLETED -eq $TOTAL ]; then
        break
    fi

    # print status header
    echo "========================================"
    echo "[$(date '+%H:%M:%S')] Progress Update"
    echo "Completed: $COMPLETED/$TOTAL experiments"
    echo "========================================"

    # show progress for each experiment
    for name in "${NAMES[@]}"; do
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
    sleep 10
done

# final wait to ensure all processes are fully cleaned up
wait

echo ""
echo "========================================"
if [ "$BATCH" = "all" ]; then
  echo "all experiments completed!"
elif [ "$BATCH" = "1" ]; then
  echo "batch 1 completed!"
else
  echo "batch 2 completed!"
fi
echo "========================================"
echo ""
echo "results saved to: $RESULTS_DIR/"
echo ""
echo "summary:"
for name in "${NAMES[@]}"; do
    file="$RESULTS_DIR/${name}.txt"
    lines=$(wc -l < "$file" 2>/dev/null || echo "0")
    size=$(du -h "$file" 2>/dev/null | cut -f1 || echo "0")
    printf "  %-25s %4d lines, %5s\n" "$name" "$lines" "$size"
done
echo ""
echo "verify results:"
echo "  ls -lh $RESULTS_DIR/"
echo ""
