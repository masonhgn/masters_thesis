#!/bin/bash

# parallel experiment runner - EFCCE and EFCE only
# usage: bash run_efcce_efce_parallel.sh

# configuration
EXECUTABLE="./open_spiel-private/build.Release/bin/ltbr/run_corr_dist"
ITERATIONS=1000
REPORT_INTERVAL=20
NUM_SAMPLES=100
SAMPLER="external"
RANDOM_SEED=42
RESULTS_DIR="./experiment_results_focused"

mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "EFCCE + EFCE parallel experiment runner"
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

declare -a PIDS
declare -a NAMES

# ==============================================================================
# efcce experiments - external sampling (4)
# ==============================================================================

echo "[1/10] starting: cfr on efcce..."
$EXECUTABLE \
  --game bargaining_small \
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

echo "[2/10] starting: efr(csps) on efcce..."
$EXECUTABLE \
  --game bargaining_small \
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

echo "[3/10] starting: efr(tips) on efcce..."
$EXECUTABLE \
  --game bargaining_small \
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

echo "[4/10] starting: efr(bhv) on efcce..."
$EXECUTABLE \
  --game bargaining_small \
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

echo "[5/10] starting: cfr (external) on efce..."
$EXECUTABLE \
  --game bargaining_small \
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

echo "[6/10] starting: cfr_in (internal) on efce..."
$EXECUTABLE \
  --game bargaining_small \
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

echo "[7/10] starting: efr(cfps_in) on efce..."
$EXECUTABLE \
  --game bargaining_small \
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

echo "[8/10] starting: efr(tips_in) on efce..."
$EXECUTABLE \
  --game bargaining_small \
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

echo "[9/10] starting: efr(csps_in) on efce..."
$EXECUTABLE \
  --game bargaining_small \
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

echo "[10/10] starting: efr(bhv_in) on efce..."
$EXECUTABLE \
  --game bargaining_small \
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

echo ""
echo "========================================"
echo "all 10 experiments launched!"
echo "========================================"
echo ""
echo "monitoring progress (updates every 10 seconds)..."
echo ""

TOTAL=${#PIDS[@]}

while true; do
    COMPLETED=0

    for i in "${!PIDS[@]}"; do
        if ! kill -0 ${PIDS[$i]} 2>/dev/null; then
            COMPLETED=$((COMPLETED + 1))
        fi
    done

    if [ $COMPLETED -eq $TOTAL ]; then
        break
    fi

    echo "========================================"
    echo "[$(date '+%H:%M:%S')] Progress Update"
    echo "Completed: $COMPLETED/$TOTAL experiments"
    echo "========================================"

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

wait

echo ""
echo "========================================"
echo "all experiments completed!"
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

