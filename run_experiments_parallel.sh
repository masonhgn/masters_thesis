#!/bin/bash

# parallel experiment runner - runs all 14 experiments simultaneously
# designed for multi-core machines (e.g., Google Cloud n1-highmem-16)
# estimated time: ~2 hours on 16-core machine
# estimated cost: ~$2 on Google Cloud

# configuration
EXECUTABLE="./open_spiel-private/build.Release/bin/ltbr/run_corr_dist"
ITERATIONS=1000
REPORT_INTERVAL=20
NUM_SAMPLES=1000  # increased from 5 for better accuracy
SAMPLER="external"  # use external sampling for speed
RANDOM_SEED=42  # seed for reproducibility
RESULTS_DIR="./experiment_results"

# create results directory
mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "parallel experiment runner"
echo "========================================"
echo ""
echo "configuration:"
echo "  iterations: $ITERATIONS"
echo "  report interval: $REPORT_INTERVAL"
echo "  num samples: $NUM_SAMPLES"
echo "  sampler: $SAMPLER"
echo "  results dir: $RESULTS_DIR"
echo ""
echo "starting all 14 experiments in parallel..."
echo "this will take approximately 1.8 hours"
echo ""

# track PIDs for monitoring
declare -a PIDS
declare -a NAMES

# ==============================================================================
# afcce experiments (2)
# ==============================================================================

echo "[1/14] starting: cfr on afcce..."
$EXECUTABLE \
  --game bargaining_small \
  --algo CFR \
  --equilibrium AFCCE \
  --output_file "$RESULTS_DIR/afcce_cfr.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED &
PIDS+=($!)
NAMES+=("afcce_cfr")

echo "[2/14] starting: efr(act) on afcce..."
$EXECUTABLE \
  --game bargaining_small \
  --algo EFR_ACT \
  --equilibrium AFCCE \
  --output_file "$RESULTS_DIR/afcce_efr_act.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED &
PIDS+=($!)
NAMES+=("afcce_efr_act")

# ==============================================================================
# afce experiments (2)
# ==============================================================================

echo "[3/14] starting: cfr on afce..."
$EXECUTABLE \
  --game bargaining_small \
  --algo CFR \
  --equilibrium AFCE \
  --output_file "$RESULTS_DIR/afce_cfr.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED &
PIDS+=($!)
NAMES+=("afce_cfr")

echo "[4/14] starting: efr(act_in) on afce..."
$EXECUTABLE \
  --game bargaining_small \
  --algo EFR_ACT_in \
  --equilibrium AFCE \
  --output_file "$RESULTS_DIR/afce_efr_act_in.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED &
PIDS+=($!)
NAMES+=("afce_efr_act_in")

# ==============================================================================
# efcce experiments - external sampling (4)
# ==============================================================================

echo "[5/14] starting: cfr on efcce..."
$EXECUTABLE \
  --game bargaining_small \
  --algo CFR \
  --equilibrium EFCCE \
  --output_file "$RESULTS_DIR/efcce_cfr.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED &
PIDS+=($!)
NAMES+=("efcce_cfr")

echo "[6/14] starting: efr(csps) on efcce..."
$EXECUTABLE \
  --game bargaining_small \
  --algo EFR_CSPS \
  --equilibrium EFCCE \
  --output_file "$RESULTS_DIR/efcce_efr_csps.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED &
PIDS+=($!)
NAMES+=("efcce_efr_csps")

echo "[7/14] starting: efr(tips) on efcce..."
$EXECUTABLE \
  --game bargaining_small \
  --algo EFR_TIPS \
  --equilibrium EFCCE \
  --output_file "$RESULTS_DIR/efcce_efr_tips.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED &
PIDS+=($!)
NAMES+=("efcce_efr_tips")

echo "[8/14] starting: efr(bhv) on efcce..."
$EXECUTABLE \
  --game bargaining_small \
  --algo EFR_BHV \
  --equilibrium EFCCE \
  --output_file "$RESULTS_DIR/efcce_efr_bhv.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED &
PIDS+=($!)
NAMES+=("efcce_efr_bhv")

# ==============================================================================
# efce experiments - internal sampling (6)
# ==============================================================================

echo "[9/14] starting: cfr (external) on efce..."
$EXECUTABLE \
  --game bargaining_small \
  --algo CFR \
  --equilibrium EFCE \
  --output_file "$RESULTS_DIR/efce_cfr.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED &
PIDS+=($!)
NAMES+=("efce_cfr")

echo "[10/14] starting: cfr_in (internal) on efce..."
$EXECUTABLE \
  --game bargaining_small \
  --algo CFR_in \
  --equilibrium EFCE \
  --output_file "$RESULTS_DIR/efce_cfr_in.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED &
PIDS+=($!)
NAMES+=("efce_cfr_in")

echo "[11/14] starting: efr(cfps_in) on efce..."
$EXECUTABLE \
  --game bargaining_small \
  --algo EFR_CFPS_in \
  --equilibrium EFCE \
  --output_file "$RESULTS_DIR/efce_efr_cfps_in.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED &
PIDS+=($!)
NAMES+=("efce_efr_cfps_in")

echo "[12/14] starting: efr(tips_in) on efce..."
$EXECUTABLE \
  --game bargaining_small \
  --algo EFR_TIPS_in \
  --equilibrium EFCE \
  --output_file "$RESULTS_DIR/efce_efr_tips_in.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED &
PIDS+=($!)
NAMES+=("efce_efr_tips_in")

echo "[13/14] starting: efr(csps_in) on efce..."
$EXECUTABLE \
  --game bargaining_small \
  --algo EFR_CSPS_in \
  --equilibrium EFCE \
  --output_file "$RESULTS_DIR/efce_efr_csps_in.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED &
PIDS+=($!)
NAMES+=("efce_efr_csps_in")

echo "[14/14] starting: efr(bhv_in) on efce..."
$EXECUTABLE \
  --game bargaining_small \
  --algo EFR_BHV_in \
  --equilibrium EFCE \
  --output_file "$RESULTS_DIR/efce_efr_bhv_in.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED &
PIDS+=($!)
NAMES+=("efce_efr_bhv_in")

echo ""
echo "========================================"
echo "all 14 experiments launched!"
echo "========================================"
echo ""
echo "monitoring progress..."
echo "you can check individual experiment outputs in: $RESULTS_DIR/"
echo ""

# wait for all background jobs and track completion
COMPLETED=0
TOTAL=${#PIDS[@]}

while [ $COMPLETED -lt $TOTAL ]; do
    sleep 30  # check every 30 seconds
    COMPLETED=0

    for i in "${!PIDS[@]}"; do
        if ! kill -0 ${PIDS[$i]} 2>/dev/null; then
            COMPLETED=$((COMPLETED + 1))
        fi
    done

    echo "[$(date '+%H:%M:%S')] completed: $COMPLETED/$TOTAL experiments"
done

# final wait to ensure all processes are fully cleaned up
wait

echo ""
echo "========================================"
echo "all experiments completed!"
echo "========================================"
echo ""
echo "results saved to: $RESULTS_DIR/"
echo ""
echo "verify results:"
echo "  ls -lh $RESULTS_DIR/"
echo ""
echo "check for any failures:"
echo "  grep -l 'error\|failed' $RESULTS_DIR/*.txt"
echo ""
