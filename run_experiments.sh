#!/bin/bash

# experiment runner for testing CFR and EFR variants on different equilibrium concepts
# based on hypotheses about convergence behavior

# configuration
EXECUTABLE="./open_spiel-private/build.Release/bin/ltbr/run_corr_dist"
ITERATIONS=10
REPORT_INTERVAL=1
NUM_SAMPLES=100  # increased from 5 for better accuracy
SAMPLER="external"  # use external sampling for speed (null=slowest, external=fast, outcome=fastest)
RANDOM_SEED=42  # seed for reproducibility
RESULTS_DIR="./experiment_results"

# create results directory
mkdir -p "$RESULTS_DIR"

echo "starting experiments..."
echo "results will be saved to: $RESULTS_DIR"
echo ""
echo "configuration:"
echo "  iterations: $ITERATIONS"
echo "  report interval: $REPORT_INTERVAL"
echo "  num samples: $NUM_SAMPLES (for correlation device sampling)"
echo ""

# ==============================================================================
# hypothesis 1: cfr (and all efr variants) should have low cce and ce dist in zero-sum game
# ==============================================================================

# echo "=========================================="
# echo "hypothesis 1: zero-sum game convergence"
# echo "=========================================="

# # cfr on bargaining_zerosum for cce
# echo "running: cfr on bargaining_zerosum for cce..."
# $EXECUTABLE \
#   --game bargaining_zerosum \
#   --algo CFR \
#   --equilibrium CCE \
#   --output_file "$RESULTS_DIR/zerosum_cfr_cce.txt" \
#   --t $ITERATIONS \
#   --report_interval $REPORT_INTERVAL \
#   --num_samples $NUM_SAMPLES \
#   --sampler $SAMPLER \
#   --random_seed $RANDOM_SEED

# # cfr on bargaining_zerosum for ce
# echo "running: cfr on bargaining_zerosum for ce..."
# $EXECUTABLE \
#   --game bargaining_zerosum \
#   --algo CFR \
#   --equilibrium CE \
#   --output_file "$RESULTS_DIR/zerosum_cfr_ce.txt" \
#   --t $ITERATIONS \
#   --report_interval $REPORT_INTERVAL \
#   --num_samples $NUM_SAMPLES \
#   --sampler $SAMPLER \
#   --random_seed $RANDOM_SEED

# echo ""

# ==============================================================================
# hypothesis 2: efr(action) for afcce and afce
# ==============================================================================

echo "=========================================="
echo "hypothesis 2: agent-form equilibria"
echo "=========================================="

# baseline: cfr on afcce
echo "running: cfr on afcce (baseline)..."
$EXECUTABLE \
  --game bargaining_small \
  --algo CFR \
  --equilibrium AFCCE \
  --output_file "$RESULTS_DIR/afcce_cfr.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED

# efr(action) on afcce - should show low afcce dist
echo "running: efr(act) on afcce..."
$EXECUTABLE \
  --game bargaining_small \
  --algo EFR_ACT \
  --equilibrium AFCCE \
  --output_file "$RESULTS_DIR/afcce_efr_act.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED

# baseline: cfr on afce
echo "running: cfr on afce (baseline)..."
$EXECUTABLE \
  --game bargaining_small \
  --algo CFR \
  --equilibrium AFCE \
  --output_file "$RESULTS_DIR/afce_cfr.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED

# efr(action {in}) on afce - should show low afce dist
echo "running: efr(act_in) on afce..."
$EXECUTABLE \
  --game bargaining_small \
  --algo EFR_ACT_in \
  --equilibrium AFCE \
  --output_file "$RESULTS_DIR/afce_efr_act_in.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED

echo ""

# ==============================================================================
# hypothesis 3: efr variants on efcce (default {ex})
# expected: efr(bhv) < efr(tips) < efr(csps) < cfr (lower is better)
# ==============================================================================

echo "=========================================="
echo "hypothesis 3: efcce with {ex} variants"
echo "=========================================="

# baseline: cfr on efcce
echo "running: cfr on efcce (baseline)..."
$EXECUTABLE \
  --game bargaining_small \
  --algo CFR \
  --equilibrium EFCCE \
  --output_file "$RESULTS_DIR/efcce_cfr.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED

# efr(csps) on efcce - should be better than cfr
echo "running: efr(csps) on efcce..."
$EXECUTABLE \
  --game bargaining_small \
  --algo EFR_CSPS \
  --equilibrium EFCCE \
  --output_file "$RESULTS_DIR/efcce_efr_csps.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED

# efr(tips) on efcce - should be better than csps
echo "running: efr(tips) on efcce..."
$EXECUTABLE \
  --game bargaining_small \
  --algo EFR_TIPS \
  --equilibrium EFCCE \
  --output_file "$RESULTS_DIR/efcce_efr_tips.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED

# efr(bhv) on efcce - should be the best but most expensive
echo "running: efr(bhv) on efcce (most expensive)..."
$EXECUTABLE \
  --game bargaining_small \
  --algo EFR_BHV \
  --equilibrium EFCCE \
  --output_file "$RESULTS_DIR/efcce_efr_bhv.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED

echo ""

# ==============================================================================
# hypothesis 4: efr variants on efce (with {in})
# expected: efr(bhv_in) >> efr(tips_in) >> efr(csps_in) >> cfr_in (lower is better)
# note: all algorithms use internal sampling for this hypothesis
# ==============================================================================

echo "=========================================="
echo "hypothesis 4: efce with {in} variants"
echo "=========================================="

# baseline: cfr with external sampling on efce
echo "running: cfr (external) on efce (baseline)..."
$EXECUTABLE \
  --game bargaining_small \
  --algo CFR \
  --equilibrium EFCE \
  --output_file "$RESULTS_DIR/efce_cfr.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED

# baseline: cfr with internal sampling on efce
echo "running: cfr_in (internal) on efce (baseline for internal algorithms)..."
$EXECUTABLE \
  --game bargaining_small \
  --algo CFR_in \
  --equilibrium EFCE \
  --output_file "$RESULTS_DIR/efce_cfr_in.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED

# efr(cfps {in}) on efce - counterfactual partial sequence (internal-only)
echo "running: efr(cfps_in) on efce..."
$EXECUTABLE \
  --game bargaining_small \
  --algo EFR_CFPS_in \
  --equilibrium EFCE \
  --output_file "$RESULTS_DIR/efce_efr_cfps_in.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED

# efr(tips {in}) on efce - should be better than cfps_in
echo "running: efr(tips_in) on efce..."
$EXECUTABLE \
  --game bargaining_small \
  --algo EFR_TIPS_in \
  --equilibrium EFCE \
  --output_file "$RESULTS_DIR/efce_efr_tips_in.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED

# efr(csps {in}) on efce - internal-only version of causal partial sequence
echo "running: efr(csps_in) on efce..."
$EXECUTABLE \
  --game bargaining_small \
  --algo EFR_CSPS_in \
  --equilibrium EFCE \
  --output_file "$RESULTS_DIR/efce_efr_csps_in.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED

# efr(bhv {in}) on efce - behavioral with internal only
echo "running: efr(bhv_in) on efce..."
$EXECUTABLE \
  --game bargaining_small \
  --algo EFR_BHV_in \
  --equilibrium EFCE \
  --output_file "$RESULTS_DIR/efce_efr_bhv_in.txt" \
  --t $ITERATIONS \
  --report_interval $REPORT_INTERVAL \
  --num_samples $NUM_SAMPLES \
  --sampler $SAMPLER \
  --random_seed $RANDOM_SEED

echo ""
echo "=========================================="
echo "all experiments completed!"
echo "results saved to: $RESULTS_DIR"
echo "=========================================="
