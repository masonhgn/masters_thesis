#!/bin/bash

# single-instance experiments for EFCCE and EFCE
# runs one experiment at a time (sequential)
# usage: bash run_single_instance_experiments.sh

EXECUTABLE="./open_spiel-private/build.Release/bin/ltbr/run_corr_dist"
ITERATIONS=10
REPORT_INTERVAL=1
RANDOM_SEED=42
RESULTS_DIR="./experiment_results_focused"

# samples: 50 for EFCCE, 500 for EFCE (needs more to make sense)
EFCCE_SAMPLES=50
EFCE_SAMPLES=500

SAMPLER="external"

# instance definitions
declare -A INSTANCES
INSTANCES[inst7]="instances/instance_7.txt"
INSTANCES[inst9]="instances/instance_9.txt"
INSTANCES[inst10]="instances/instance_10.txt"

# instance descriptions for logging
declare -A INST_DESC
INST_DESC[inst7]="complement (1,1 1,4 5,0)"
INST_DESC[inst9]="identical (1,1 2,3 2,3)"
INST_DESC[inst10]="asymmetric (3,1 0,5 1,2)"

# efcce algorithms
EFCCE_ALGOS=("CFR" "EFR_BHV" "EFR_CSPS" "EFR_TIPS")

# efce algorithms
EFCE_ALGOS=("CFR" "CFR_in" "EFR_BHV_in" "EFR_CFPS_in" "EFR_CSPS_in" "EFR_TIPS_in")

mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "single-instance experiments"
echo "========================================"
echo ""
echo "instances:"
for inst in inst7 inst9 inst10; do
    echo "  $inst: ${INST_DESC[$inst]}"
done
echo ""
echo "EFCCE: ${#EFCCE_ALGOS[@]} algorithms x 3 instances, $EFCCE_SAMPLES samples"
echo "EFCE:  ${#EFCE_ALGOS[@]} algorithms x 3 instances, $EFCE_SAMPLES samples"
echo ""

TOTAL_EXPS=$(( ${#EFCCE_ALGOS[@]} * 3 + ${#EFCE_ALGOS[@]} * 3 ))
CURRENT=0

# run efcce experiments
echo "========================================"
echo "EFCCE experiments ($EFCCE_SAMPLES samples)"
echo "========================================"

for inst in inst7 inst9 inst10; do
    for algo in "${EFCCE_ALGOS[@]}"; do
        CURRENT=$((CURRENT + 1))
        INST_FILE="${INSTANCES[$inst]}"
        GAME_STR="bargaining_small(instances_file=$(pwd)/$INST_FILE)"
        OUTPUT="$RESULTS_DIR/efcce_${algo,,}_${inst}.txt"

        echo ""
        echo "[$CURRENT/$TOTAL_EXPS] $algo on EFCCE, ${INST_DESC[$inst]}"
        echo "  game: $GAME_STR"
        echo "  output: $OUTPUT"

        $EXECUTABLE \
            --game "$GAME_STR" \
            --algo "$algo" \
            --equilibrium EFCCE \
            --output_file "$OUTPUT" \
            --t $ITERATIONS \
            --report_interval $REPORT_INTERVAL \
            --num_samples $EFCCE_SAMPLES \
            --sampler $SAMPLER \
            --random_seed $RANDOM_SEED

        if [ $? -eq 0 ]; then
            echo "  done."
        else
            echo "  FAILED!"
        fi
    done
done

# run efce experiments (more samples)
echo ""
echo "========================================"
echo "EFCE experiments ($EFCE_SAMPLES samples)"
echo "========================================"

for inst in inst7 inst9 inst10; do
    for algo in "${EFCE_ALGOS[@]}"; do
        CURRENT=$((CURRENT + 1))
        INST_FILE="${INSTANCES[$inst]}"
        GAME_STR="bargaining_small(instances_file=$(pwd)/$INST_FILE)"
        OUTPUT="$RESULTS_DIR/efce_${algo,,}_${inst}.txt"

        echo ""
        echo "[$CURRENT/$TOTAL_EXPS] $algo on EFCE, ${INST_DESC[$inst]}"
        echo "  game: $GAME_STR"
        echo "  output: $OUTPUT"

        $EXECUTABLE \
            --game "$GAME_STR" \
            --algo "$algo" \
            --equilibrium EFCE \
            --output_file "$OUTPUT" \
            --t $ITERATIONS \
            --report_interval $REPORT_INTERVAL \
            --num_samples $EFCE_SAMPLES \
            --sampler $SAMPLER \
            --random_seed $RANDOM_SEED

        if [ $? -eq 0 ]; then
            echo "  done."
        else
            echo "  FAILED!"
        fi
    done
done

echo ""
echo "========================================"
echo "all $TOTAL_EXPS experiments completed!"
echo "results saved to: $RESULTS_DIR/"
echo "========================================"
echo ""
echo "summary:"
for f in "$RESULTS_DIR"/*.txt; do
    if [ -f "$f" ]; then
        name=$(basename "$f")
        lines=$(wc -l < "$f" 2>/dev/null || echo "0")
        printf "  %-45s %4d lines\n" "$name" "$lines"
    fi
done
