#!/bin/bash
# Script to run MCTS self-play performance optimization

# Activate virtual environment if it exists
if [ -d "~/venv" ]; then
    source ~/venv/bin/activate
fi

# Set environment variables
export PYTHONPATH="${PWD}/python:${PYTHONPATH}"
export LD_LIBRARY_PATH="${PWD}/python:${LD_LIBRARY_PATH}"

# Default parameters
CONFIG="configs/optuna_optimization_base.yaml"
OUTPUT_DIR="optimization_results"
N_TRIALS=100
NUM_GAMES=3
MOVES_PER_GAME=30
GAME_TYPE="gomoku"
BOARD_SIZE=15
DEVICE="cuda"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --n-trials)
            N_TRIALS="$2"
            shift 2
            ;;
        --num-games)
            NUM_GAMES="$2"
            shift 2
            ;;
        --moves-per-game)
            MOVES_PER_GAME="$2"
            shift 2
            ;;
        --game-type)
            GAME_TYPE="$2"
            shift 2
            ;;
        --board-size)
            BOARD_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --config FILE           Base configuration file (default: configs/optuna_optimization_base.yaml)"
            echo "  --output-dir DIR        Output directory (default: optimization_results)"
            echo "  --n-trials N            Number of Optuna trials (default: 100)"
            echo "  --num-games N           Number of games per trial (default: 3)"
            echo "  --moves-per-game N      Number of moves per game (default: 30)"
            echo "  --game-type TYPE        Game type: chess, go, gomoku (default: gomoku)"
            echo "  --board-size N          Board size (default: 15)"
            echo "  --device DEVICE         Device: cuda or cpu (default: cuda)"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FULL_OUTPUT_DIR="${OUTPUT_DIR}_${GAME_TYPE}_${TIMESTAMP}"

echo "Starting MCTS Performance Optimization"
echo "====================================="
echo "Configuration: ${CONFIG}"
echo "Output directory: ${FULL_OUTPUT_DIR}"
echo "Number of trials: ${N_TRIALS}"
echo "Games per trial: ${NUM_GAMES}"
echo "Moves per game: ${MOVES_PER_GAME}"
echo "Game type: ${GAME_TYPE}"
echo "Board size: ${BOARD_SIZE}"
echo "Device: ${DEVICE}"
echo "====================================="

# Run optimization
python optimize_selfplay_performance.py \
    --config "${CONFIG}" \
    --output-dir "${FULL_OUTPUT_DIR}" \
    --n-trials ${N_TRIALS} \
    --num-games ${NUM_GAMES} \
    --moves-per-game ${MOVES_PER_GAME} \
    --game-type ${GAME_TYPE} \
    --board-size ${BOARD_SIZE} \
    --device ${DEVICE} \
    --storage "sqlite:///${FULL_OUTPUT_DIR}/optuna_study.db" \
    --pruner median \
    --sampler tpe

# Check if optimization completed successfully
if [ $? -eq 0 ]; then
    echo "====================================="
    echo "Optimization completed successfully!"
    echo "Results saved to: ${FULL_OUTPUT_DIR}"
    echo ""
    echo "Best configuration: ${FULL_OUTPUT_DIR}/best_config.yaml"
    echo "All results: ${FULL_OUTPUT_DIR}/optimization_results.json"
    echo "====================================="
else
    echo "Optimization failed!"
    exit 1
fi