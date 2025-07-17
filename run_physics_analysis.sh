#!/bin/bash
# Enhanced MCTS physics analysis script with overnight and parameter sweep support

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_colored() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check system resources
check_system_resources() {
    print_colored $BLUE "=== System Resource Check ==="
    
    # Check available memory
    MEM_AVAILABLE=$(free -m | awk 'NR==2{printf "%.1f", $7/1024}')
    print_colored $GREEN "Available Memory: ${MEM_AVAILABLE}GB"
    
    # Check GPU memory if available
    if command -v nvidia-smi &> /dev/null; then
        GPU_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1)
        GPU_MEM_GB=$(echo "scale=1; $GPU_MEM / 1024" | bc)
        print_colored $GREEN "GPU Memory Available: ${GPU_MEM_GB}GB"
    fi
    
    # Check disk space
    DISK_AVAILABLE=$(df -h "${SCRIPT_DIR}" | awk 'NR==2{print $4}')
    print_colored $GREEN "Available Disk Space: ${DISK_AVAILABLE}"
    
    # Check CPU cores
    CPU_CORES=$(nproc)
    print_colored $GREEN "CPU Cores: ${CPU_CORES}"
    
    echo ""
}

# Function to estimate runtime
estimate_runtime() {
    local preset=$1
    local games=$2
    local sims=$3
    local cpuct_sweep=$4
    
    print_colored $BLUE "=== Runtime Estimation ==="
    
    # Use empirical rate based on actual performance
    # Benchmark shows ~7800 sims/sec for 5K simulations
    # With ~50 moves per game: 5000*50/7800 = ~32 seconds per game
    local base_rate=0.03  # ~33 seconds per game for 5000 sims
    local base_sims=5000
    
    # Calculate actual rate based on simulations
    local actual_rate
    if [[ -n "$sims" ]]; then
        # Rate scales inversely with simulations
        actual_rate=$(echo "scale=4; $base_rate * $base_sims / $sims" | bc)
    else
        actual_rate=$base_rate
    fi
    
    # c_puct sweep doesn't multiply total time, just distributes games differently
    # Same total games, just tested across different c_puct values
    
    # Total time = games / rate
    total_seconds=$(echo "scale=0; $games / $actual_rate" | bc)
    
    # Convert to human readable
    hours=$((total_seconds / 3600))
    minutes=$(((total_seconds % 3600) / 60))
    
    print_colored $YELLOW "Estimated Runtime: ${hours}h ${minutes}m"
    print_colored $BLUE "Game rate: ${actual_rate} games/sec"
    
    if [[ $hours -gt 8 ]]; then
        print_colored $RED "WARNING: This analysis will take more than 8 hours!"
        print_colored $YELLOW "Consider running overnight or using a smaller preset."
    elif [[ $hours -gt 2 ]]; then
        print_colored $YELLOW "This is a long-running analysis (${hours}h ${minutes}m)"
        print_colored $YELLOW "Consider using screen or tmux for background execution."
    fi
    
    echo ""
}

# Function to show help
show_help() {
    echo "Enhanced MCTS Physics Analysis Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Presets:"
    echo "  --preset quick         : Quick analysis (10 games, 1000 sims)"
    echo "  --preset standard      : Standard analysis (50 games, 5000 sims)"
    echo "  --preset comprehensive : Comprehensive analysis (100 games, 10000 sims)"
    echo "  --preset deep          : Deep analysis with parameter sweeps (200 games)"
    echo "  --preset overnight     : Overnight analysis (1000 games, 25000 sims)"
    echo ""
    echo "Parameter Sweeps:"
    echo "  --c-puct-sweep         : Enable c_puct parameter sweep"
    echo "  --c-puct-range MIN MAX : Set c_puct range (default: 0.5 3.0)"
    echo "  --c-puct-steps N       : Number of c_puct values (default: 10)"
    echo ""
    echo "Overrides:"
    echo "  --games N              : Override number of games"
    echo "  --sims N               : Override simulations per game"
    echo "  --game-type TYPE       : Game type (gomoku, go, chess)"
    echo "  --output DIR           : Output directory"
    echo "  --evaluator-type TYPE  : Evaluator type (resnet, random, fast_random)"
    echo ""
    echo "Examples:"
    echo "  $0 --preset overnight                    # Full overnight analysis"
    echo "  $0 --preset standard --c-puct-sweep     # Standard with c_puct sweep"
    echo "  $0 --preset quick --games 20 --sims 2000 # Quick with custom params"
    echo "  $0 --preset quick --evaluator-type random # Quick with random evaluator"
    echo ""
    exit 0
}

# Store all original arguments before parsing
ORIG_ARGS=("$@")

# Parse arguments for resource checking
PRESET="standard"
GAMES=""
SIMS=""
CPUCT_SWEEP="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --preset)
            PRESET="$2"
            shift 2
            ;;
        --games)
            GAMES="$2"
            shift 2
            ;;
        --sims)
            SIMS="$2"
            shift 2
            ;;
        --c-puct-sweep)
            CPUCT_SWEEP="true"
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            # Keep other arguments for passing to Python script
            shift
            ;;
    esac
done

# Set defaults based on preset if not overridden
case $PRESET in
    "quick")
        GAMES=${GAMES:-10}
        SIMS=${SIMS:-1000}
        ;;
    "standard")
        GAMES=${GAMES:-50}
        SIMS=${SIMS:-5000}
        ;;
    "comprehensive")
        GAMES=${GAMES:-100}
        SIMS=${SIMS:-5000}  # Reduced from 10K for better performance
        ;;
    "deep")
        GAMES=${GAMES:-200}
        SIMS=${SIMS:-10000}
        ;;
    "overnight")
        GAMES=${GAMES:-1000}
        SIMS=${SIMS:-5000}
        ;;
esac

# Print banner
print_colored $BLUE "================================================================"
print_colored $BLUE "  Enhanced MCTS Physics Analysis - Overnight Data Generation"
print_colored $BLUE "================================================================"
echo ""

# Check system resources
check_system_resources

# Estimate runtime
estimate_runtime "$PRESET" "$GAMES" "$SIMS" "$CPUCT_SWEEP"

# Activate virtual environment if not already activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    print_colored $YELLOW "Activating virtual environment..."
    source ~/venv/bin/activate
fi

# Export library paths - include venv site-packages
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="${SITE_PACKAGES}:${SCRIPT_DIR}/python:${SCRIPT_DIR}/build_cpp/lib/Release:${LD_LIBRARY_PATH}"

# Set environment variables for maximum performance and VRAM utilization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048,expandable_segments:True,garbage_collection_threshold:0.9,roundup_power2_divisions:4
export CUDA_LAUNCH_BLOCKING=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export CUDA_CACHE_MAXSIZE=2147483648
export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24
export NUMBA_NUM_THREADS=24

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${SCRIPT_DIR}/physics_analysis_${PRESET}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# Log system information
print_colored $GREEN "Starting MCTS physics analysis..."
print_colored $GREEN "Preset: $PRESET"
print_colored $GREEN "Games: $GAMES"
print_colored $GREEN "Simulations per game: $SIMS"
print_colored $GREEN "c_puct sweep: $CPUCT_SWEEP"
print_colored $GREEN "Output directory: $OUTPUT_DIR"
print_colored $GREEN "Start time: $(date)"
echo ""

# Run the physics analysis with enhanced logging
python "${SCRIPT_DIR}/run_mcts_physics_analysis.py" \
    --preset "$PRESET" \
    --output "$OUTPUT_DIR" \
    --games "$GAMES" \
    --sims "$SIMS" \
    "${ORIG_ARGS[@]}" \
    2>&1 | tee "$OUTPUT_DIR/analysis_log.txt"

# Capture exit code
EXIT_CODE=${PIPESTATUS[0]}

# Final status
echo ""
print_colored $BLUE "================================================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    print_colored $GREEN "Analysis completed successfully!"
else
    print_colored $RED "Analysis failed with exit code $EXIT_CODE"
fi
print_colored $GREEN "End time: $(date)"
print_colored $GREEN "Results saved to: $OUTPUT_DIR"
print_colored $BLUE "================================================================"

exit $EXIT_CODE