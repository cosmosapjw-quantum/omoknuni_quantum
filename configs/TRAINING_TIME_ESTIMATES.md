# Conservative Training Time Estimates

## Hardware: Ryzen 9 5900X (12 cores) + RTX 3060 Ti (8GB VRAM)

### Gomoku (15x15 board)
**Configuration**: `ryzen9_3060ti_optimized.yaml`
- **Per Game**: 5-7 seconds (1600 simulations/move)
- **Per Iteration**: 
  - Self-play (200 games): ~20-25 minutes
  - Neural network training: ~5-10 minutes
  - Arena evaluation (120 games): ~15-20 minutes
  - **Total per iteration**: ~45-55 minutes
- **Full Training (300 iterations)**: **225-275 hours (9-11 days)**
- **Reasonable checkpoint (50 iterations)**: **38-46 hours (1.5-2 days)**

### Chess
**Configuration**: `chess_ryzen9_3060ti.yaml`
- **Per Game**: 15-20 seconds (2400 simulations/move, ~80 moves/game)
- **Per Iteration**:
  - Self-play (150 games): ~40-50 minutes
  - Neural network training: ~10-15 minutes
  - Arena evaluation (100 games): ~25-35 minutes
  - **Total per iteration**: ~75-100 minutes
- **Full Training (300 iterations)**: **375-500 hours (16-21 days)**
- **Reasonable checkpoint (50 iterations)**: **63-83 hours (2.5-3.5 days)**

### Go 19x19
**Configuration**: `go_19x19_ryzen9_3060ti.yaml`
- **Per Game**: 30-45 seconds (3200 simulations/move, ~200 moves/game)
- **Per Iteration**:
  - Self-play (100 games): ~50-75 minutes
  - Neural network training: ~15-20 minutes
  - Arena evaluation (60 games): ~30-45 minutes
  - **Total per iteration**: ~95-140 minutes
- **Full Training (300 iterations)**: **475-700 hours (20-29 days)**
- **Reasonable checkpoint (50 iterations)**: **79-117 hours (3.3-4.9 days)**

## Practical Recommendations

### Quick Testing & Development
```bash
# 1-hour test run (see if everything works)
python train.py --config configs/gomoku_classical.yaml --iterations 1

# Overnight training (8-10 hours, ~10-15 iterations)
python train.py --config configs/ryzen9_3060ti_optimized.yaml --iterations 10
```

### Meaningful Results
```bash
# Weekend project (48-72 hours)
# Gomoku: 50-80 iterations (noticeable improvement)
python train.py --config configs/ryzen9_3060ti_optimized.yaml --iterations 50

# Chess: 30-40 iterations (basic competence)
python train.py --config configs/chess_ryzen9_3060ti.yaml --iterations 30
```

### Production Quality
```bash
# 1-2 weeks commitment
# Gomoku: 150-200 iterations (strong play)
python train.py --config configs/ryzen9_3060ti_optimized.yaml --iterations 150

# Chess: 100 iterations (decent amateur level)
python train.py --config configs/chess_ryzen9_3060ti.yaml --iterations 100
```

## Time-Saving Tips

1. **Start with Gomoku** - Fastest iteration time, good for testing
2. **Use checkpoints** - Training auto-saves every 50 iterations
3. **Monitor early** - Check tensorboard after 5-10 iterations
4. **Reduce scope** - Consider fewer iterations or smaller networks initially

## Performance Monitoring

```bash
# Check training speed after first iteration
grep "Iteration.*completed in" training.log

# Estimate total time
python -c "
import sys
minutes_per_iter = float(sys.argv[1])
total_iters = int(sys.argv[2])
hours = (minutes_per_iter * total_iters) / 60
days = hours / 24
print(f'Total time: {hours:.1f} hours ({days:.1f} days)')
" 50 300  # Example: 50 min/iter, 300 iterations
```

## Early Stopping Milestones

### Gomoku
- **10 iterations** (~8 hours): Learns basic patterns
- **25 iterations** (~20 hours): Decent tactical play
- **50 iterations** (~40 hours): Good strategic understanding
- **100 iterations** (~80 hours): Strong club player level

### Chess  
- **10 iterations** (~15 hours): Learns piece movement
- **25 iterations** (~35 hours): Basic tactics (forks, pins)
- **50 iterations** (~70 hours): Positional understanding
- **100 iterations** (~140 hours): ~1500-1800 ELO equivalent

### Go 19x19
- **10 iterations** (~20 hours): Territory concepts
- **25 iterations** (~50 hours): Basic life and death
- **50 iterations** (~100 hours): Fighting and influence
- **100 iterations** (~200 hours): Decent amateur level

## Resource Usage During Training

### Expected Load
- **GPU**: 95-100% utilization (normal)
- **CPU**: 50-80% utilization (self-play on CPU)
- **RAM**: 15-25GB used
- **VRAM**: 5-7GB used (leaves headroom)
- **Disk I/O**: Moderate (checkpoints every 50 iterations)

### Temperature Guidelines
- **GPU**: Keep under 83°C (thermal throttle)
- **CPU**: Keep under 85°C (boost limits)
- Consider additional cooling or reduced settings if temperatures exceed these

## Realistic Expectations

**For most users**, I recommend:
- **Gomoku**: 50-100 iterations (2-4 days) for satisfying results
- **Chess**: 50 iterations (3 days) for basic competence
- **Go**: Start with 9x9 board for faster iteration

These estimates are conservative and include overhead. Actual times may be 20-30% faster with good cooling and no interruptions.