# Gomoku GUI

A graphical interface for playing Gomoku against the AlphaZero AI.

## Features

- **PT Model Loading**: Load trained PyTorch models (.pt files)
- **Auto Board Size Detection**: Automatically adjusts board size based on loaded model
- **Color Selection**: Choose to play as Black (first) or White (second)
- **Difficulty Levels**: 6 difficulty settings from Beginner (1k simulations) to Master (100k simulations)
- **Undo Move**: Undo your last move (automatically undoes AI's response too)
- **Game Restart**: Start a new game anytime
- **Visual Feedback**: Shows last move highlight and AI thinking progress
- **Performance Optimized**: Uses vectorized MCTS with GPU acceleration

## Requirements

1. Build the C++ extensions first:
   ```bash
   cd /home/cosmo/omoknuni_quantum
   mkdir build && cd build
   cmake .. -DBUILD_PYTHON_BINDINGS=ON -DWITH_TORCH=ON
   make -j$(nproc)
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the GUI:
```bash
cd /home/cosmo/omoknuni_quantum
python python/gui/gomoku_gui.py
```

### How to Play

1. **Load a Model**: Click "Load Model" and select a .pt file
2. **Choose Your Color**: Select Black (play first) or White (play second)
3. **Set Difficulty**: Choose from 6 difficulty levels
4. **Make Moves**: Click on the board to place stones
5. **Game Controls**:
   - **New Game**: Start fresh
   - **Undo Move**: Take back your last move

### Difficulty Levels

- **Beginner**: 1,000 simulations (fast, weaker play)
- **Easy**: 5,000 simulations
- **Medium**: 10,000 simulations (balanced)
- **Hard**: 20,000 simulations
- **Expert**: 50,000 simulations
- **Master**: 100,000 simulations (slow, strongest play)

### Board Sizes

The GUI automatically detects and adjusts to the model's board size:
- Standard Gomoku: 15x15
- Go-style boards: 19x19, 13x13, 9x9
- Custom sizes supported

### Performance Tips

- The AI uses GPU acceleration if available
- For best performance, the system uses:
  - Vectorized MCTS with 3072 parallel paths
  - Mixed precision computation
  - CUDA graphs and tensor cores
  - Achieves 168,000+ simulations/second on RTX 3060 Ti

### Keyboard Shortcuts

Currently, all controls are mouse-based. Keyboard shortcuts may be added in future versions.

## Troubleshooting

1. **"Import Error" on startup**: Make sure C++ extensions are built
2. **Model fails to load**: Ensure the .pt file is a valid AlphaZero model
3. **Slow AI response**: Check if CUDA is available; CPU-only mode is slower
4. **Out of memory**: Reduce difficulty level or close other GPU applications