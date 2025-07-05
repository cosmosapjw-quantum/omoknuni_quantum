#!/usr/bin/env python3
"""
Gomoku GUI for playing against AlphaZero AI
Supports PT file loading, board size detection, undo/restart, color selection, and difficulty levels
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import torch
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List
import threading
import queue
import logging

# Add parent directory to path for imports and prioritize it
python_path = str(Path(__file__).parent.parent)
if python_path in sys.path:
    sys.path.remove(python_path)
sys.path.insert(0, python_path)  # Insert at beginning to prioritize local code

# Setup logging
logger = logging.getLogger(__name__)

try:
    import alphazero_py
    from mcts.core.mcts import MCTS, MCTSConfig
    from mcts.gpu.gpu_game_states import GameType
    from mcts.neural_networks.resnet_evaluator import ResNetEvaluator
    from mcts.neural_networks.nn_framework import ModelLoader, ModelMetadata
    from mcts.utils.config_system import AlphaZeroConfig
except ImportError as e:
    messagebox.showerror("Import Error", f"Failed to import required modules: {e}\n\nMake sure to build the C++ extensions first.")
    sys.exit(1)


class GomokuGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AlphaZero Gomoku")
        self.root.resizable(False, False)
        
        # Game state
        self.board_size = 15
        self.board = None
        self.game_state = None
        self.move_history = []
        self.ai_evaluator = None
        self.mcts = None
        self.ai_thread = None
        self.ai_queue = queue.Queue()
        
        # UI state
        self.cell_size = 40
        self.margin = 30
        self.stone_radius = 16
        self.current_player = 0  # 0 = black, 1 = white
        self.human_color = 0  # 0 = black, 1 = white
        self.ai_thinking = False
        self.game_over = False
        self.last_move = None
        
        # Difficulty settings
        self.difficulty_levels = {
            "Beginner": 1000,
            "Easy": 5000,
            "Medium": 10000,
            "Hard": 20000,
            "Expert": 50000,
            "Master": 100000
        }
        self.current_difficulty = "Medium"
        
        # Resignation settings
        self.resign_threshold = -0.9  # AI resigns if evaluation drops below this
        self.resign_check_moves = 5   # Number of consecutive moves to check
        self.ai_evaluation_history = []  # Track AI's position evaluations
        
        self.setup_ui()
        self.new_game()
        
    def setup_ui(self):
        """Create the GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Game Controls", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Model selection
        model_frame = ttk.Frame(control_frame)
        model_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(model_frame, text="AI Model:").grid(row=0, column=0, padx=(0, 5))
        self.model_label = ttk.Label(model_frame, text="No model loaded", foreground="red")
        self.model_label.grid(row=0, column=1, padx=(0, 10))
        ttk.Button(model_frame, text="Load Model", command=self.load_model).grid(row=0, column=2)
        ttk.Button(model_frame, text="Load with Config", command=self.load_model_with_config).grid(row=0, column=3)
        
        # Player color selection
        color_frame = ttk.Frame(control_frame)
        color_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=(0, 20))
        
        ttk.Label(color_frame, text="Play as:").grid(row=0, column=0, padx=(0, 5))
        self.color_var = tk.StringVar(value="Black")
        ttk.Radiobutton(color_frame, text="Black (First)", variable=self.color_var, 
                       value="Black", command=self.on_color_change).grid(row=0, column=1, padx=5)
        ttk.Radiobutton(color_frame, text="White (Second)", variable=self.color_var, 
                       value="White", command=self.on_color_change).grid(row=0, column=2, padx=5)
        
        # Difficulty selection
        diff_frame = ttk.Frame(control_frame)
        diff_frame.grid(row=1, column=1, sticky=(tk.W, tk.E))
        
        ttk.Label(diff_frame, text="Difficulty:").grid(row=0, column=0, padx=(0, 5))
        self.difficulty_var = tk.StringVar(value=self.current_difficulty)
        difficulty_menu = ttk.Combobox(diff_frame, textvariable=self.difficulty_var, 
                                     values=list(self.difficulty_levels.keys()), 
                                     state="readonly", width=10)
        difficulty_menu.grid(row=0, column=1)
        difficulty_menu.bind("<<ComboboxSelected>>", self.on_difficulty_change)
        
        # Game buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=1, column=2, sticky=(tk.W, tk.E))
        
        ttk.Button(button_frame, text="New Game", command=self.new_game).grid(row=0, column=0, padx=2)
        ttk.Button(button_frame, text="Undo Move", command=self.undo_move).grid(row=0, column=1, padx=2)
        self.resign_button = ttk.Button(button_frame, text="Resign", command=self.resign_game)
        self.resign_button.grid(row=0, column=2, padx=2)
        
        # Board canvas
        canvas_size = 2 * self.margin + (self.board_size - 1) * self.cell_size
        self.canvas = tk.Canvas(main_frame, width=canvas_size, height=canvas_size, 
                               bg="#DEB887", highlightthickness=1, highlightbackground="black")
        self.canvas.grid(row=1, column=0, columnspan=2)
        self.canvas.bind("<Button-1>", self.on_board_click)
        
        # Status bar
        self.status_var = tk.StringVar(value="Load an AI model to start playing")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Info panel
        info_frame = ttk.LabelFrame(main_frame, text="Game Info", padding="10")
        info_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.info_text = tk.Text(info_frame, height=4, width=50, state=tk.DISABLED, wrap=tk.WORD)
        self.info_text.grid(row=0, column=0)
        
        # Progress bar for AI thinking
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        self.progress.grid_remove()
        
    def load_model(self):
        """Load a PT file and configure the AI"""
        filename = filedialog.askopenfilename(
            title="Select AI Model",
            filetypes=[("PyTorch Models", "*.pt"), ("All Files", "*.*")]
        )
        
        if not filename:
            return
            
        try:
            # Show loading message
            self.status_var.set("Loading model...")
            self.root.update()
            
            # Load model and detect board size
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Create evaluator with automatic config loading
            # Handle both old and new ResNetEvaluator APIs
            try:
                # Try new API with network_config parameter
                from mcts.neural_networks.resnet_evaluator import load_config_for_model
                network_config = load_config_for_model(filename)
                
                self.ai_evaluator = ResNetEvaluator(
                    checkpoint_path=filename,
                    device=device,
                    game_type='gomoku',
                    network_config=network_config
                )
                if network_config:
                    self.status_var.set(f"Loaded with config: {network_config.num_res_blocks} blocks, {network_config.num_filters} filters")
                
            except TypeError as e:
                if "unexpected keyword argument 'network_config'" in str(e):
                    # Fallback to old API without network_config
                    logger.info("Using legacy ResNetEvaluator API")
                    self.ai_evaluator = ResNetEvaluator(
                        checkpoint_path=filename,
                        device=device,
                        game_type='gomoku'
                    )
                    self.status_var.set("Loaded with legacy API (may use default architecture)")
                else:
                    raise
            except Exception as e:
                # If checkpoint loading fails completely, try to create new model with config
                logger.warning(f"Checkpoint loading failed: {e}")
                self.status_var.set("Checkpoint failed, trying with manual config...")
                self.root.update()
                
                try:
                    from mcts.neural_networks.resnet_evaluator import load_config_for_model
                    network_config = load_config_for_model(filename)
                    
                    if network_config is not None:
                        try:
                            # Try new API
                            self.ai_evaluator = ResNetEvaluator(
                                checkpoint_path=None,
                                device=device,
                                game_type='gomoku',
                                network_config=network_config
                            )
                            self.status_var.set("Created new model with detected config")
                        except TypeError:
                            # Fallback to old API
                            self.ai_evaluator = ResNetEvaluator(
                                checkpoint_path=None,
                                device=device,
                                game_type='gomoku'
                            )
                            self.status_var.set("Created new model with default config (legacy API)")
                    else:
                        # No config found, use default
                        self.ai_evaluator = ResNetEvaluator(
                            checkpoint_path=None,
                            device=device,
                            game_type='gomoku'
                        )
                        self.status_var.set("Created new model with default config")
                except ImportError:
                    # load_config_for_model not available in old version
                    self.ai_evaluator = ResNetEvaluator(
                        checkpoint_path=None,
                        device=device,
                        game_type='gomoku'
                    )
                    self.status_var.set("Created new model with default config (legacy version)")
                
                self.root.update()
            
            # Get model metadata for board size detection
            metadata = self.ai_evaluator.model.metadata
            
            # Try to detect board size from model architecture
            if hasattr(metadata, 'board_size'):
                detected_size = metadata.board_size
            else:
                # Default to 15 for Gomoku
                detected_size = 15
                
            # Update board size if different
            if detected_size != self.board_size:
                self.board_size = detected_size
                self.setup_board_canvas()
            
            # Configure MCTS
            config = MCTSConfig(
                game_type=GameType.GOMOKU,
                board_size=self.board_size,
                num_simulations=self.difficulty_levels[self.current_difficulty],
                min_wave_size=3072,
                max_wave_size=3072,
                adaptive_wave_sizing=False,
                temperature=0.0,  # Deterministic play for consistent user experience
                dirichlet_alpha=0.3,  # Keep default valid value (must be > 0)
                dirichlet_epsilon=0.0,  # Zero epsilon disables noise completely
                device=device,
                use_mixed_precision=True,
                use_cuda_graphs=True,
                use_tensor_cores=True,
                memory_pool_size_mb=2048,
                max_tree_nodes=500000
            )
            
            # Create MCTS instance
            self.mcts = MCTS(config, self.ai_evaluator)
            self.mcts.optimize_for_hardware()
            
            # Update UI
            model_name = Path(filename).name
            self.model_label.config(text=f"{model_name} (Board: {self.board_size}x{self.board_size})", 
                                  foreground="green")
            self.status_var.set(f"Model loaded successfully! Board size: {self.board_size}x{self.board_size}")
            
            # Update info
            self.update_info(f"Loaded model: {model_name}\n"
                           f"Board size: {self.board_size}x{self.board_size}\n"
                           f"Device: {device.upper()}\n"
                           f"Difficulty: {self.current_difficulty} ({self.difficulty_levels[self.current_difficulty]} simulations)")
            
            # Start new game
            self.new_game()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            self.status_var.set("Failed to load model")
    
    def load_model_with_config(self):
        """Load a model and explicitly specify a config file"""
        model_filename = filedialog.askopenfilename(
            title="Select AI Model",
            filetypes=[("PyTorch Models", "*.pt"), ("All Files", "*.*")]
        )
        
        if not model_filename:
            return
            
        config_filename = filedialog.askopenfilename(
            title="Select Config File",
            filetypes=[("YAML Config", "*.yaml"), ("YAML Config", "*.yml"), ("All Files", "*.*")]
        )
        
        if not config_filename:
            return
            
        try:
            # Show loading message
            self.status_var.set("Loading model with custom config...")
            self.root.update()
            
            # Load config and extract network config
            full_config = AlphaZeroConfig.load(config_filename)
            network_config = full_config.network
            
            # Load model and detect board size
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Create evaluator with specified config
            self.ai_evaluator = ResNetEvaluator(
                checkpoint_path=model_filename,
                device=device,
                game_type='gomoku',
                network_config=network_config
            )
            
            # Get model metadata for board size detection
            metadata = self.ai_evaluator.model.metadata
            
            # Try to detect board size from config or model
            if hasattr(full_config.game, 'board_size'):
                detected_size = full_config.game.board_size
            elif hasattr(metadata, 'board_size'):
                detected_size = metadata.board_size
            else:
                detected_size = 15
                
            # Update board size if different
            if detected_size != self.board_size:
                self.board_size = detected_size
                self.setup_board_canvas()
                
            # Configure MCTS using config values
            config = MCTSConfig(
                game_type=GameType.GOMOKU,
                board_size=self.board_size,
                num_simulations=self.difficulty_levels[self.current_difficulty],
                min_wave_size=getattr(full_config.mcts, 'min_wave_size', 3072),
                max_wave_size=getattr(full_config.mcts, 'max_wave_size', 3072),
                adaptive_wave_sizing=getattr(full_config.mcts, 'adaptive_wave_sizing', False),
                temperature=0.0,  # Deterministic play for consistent user experience
                dirichlet_alpha=getattr(full_config.mcts, 'dirichlet_alpha', 0.3),
                dirichlet_epsilon=0.0,  # Zero epsilon disables noise completely
                device=device,
                use_mixed_precision=getattr(full_config.mcts, 'use_mixed_precision', True),
                use_cuda_graphs=getattr(full_config.mcts, 'use_cuda_graphs', True),
                use_tensor_cores=getattr(full_config.mcts, 'use_tensor_cores', True),
                memory_pool_size_mb=getattr(full_config.mcts, 'memory_pool_size_mb', 2048),
                max_tree_nodes=getattr(full_config.mcts, 'max_tree_nodes', 500000)
            )
            
            # Create MCTS instance
            self.mcts = MCTS(config, self.ai_evaluator)
            self.mcts.optimize_for_hardware()
            
            # Update UI
            model_name = Path(model_filename).name
            config_name = Path(config_filename).name
            self.model_label.config(
                text=f"{model_name} + {config_name} (Board: {self.board_size}x{self.board_size})", 
                foreground="green"
            )
            self.status_var.set(f"Model loaded with config! Board size: {self.board_size}x{self.board_size}")
            
            # Update info
            self.update_info(f"Loaded model: {model_name}\n"
                           f"Config: {config_name}\n"
                           f"Architecture: {network_config.num_res_blocks} blocks, {network_config.num_filters} filters\n"
                           f"Board size: {self.board_size}x{self.board_size}\n"
                           f"Device: {device.upper()}\n"
                           f"Difficulty: {self.current_difficulty} ({self.difficulty_levels[self.current_difficulty]} simulations)")
            
            # Start new game
            self.new_game()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model with config:\n{str(e)}")
            self.status_var.set("Failed to load model with config")
            
    def setup_board_canvas(self):
        """Reconfigure canvas for new board size"""
        canvas_size = 2 * self.margin + (self.board_size - 1) * self.cell_size
        self.canvas.config(width=canvas_size, height=canvas_size)
        
    def on_color_change(self):
        """Handle player color selection change"""
        self.human_color = 0 if self.color_var.get() == "Black" else 1
        if not self.game_over and self.move_history:
            # If game is in progress, ask for confirmation
            if messagebox.askyesno("New Game", "Changing color will start a new game. Continue?"):
                self.new_game()
            else:
                # Revert selection
                self.color_var.set("Black" if self.human_color == 0 else "White")
        else:
            self.new_game()
            
    def on_difficulty_change(self, event=None):
        """Handle difficulty change"""
        self.current_difficulty = self.difficulty_var.get()
        if self.mcts:
            # Update MCTS config
            self.mcts.config.num_simulations = self.difficulty_levels[self.current_difficulty]
            self.update_info(f"Difficulty changed to {self.current_difficulty}\n"
                           f"({self.difficulty_levels[self.current_difficulty]} simulations)")
            
    def new_game(self):
        """Start a new game"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.game_state = alphazero_py.GomokuState() if self.board_size == 15 else None
        self.move_history = []
        self.current_player = 0
        self.game_over = False
        self.last_move = None
        self.ai_evaluation_history = []  # Reset AI evaluation history
        
        if self.mcts:
            self.mcts.reset_tree()
            
        self.draw_board()
        
        # Update status
        if self.ai_evaluator:
            self.status_var.set("New game started! " + 
                              ("Your turn (Black)" if self.human_color == 0 else "AI's turn..."))
            # If AI plays first (black)
            if self.human_color == 1:
                self.root.after(100, self.make_ai_move)
        else:
            self.status_var.set("Load an AI model to start playing")
            
    def draw_board(self):
        """Draw the game board"""
        self.canvas.delete("all")
        
        # Draw grid lines
        for i in range(self.board_size):
            # Vertical lines
            x = self.margin + i * self.cell_size
            self.canvas.create_line(x, self.margin, x, 
                                  self.margin + (self.board_size - 1) * self.cell_size,
                                  fill="black", width=1)
            # Horizontal lines
            y = self.margin + i * self.cell_size
            self.canvas.create_line(self.margin, y,
                                  self.margin + (self.board_size - 1) * self.cell_size, y,
                                  fill="black", width=1)
            
        # Draw star points for standard board sizes
        if self.board_size == 15:
            star_points = [(3, 3), (3, 11), (7, 7), (11, 3), (11, 11)]
        elif self.board_size == 19:
            star_points = [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), 
                          (9, 15), (15, 3), (15, 9), (15, 15)]
        else:
            star_points = []
            
        for row, col in star_points:
            if row < self.board_size and col < self.board_size:
                x = self.margin + col * self.cell_size
                y = self.margin + row * self.cell_size
                self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="black", outline="")
                
        # Draw stones
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row, col] != 0:
                    self.draw_stone(row, col, self.board[row, col])
                    
        # Highlight last move
        if self.last_move:
            row, col = self.last_move
            x = self.margin + col * self.cell_size
            y = self.margin + row * self.cell_size
            color = "white" if self.board[row, col] == 1 else "black"
            self.canvas.create_rectangle(x-5, y-5, x+5, y+5, outline=color, width=2)
            
    def draw_stone(self, row, col, player):
        """Draw a stone on the board"""
        x = self.margin + col * self.cell_size
        y = self.margin + row * self.cell_size
        
        if player == 1:  # Black stone
            self.canvas.create_oval(x - self.stone_radius, y - self.stone_radius,
                                  x + self.stone_radius, y + self.stone_radius,
                                  fill="black", outline="black", width=1)
        else:  # White stone
            self.canvas.create_oval(x - self.stone_radius, y - self.stone_radius,
                                  x + self.stone_radius, y + self.stone_radius,
                                  fill="white", outline="black", width=1)
            
    def get_board_position(self, x, y):
        """Convert canvas coordinates to board position"""
        col = round((x - self.margin) / self.cell_size)
        row = round((y - self.margin) / self.cell_size)
        
        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            return row, col
        return None
        
    def on_board_click(self, event):
        """Handle board click events"""
        if self.game_over or self.ai_thinking or not self.ai_evaluator:
            return
            
        if self.current_player != self.human_color:
            return
            
        pos = self.get_board_position(event.x, event.y)
        if not pos:
            return
            
        row, col = pos
        if self.board[row, col] != 0:
            return
            
        # Make human move
        self.make_move(row, col)
        
        # Check game state
        if not self.game_over:
            # AI's turn
            self.root.after(100, self.make_ai_move)
            
    def make_move(self, row, col):
        """Make a move on the board"""
        if self.board[row, col] != 0:
            return False
            
        # Update board
        self.board[row, col] = 1 if self.current_player == 0 else -1
        self.last_move = (row, col)
        
        # Update game state
        if self.game_state:
            action = row * self.board_size + col
            self.game_state.make_move(action)
            
        # Save to history
        self.move_history.append((row, col, self.current_player))
        
        # Draw the move
        self.draw_stone(row, col, self.board[row, col])
        self.draw_board()
        
        # Check for win
        if self.check_win(row, col):
            winner = "Black" if self.current_player == 0 else "White"
            self.game_over = True
            self.status_var.set(f"{winner} wins!")
            messagebox.showinfo("Game Over", f"{winner} wins!")
            return True
            
        # Check for draw
        if len(self.move_history) >= self.board_size * self.board_size:
            self.game_over = True
            self.status_var.set("Game drawn!")
            messagebox.showinfo("Game Over", "Game drawn!")
            return True
            
        # Switch player
        self.current_player = 1 - self.current_player
        
        # Update status
        if not self.game_over:
            if self.current_player == self.human_color:
                self.status_var.set("Your turn" + (" (Black)" if self.human_color == 0 else " (White)"))
            else:
                self.status_var.set("AI thinking...")
                
        return True
        
    def make_ai_move(self):
        """Make AI move in a separate thread"""
        if self.game_over or not self.ai_evaluator:
            return
            
        self.ai_thinking = True
        self.progress.grid()
        self.progress.start(10)
        
        # Run AI in separate thread
        self.ai_thread = threading.Thread(target=self.ai_think, daemon=True)
        self.ai_thread.start()
        
        # Check for AI result
        self.root.after(100, self.check_ai_result)
        
    def ai_think(self):
        """AI thinking process (runs in separate thread)"""
        try:
            # Run MCTS search and get best action
            if self.game_state:
                best_action = self.mcts.select_action(self.game_state, temperature=0.0)
                
                # Get root value for resignation check
                root_value = self.mcts.get_root_value() if hasattr(self.mcts, 'get_root_value') else 0.0
                
                # Track AI's evaluation from its perspective
                # AI color is opposite to human color
                ai_color = 1 - self.human_color
                # If current player matches AI color, value is already from AI's perspective
                ai_perspective_value = root_value if self.current_player == ai_color else -root_value
                self.ai_evaluation_history.append(ai_perspective_value)
                
                # Keep only recent evaluations
                if len(self.ai_evaluation_history) > self.resign_check_moves:
                    self.ai_evaluation_history = self.ai_evaluation_history[-self.resign_check_moves:]
                
                # Check if AI should resign
                if (len(self.ai_evaluation_history) >= self.resign_check_moves and 
                    len(self.move_history) >= 10):  # Don't resign too early
                    recent_values = self.ai_evaluation_history[-self.resign_check_moves:]
                    if all(v < self.resign_threshold for v in recent_values):
                        # AI resigns
                        self.ai_queue.put("resign")
                        return
                
                # Convert to coordinates
                row = best_action // self.board_size
                col = best_action % self.board_size
                confidence = policy[best_action]
                
                # Reset tree for next search
                self.mcts.reset_tree()
                
                # Put result in queue
                self.ai_queue.put((row, col, confidence, root_value))
            else:
                # Fallback: random valid move
                valid_moves = [(r, c) for r in range(self.board_size) 
                             for c in range(self.board_size) 
                             if self.board[r, c] == 0]
                if valid_moves:
                    import random
                    row, col = random.choice(valid_moves)
                    self.ai_queue.put((row, col, 0.0, 0.0))
                    
        except Exception as e:
            print(f"AI error: {e}")
            self.ai_queue.put(None)
            
    def check_ai_result(self):
        """Check if AI has finished thinking"""
        try:
            result = self.ai_queue.get_nowait()
            
            # Stop progress bar
            self.progress.stop()
            self.progress.grid_remove()
            self.ai_thinking = False
            
            if result == "resign":
                # AI resigns
                self.ai_resign()
            elif result and len(result) >= 3:
                # Normal move result
                row, col, confidence = result[0], result[1], result[2]
                root_value = result[3] if len(result) > 3 else 0.0
                
                self.make_move(row, col)
                
                # Show AI confidence and evaluation
                if confidence > 0:
                    self.update_info(f"AI played at {chr(ord('A') + col)}{self.board_size - row}\n"
                                   f"Confidence: {confidence:.1%}\n"
                                   f"Position evaluation: {root_value:.3f}")
            else:
                self.status_var.set("AI error occurred")
                
        except queue.Empty:
            # AI still thinking
            self.root.after(100, self.check_ai_result)
            
    def check_win(self, row, col):
        """Check if the last move created a winning line"""
        player = self.board[row, col]
        
        # Check all four directions
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            count = 1
            
            # Check positive direction
            r, c = row + dx, col + dy
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                count += 1
                r += dx
                c += dy
                
            # Check negative direction
            r, c = row - dx, col - dy
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                count += 1
                r -= dx
                c -= dy
                
            if count >= 5:
                return True
                
        return False
        
    def undo_move(self):
        """Undo the last move (or last two moves if it's human's turn)"""
        if not self.move_history or self.ai_thinking:
            return
            
        # Determine how many moves to undo
        moves_to_undo = 1
        if self.current_player == self.human_color and len(self.move_history) >= 2:
            moves_to_undo = 2  # Undo both AI and human moves
            
        # Create new game state and replay moves
        self.game_state = alphazero_py.GomokuState() if self.game_state else None
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        
        # Remove moves from history
        for _ in range(min(moves_to_undo, len(self.move_history))):
            self.move_history.pop()
            
        # Replay remaining moves
        temp_history = self.move_history.copy()
        self.move_history = []
        self.current_player = 0
        self.game_over = False
        
        for row, col, player in temp_history:
            self.board[row, col] = 1 if player == 0 else -1
            if self.game_state:
                action = row * self.board_size + col
                self.game_state.make_move(action)
            self.move_history.append((row, col, player))
            self.current_player = 1 - self.current_player
            
        # Update last move
        self.last_move = self.move_history[-1][:2] if self.move_history else None
        
        # Reset AI evaluation history
        self.ai_evaluation_history = []
        
        # Reset MCTS tree
        if self.mcts:
            self.mcts.reset_tree()
            
        # Redraw board
        self.draw_board()
        
        # Update status
        if self.current_player == self.human_color:
            self.status_var.set("Move undone. Your turn.")
        else:
            self.status_var.set("Move undone.")
            self.root.after(100, self.make_ai_move)
            
    def resign_game(self):
        """Handle human player resignation"""
        if self.game_over or not self.ai_evaluator:
            return
            
        # Confirm resignation
        if not messagebox.askyesno("Resign Game", "Are you sure you want to resign?"):
            return
            
        # Set game over
        self.game_over = True
        
        # Determine winner (opponent of current player if it's human's turn, 
        # or opponent of human if it's AI's turn)
        if self.current_player == self.human_color:
            # Human's turn - human resigns
            winner = "White" if self.human_color == 0 else "Black"
            self.status_var.set(f"You resigned. {winner} (AI) wins!")
            messagebox.showinfo("Game Over", f"You resigned. {winner} (AI) wins!")
        else:
            # AI's turn - but human clicked resign, so human still resigns
            winner = "White" if self.human_color == 0 else "Black"  
            self.status_var.set(f"You resigned. {winner} (AI) wins!")
            messagebox.showinfo("Game Over", f"You resigned. {winner} (AI) wins!")
            
        # Stop AI if it's thinking
        if self.ai_thinking:
            self.ai_thinking = False
            self.progress.stop()
            self.progress.grid_remove()
            
        # Update info
        human_color_name = "Black" if self.human_color == 0 else "White"
        ai_color_name = "White" if self.human_color == 0 else "Black"
        self.update_info(f"Game ended by resignation\n"
                        f"{human_color_name} (You) resigned\n"
                        f"{ai_color_name} (AI) wins\n"
                        f"Moves played: {len(self.move_history)}")
                        
    def ai_resign(self):
        """Handle AI resignation"""
        if self.game_over:
            return
            
        # Set game over
        self.game_over = True
        
        # Human wins
        winner = "Black" if self.human_color == 0 else "White"
        self.status_var.set(f"AI resigned. {winner} (You) win!")
        messagebox.showinfo("Game Over", f"AI resigned. {winner} (You) win!")
        
        # Stop AI thinking
        if self.ai_thinking:
            self.ai_thinking = False
            self.progress.stop()
            self.progress.grid_remove()
            
        # Update info
        human_color_name = "Black" if self.human_color == 0 else "White"
        ai_color_name = "White" if self.human_color == 0 else "Black"
        self.update_info(f"Game ended by AI resignation\n"
                        f"{ai_color_name} (AI) resigned\n"
                        f"{human_color_name} (You) win\n"
                        f"Moves played: {len(self.move_history)}\n"
                        f"AI evaluation was too poor")

    def update_info(self, text):
        """Update the info panel"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, text)
        self.info_text.config(state=tk.DISABLED)


def main():
    """Main entry point"""
    root = tk.Tk()
    app = GomokuGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()