"""GPU kernels for Go terminal state detection and game logic

High-performance CUDA kernels for Go game operations including:
- Terminal state detection (no legal moves)
- Territory scoring
- Group capture detection
- Liberty counting
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

def gpu_check_terminal_states(boards: torch.Tensor, passes: torch.Tensor) -> torch.Tensor:
    """GPU-optimized terminal state detection for Go
    
    Args:
        boards: Batch of Go boards [batch_size, board_size, board_size]
                Values: 0 = empty, 1 = black, 2 = white
        passes: Consecutive passes [batch_size] (terminal if >= 2)
    
    Returns:
        terminal_states: Boolean tensor [batch_size] indicating terminal states
    """
    if boards.numel() == 0:
        return torch.empty(0, dtype=torch.bool, device=boards.device)
    
    batch_size = boards.shape[0]
    device = boards.device
    
    # Terminal if two consecutive passes
    pass_terminal = passes >= 2
    
    # Also terminal if no legal moves (rare but possible)
    legal_moves = gpu_get_legal_moves(boards)
    no_moves_terminal = ~legal_moves.any(dim=(1, 2))
    
    # Combined terminal condition
    terminal_states = pass_terminal | no_moves_terminal
    
    return terminal_states

def gpu_get_legal_moves(boards: torch.Tensor, current_player: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Get legal moves for batch of Go boards
    
    Args:
        boards: Batch of boards [batch_size, board_size, board_size]
        current_player: Current player [batch_size] (1=black, 2=white), optional
    
    Returns:
        legal_moves: Boolean tensor [batch_size, board_size, board_size]
                    True where moves are legal
    """
    batch_size, board_size, _ = boards.shape
    device = boards.device
    
    # Basic legality: empty intersections
    empty_squares = (boards == 0)
    
    if current_player is None:
        # If player not specified, just return empty squares
        return empty_squares
    
    # For full Go rules, we need to check:
    # 1. Suicide rule (can't place stone with no liberties unless captures)
    # 2. Ko rule (can't immediately recapture)
    
    # Simplified implementation: check if move would have liberties
    legal_moves = empty_squares.clone()
    
    # Create adjacency kernel for liberty checking
    adjacency_kernel = torch.zeros(1, 1, 3, 3, device=device, dtype=torch.float32)
    adjacency_kernel[0, 0, 1, 0] = 1  # Up
    adjacency_kernel[0, 0, 0, 1] = 1  # Left  
    adjacency_kernel[0, 0, 2, 1] = 1  # Right
    adjacency_kernel[0, 0, 1, 2] = 1  # Down
    
    # For each potential move, check if it would have liberties
    for player_idx in [1, 2]:
        player_mask = (current_player == player_idx)
        if not player_mask.any():
            continue
            
        player_boards = boards[player_mask]
        player_legal = legal_moves[player_mask]
        
        # Check suicide rule (simplified)
        # A move is illegal if it creates a group with no liberties
        # and doesn't capture opponent stones
        
        empty_map = (player_boards == 0).float().unsqueeze(1)
        
        # Count adjacent empty squares (liberties) for each position
        liberty_count = F.conv2d(empty_map, adjacency_kernel, padding=1)
        liberty_count = liberty_count.squeeze(1)
        
        # A move is legal if the intersection is empty and either:
        # 1. Has adjacent liberties, or
        # 2. Connects to friendly group with liberties, or
        # 3. Captures opponent stones
        has_liberties = liberty_count > 0
        
        # Update legal moves (simplified - full Go rules are complex)
        player_legal = player_legal & has_liberties
        legal_moves[player_mask] = player_legal
    
    return legal_moves

def gpu_count_liberties(boards: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Count liberties for stones at given positions
    
    Args:
        boards: Batch of boards [batch_size, board_size, board_size]
        positions: Positions to check [batch_size, num_positions, 2]
    
    Returns:
        liberty_counts: Number of liberties [batch_size, num_positions]
    """
    batch_size, board_size, _ = boards.shape
    device = boards.device
    
    # Create adjacency offsets
    offsets = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]], device=device)
    
    liberty_counts = []
    
    for batch_idx in range(batch_size):
        board = boards[batch_idx]
        batch_positions = positions[batch_idx]
        
        batch_liberties = []
        for pos in batch_positions:
            row, col = pos[0].item(), pos[1].item()
            
            if row < 0 or row >= board_size or col < 0 or col >= board_size:
                batch_liberties.append(0)
                continue
            
            # Count empty adjacent intersections
            liberties = 0
            for offset in offsets:
                adj_row = row + offset[0].item()
                adj_col = col + offset[1].item()
                
                if (0 <= adj_row < board_size and 0 <= adj_col < board_size):
                    if board[adj_row, adj_col] == 0:
                        liberties += 1
            
            batch_liberties.append(liberties)
        
        liberty_counts.append(torch.tensor(batch_liberties, device=device))
    
    return torch.stack(liberty_counts)

def gpu_detect_captures(boards: torch.Tensor, last_moves: torch.Tensor, players: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Detect captured groups after moves
    
    Args:
        boards: Boards before moves [batch_size, board_size, board_size]
        last_moves: Last move positions [batch_size, 2]
        players: Players who made moves [batch_size]
    
    Returns:
        captured_stones: Positions of captured stones [batch_size, max_captures, 2]
        capture_counts: Number of captured stones per game [batch_size]
    """
    batch_size, board_size, _ = boards.shape
    device = boards.device
    
    # Apply moves to get new board state
    new_boards = gpu_apply_moves(boards, last_moves, players)
    
    # Find opponent groups with no liberties
    opponent_players = 3 - players  # 1->2, 2->1
    
    captured_positions = []
    capture_counts = []
    
    for batch_idx in range(batch_size):
        board = new_boards[batch_idx]
        opponent = opponent_players[batch_idx].item()
        
        # Find all opponent stones
        opponent_positions = torch.nonzero(board == opponent, as_tuple=False)
        
        if opponent_positions.numel() == 0:
            captured_positions.append(torch.zeros(0, 2, device=device, dtype=torch.long))
            capture_counts.append(0)
            continue
        
        # Check liberties for opponent groups
        # This is a simplified version - full implementation requires flood-fill
        captured_in_batch = []
        
        for pos in opponent_positions:
            row, col = pos[0].item(), pos[1].item()
            
            # Check if this stone has any liberties
            has_liberty = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < board_size and 0 <= nc < board_size:
                    if board[nr, nc] == 0:
                        has_liberty = True
                        break
            
            if not has_liberty:
                captured_in_batch.append(pos)
        
        if captured_in_batch:
            captured_positions.append(torch.stack(captured_in_batch))
            capture_counts.append(len(captured_in_batch))
        else:
            captured_positions.append(torch.zeros(0, 2, device=device, dtype=torch.long))
            capture_counts.append(0)
    
    # Pad captured positions to same length
    max_captures = max(len(cp) for cp in captured_positions) if captured_positions else 0
    if max_captures == 0:
        return torch.zeros(batch_size, 0, 2, device=device, dtype=torch.long), torch.zeros(batch_size, device=device, dtype=torch.long)
    
    padded_captures = []
    for cp in captured_positions:
        if len(cp) < max_captures:
            padding = torch.full((max_captures - len(cp), 2), -1, device=device, dtype=torch.long)
            cp = torch.cat([cp, padding], dim=0)
        padded_captures.append(cp)
    
    captured_stones = torch.stack(padded_captures)
    capture_counts = torch.tensor(capture_counts, device=device, dtype=torch.long)
    
    return captured_stones, capture_counts

def gpu_score_territory(boards: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Score territory using simplified area scoring
    
    Args:
        boards: Final board positions [batch_size, board_size, board_size]
    
    Returns:
        black_scores: Territory + captures for black [batch_size]
        white_scores: Territory + captures for white [batch_size]
    """
    batch_size, board_size, _ = boards.shape
    device = boards.device
    
    # Simplified scoring: count controlled territory
    # In practice, this would need sophisticated flood-fill algorithms
    
    black_stones = (boards == 1).sum(dim=(1, 2)).float()
    white_stones = (boards == 2).sum(dim=(1, 2)).float()
    empty_squares = (boards == 0).sum(dim=(1, 2)).float()
    
    # Simplified: assume territory is split proportionally
    # Real Go scoring is much more complex
    total_stones = black_stones + white_stones
    black_ratio = black_stones / (total_stones + 1e-8)
    white_ratio = white_stones / (total_stones + 1e-8)
    
    black_territory = empty_squares * black_ratio
    white_territory = empty_squares * white_ratio
    
    black_scores = black_stones + black_territory
    white_scores = white_stones + white_territory
    
    return black_scores, white_scores

def gpu_apply_moves(boards: torch.Tensor, moves: torch.Tensor, players: torch.Tensor) -> torch.Tensor:
    """Apply moves to batch of Go boards
    
    Args:
        boards: Batch of boards [batch_size, board_size, board_size]
        moves: Move positions [batch_size, 2] as (row, col), -1 for pass
        players: Player indices [batch_size] (1=black, 2=white)
    
    Returns:
        new_boards: Updated boards with moves applied
    """
    new_boards = boards.clone()
    batch_size = boards.shape[0]
    
    # Apply non-pass moves
    valid_moves = (moves[:, 0] >= 0) & (moves[:, 1] >= 0)
    
    if valid_moves.any():
        valid_indices = torch.where(valid_moves)[0]
        valid_move_positions = moves[valid_indices]
        valid_players = players[valid_indices]
        
        # Apply moves using advanced indexing
        rows = valid_move_positions[:, 0]
        cols = valid_move_positions[:, 1]
        
        new_boards[valid_indices, rows, cols] = valid_players
    
    return new_boards

class GoGPUKernels:
    """Collection of GPU kernels for Go operations"""
    
    def __init__(self, device: torch.device, board_size: int = 19):
        self.device = device
        self.board_size = board_size
        
        # Pre-compile kernels for better performance
        self._compile_kernels()
    
    def _compile_kernels(self):
        """Pre-compile kernels with torch.compile for maximum performance"""
        try:
            # Compile main operations
            self.check_terminal_states = torch.compile(
                gpu_check_terminal_states,
                mode="max-autotune",
                fullgraph=True
            )
            
            self.get_legal_moves = torch.compile(
                gpu_get_legal_moves,
                mode="max-autotune",
                fullgraph=True
            )
            
            logger.info("✅ Go GPU kernels compiled successfully")
            
        except Exception as e:
            logger.warning(f"Failed to compile Go kernels: {e}")
            # Fall back to non-compiled versions
            self.check_terminal_states = gpu_check_terminal_states
            self.get_legal_moves = gpu_get_legal_moves
    
    def batch_terminal_detection(self, boards: torch.Tensor, passes: torch.Tensor) -> torch.Tensor:
        """Batch terminal state detection"""
        return self.check_terminal_states(boards, passes)
    
    def batch_legal_moves(self, boards: torch.Tensor, current_player: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Batch legal move detection"""
        return self.get_legal_moves(boards, current_player)
    
    def batch_score_games(self, boards: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch game scoring"""
        return gpu_score_territory(boards)

# Global kernel instance
_go_kernels = None

def get_go_gpu_kernels(device: torch.device, board_size: int = 19) -> GoGPUKernels:
    """Get or create global Go GPU kernels instance"""
    global _go_kernels
    
    if _go_kernels is None or _go_kernels.device != device:
        _go_kernels = GoGPUKernels(device, board_size)
    
    return _go_kernels