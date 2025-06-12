#!/usr/bin/env python3
"""Fix game states indexing issue"""

# Read the file
with open('/home/cosmo/omoknuni_quantum/python/mcts/gpu/gpu_game_states.py', 'r') as f:
    content = f.read()

# Fix the indexing issue by converting to long
old_code = """            # Store in full move history at current move count position
            move_counts = self.move_count[state_indices]
            # Use state_indices directly for first dimension, move_counts for second
            self.full_move_history[state_indices, move_counts] = actions.to(torch.int16)"""

new_code = """            # Store in full move history at current move count position
            move_counts = self.move_count[state_indices].long()
            # Use state_indices directly for first dimension, move_counts for second
            self.full_move_history[state_indices.long(), move_counts] = actions.to(torch.int16)"""

content = content.replace(old_code, new_code)

# Write back
with open('/home/cosmo/omoknuni_quantum/python/mcts/gpu/gpu_game_states.py', 'w') as f:
    f.write(content)

print("Fixed indexing issue in gpu_game_states.py")