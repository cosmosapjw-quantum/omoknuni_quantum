#!/usr/bin/env python3
"""Fix all dtype issues comprehensively"""

import re

# Fix optimized_mcts.py
print("Fixing optimized_mcts.py...")
with open('/home/cosmo/omoknuni_quantum/python/mcts/core/optimized_mcts.py', 'r') as f:
    content = f.read()

# Fix node_to_state assignment
content = re.sub(
    r'self\.node_to_state\[non_root_nodes\] = non_root_new_states',
    'self.node_to_state[non_root_nodes] = non_root_new_states.int()',
    content
)

# Fix any other potential int32/int64 mismatches
content = re.sub(
    r'self\.node_to_state\[([^\]]+)\] = ([^\n]+)$',
    lambda m: f'self.node_to_state[{m.group(1)}] = ({m.group(2)}).int()',
    content,
    flags=re.MULTILINE
)

with open('/home/cosmo/omoknuni_quantum/python/mcts/core/optimized_mcts.py', 'w') as f:
    f.write(content)

print("Fixed dtype issues in optimized_mcts.py")

# Also ensure state indices are int32 when allocated
print("\nFixing _allocate_states return type...")
with open('/home/cosmo/omoknuni_quantum/python/mcts/core/optimized_mcts.py', 'r') as f:
    content = f.read()

# Fix the return type
content = re.sub(
    r'return allocated$',
    'return allocated.int()',
    content,
    flags=re.MULTILINE
)

with open('/home/cosmo/omoknuni_quantum/python/mcts/core/optimized_mcts.py', 'w') as f:
    f.write(content)

print("All dtype fixes applied!")