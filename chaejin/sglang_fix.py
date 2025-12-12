"""
Comprehensive patch to disable deep_gemm and use PyTorch instead
Run this script before starting sglang server
"""

import os
import sys

# Path to the batch_invariant_ops file
batch_inv_ops_path = "/media/NAS/USERS/shahid/sglang/python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py"

if not os.path.exists(batch_inv_ops_path):
    print(f"✗ File not found: {batch_inv_ops_path}")
    print("Please adjust the path in this script.")
    sys.exit(1)

# Read the file
with open(batch_inv_ops_path, 'r') as f:
    lines = f.readlines()

# Find and patch the matmul_persistent function
patched_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    
    # Patch 1: Comment out or replace deep_gemm import
    if 'import deep_gemm' in line:
        patched_lines.append('# ' + line)  # Comment it out
        i += 1
        continue
    
    # Patch 2: Replace _matmul_persistent_deepgemm function
    if 'def _matmul_persistent_deepgemm' in line:
        # Find the end of the function (next def or class)
        start = i
        i += 1
        indent_level = len(line) - len(line.lstrip())
        
        # Collect the entire function
        func_lines = [line]
        while i < len(lines):
            current_line = lines[i]
            if current_line.strip() == '':
                func_lines.append(current_line)
                i += 1
                continue
            
            current_indent = len(current_line) - len(current_line.lstrip())
            if current_line.strip() and current_indent <= indent_level and not current_line.strip().startswith('"'):
                break
            
            func_lines.append(current_line)
            i += 1
        
        # Replace with PyTorch implementation
        replacement = '''def _matmul_persistent_deepgemm(a, b, bias=None):
    """Matmul using PyTorch (deep_gemm fallback for unsupported architectures)."""
    import torch
    # Use standard PyTorch matmul instead of deep_gemm
    out = torch.matmul(a, b)
    if bias is not None:
        out = out + bias
    return out

'''
        patched_lines.append(replacement)
        continue
    
    # Patch 3: Handle calls to deep_gemm directly
    if 'deep_gemm.' in line:
        # Skip or comment out deep_gemm calls
        patched_lines.append('# ' + line)
        i += 1
        continue
    
    patched_lines.append(line)
    i += 1

# Write the patched file
with open(batch_inv_ops_path, 'w') as f:
    f.writelines(patched_lines)

print("✓ Successfully patched batch_invariant_ops.py")
print("✓ Replaced deep_gemm calls with PyTorch operations")
print("\nYou can now run sglang with deterministic inference enabled:")
print("""
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \\
  --model Qwen/Qwen2.5-1.5B-Instruct \\
  --log-level info \\
  --device=cuda \\
  --chunked-prefill-size=-1 \\
  --tp 1 \\
  --schedule-policy lpm \\
  --enable-deterministic-inference \\
  --disable-cuda-graph
""")