import os
import sys

batch_inv_ops_path = "/media/NAS/USERS/shahid/sglang/python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py"

if not os.path.exists(batch_inv_ops_path):
    print(f"✗ File not found: {batch_inv_ops_path}")
    sys.exit(1)

with open(batch_inv_ops_path, 'r') as f:
    content = f.read()

# Replace the problematic function
old_pattern = "def _matmul_persistent_deepgemm"
if old_pattern in content:
    # Find and replace the entire function
    start_idx = content.find(old_pattern)
    next_def_idx = content.find("\ndef ", start_idx + 1)
    if next_def_idx == -1:
        next_def_idx = len(content)
    
    replacement = '''def _matmul_persistent_deepgemm(a, b, bias=None):
    """Matmul using PyTorch instead of deep_gemm."""
    import torch
    out = torch.matmul(a, b)
    if bias is not None:
        out = out + bias
    return out

'''
    
    content = content[:start_idx] + replacement + content[next_def_idx:]
    
    with open(batch_inv_ops_path, 'w') as f:
        f.write(content)
    
    print("✓ Patched successfully!")
else:
    print("✗ Could not find function to patch")
    sys.exit(1)