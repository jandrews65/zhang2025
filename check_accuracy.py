"""
Quick Check: What is the True Mean Accuracy?
=============================================

This tells you which number to use in your paper.
"""

import pandas as pd
from pathlib import Path

print("="*60)
print("ACCURACY VERIFICATION")
print("="*60)

# Load fold results
fold_file = Path('results/fold_results.csv')

if not fold_file.exists():
    print("\n❌ ERROR: results/fold_results.csv not found!")
    print("   Run zhang_step3_ml.py first")
    exit(1)

fold_results = pd.read_csv(fold_file)

# Calculate mean and std
mean_acc = fold_results['accuracy'].mean()
std_acc = fold_results['accuracy'].std()

print(f"\n✓ Loaded {len(fold_results)} folds")
print(f"\nPer-fold accuracies:")
for i, acc in enumerate(fold_results['accuracy'], 1):
    print(f"  Fold {i}: {acc*100:.2f}%")

print(f"\n{'='*60}")
print("CORRECT VALUES FOR YOUR PAPER:")
print("="*60)
print(f"\nMean Accuracy: {mean_acc*100:.1f}% ± {std_acc*100:.1f}%")
print(f"\n✅ USE THIS IN:")
print(f"   - Abstract: {mean_acc*100:.1f}%")
print(f"   - Table 1: {mean_acc*100:.1f}% ± {std_acc*100:.1f}%")
print(f"   - All other mentions: {mean_acc*100:.1f}%")

# Check which needs fixing
if abs(mean_acc * 100 - 50.2) < 0.1:
    print(f"\n📝 FIX NEEDED:")
    print(f"   Your abstract says 49.5% but should be 50.2%")
    print(f"   Change abstract to match Table 1")
elif abs(mean_acc * 100 - 49.5) < 0.1:
    print(f"\n📝 FIX NEEDED:")
    print(f"   Your Table 1 says 50.2% but should be 49.5%")
    print(f"   Change Table 1 to match abstract")
else:
    print(f"\n⚠️  BOTH ARE WRONG!")
    print(f"   Abstract (49.5%) and Table 1 (50.2%) should both be {mean_acc*100:.1f}%")

print("\n" + "="*60)
