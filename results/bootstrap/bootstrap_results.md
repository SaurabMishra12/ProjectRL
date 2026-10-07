# Paired Question-Clustered Bootstrap Analysis

## Accuracy Difference CIs (95% BCa, B=10,000)

| Dataset | Comparison | n_paired | Point Δ | 95% BCa CI | Interpretation |
|---------|-----------|----------|---------|------------|---------------|
| GSM8K | SparseComm − MAD | 551 | -1.09% | [-3.81%, +1.27%] | ✓ Non-inferior |
| GSM8K | SparseComm − NoComm | 551 | +8.53% | [+4.90%, +11.80%] | ✓ Non-inferior |
| AQUA | SparseComm − MAD | 254 | -0.39% | [-2.64%, +0.39%] | ✓ Non-inferior |
| AQUA | SparseComm − NoComm | 254 | +3.94% | [+1.57%, +6.69%] | ✓ Non-inferior |
| HOTPOTQA | SparseComm − MAD | 863 | -2.43% | [-4.17%, -1.04%] | ✓ Non-inferior |
| HOTPOTQA | SparseComm − NoComm | 863 | +1.51% | [-0.35%, +3.24%] | ✓ Non-inferior |

## TOST Non-Inferiority p-values

| Dataset | Comparison | δ=3pp | δ=5pp | δ=10pp |
|---------|-----------|-------|-------|--------|
| GSM8K | SparseComm − MAD | p=0.0692 | p=0.0017 | p=0.0000 |
| GSM8K | SparseComm − NoComm | p=0.0000 | p=0.0000 | p=0.0000 |
| AQUA | SparseComm − MAD | p=0.0008 | p=0.0000 | p=0.0000 |
| AQUA | SparseComm − NoComm | p=0.0000 | p=0.0000 | p=0.0000 |
| HOTPOTQA | SparseComm − MAD | p=0.2473 | p=0.0012 | p=0.0000 |
| HOTPOTQA | SparseComm − NoComm | p=0.0000 | p=0.0000 | p=0.0000 |

## Method-Level Accuracy (from matched questions)

| Dataset | Method | Accuracy | n |
|---------|--------|----------|---|
| GSM8K | SparseComm | 94.56% | 551 |
| GSM8K | MAD | 95.64% | 551 |
| AQUA | SparseComm | 86.61% | 254 |
| AQUA | MAD | 87.01% | 254 |
| HOTPOTQA | SparseComm | 31.29% | 863 |
| HOTPOTQA | MAD | 33.72% | 863 |