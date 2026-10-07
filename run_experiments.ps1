$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING='utf-8'

Write-Host "Running GSM8K..."
python main.py --dataset gsm8k --mode full --n-samples 20 --llm azure --results-dir 08-06results

Write-Host "Running HotpotQA..."
python main.py --dataset hotpotqa --mode full --n-samples 20 --llm azure --results-dir 08-06results

Write-Host "Running MATH..."
python main.py --dataset math --mode full --n-samples 20 --llm azure --results-dir 08-06results

Write-Host "All experiments completed!"
