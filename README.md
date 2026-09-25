# LLM Loan Underwriting Audit- UNC DATA709 course project

This project evaluates how a large language model behaves as a loan underwriter on the German Credit dataset. It loads applicant records, samples a subset, sends applications to an Azure OpenAI model, compares the LLM decisions with ground-truth labels, and benchmarks the results against a Logistic Regression baseline.

The project is designed for quick experimentation and auditing of model behavior, output quality, and fairness-related patterns.

## Run the project

### 1) Full run

```bash
python main.py
```

This runs the full pipeline:
- load data
- sample records
- call the LLM
- evaluate results against ground truth
- compare with the logistic regression baseline
- save outputs to the `output/` folder

### 2) Debug run with a small sample

```bash
python main.py --dry-run 2
```

This only processes the first 2 records for quick testing.

### 3) Skip the baseline model

```bash
python main.py --skip-baseline
```

This skips the Logistic Regression comparison and keeps only the LLM evaluation.

### 4) Reuse an existing checkpoint

```bash
python main.py --skip-sample
```

This skips sampling and evaluates using the current stored checkpoint data.

### 5) Scan thresholds

```bash
python main.py --scan-thresholds
```

This scans multiple decision thresholds to inspect accuracy, F1, and cost-sensitive trade-offs.

## Main outputs

The program writes results to the `output/` directory:

- `output/results_with_llm.json`
- `output/ground_truth_eval.json`
- `output/summary_table.csv`
- `output/baseline_gaps.json`
- `output/llm_checkpoint.jsonl`

## Project notes

- The model endpoint and configuration are defined in `config.py`.
- The main entry point is `main.py`.
- The project uses the German Credit dataset and focuses on decision audit and bias analysis.

## Authors

This project was developed by:

- Zhaohan Hou
- Jiaxin Wang
- Jingxia Jiang
