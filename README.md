# FairFace Audit Pipeline

This project performs a fairness audit of the [FairFace](https://github.com/dchen236/FairFace) facial recognition model. Given a labeled `.xlsx` file of human-validated facial attributes (age, gender, race), the pipeline:

1. Converts the input into a CSV for prediction
2. Runs FairFace detection and classification
3. Merges predictions with ground truth
4. Scores the predictions using a structured fairness rubric
5. Outputs an audit report in `.xlsx` format

---

## 📆 Environment Setup

To ensure all dependencies (e.g., `torch`, `dlib`, `pandas`, `openpyxl`) are installed correctly, use the provided Conda environment file:

### Step 1: Create the environment

```bash
conda env create -f environment.yaml
```

### Step 2: Activate it

```bash
conda activate fairface
```

---

## 🚀 How to Run the Pipeline

The entire pipeline is contained in `audit.py`. To run it:

```bash
python audit.py
```

This script will:

- Convert your input Excel file (e.g., `provided_labels.xlsx` or `extra_credit_labels.xlsx`)
- Run FairFace predictions
- Merge and score the results
- Save the final audit file as `scored_audit_results.xlsx` 

### Input Format

Ensure your input Excel file has:

- 3 metadata/header rows (the actual data starts from row 4)
- A column labeled `file` that contains image paths like `val/123.jpg`

---

## 📊 Scoring Rubric

Each image is scored out of **10** based on how well FairFace's predictions match the human-validated labels.

| Attribute  | Max Points | Deduction Rule                                                      |
| ---------- | ---------- | ------------------------------------------------------------------- |
| **Race**   | 4 pts      | ✅ 0 if match, ❌ -4 if mismatch                                      |
| **Gender** | 2 pts      | ✅ 0 if match, ❌ -2 if mismatch                                      |
| **Age**    | 4 pts      | ✅ 0 if exact match⚠️ -2 if off by 1 bucket❌ -4 if off by 2+ buckets |

### Age Buckets

```
["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
```

### Special Handling

- `more than 70` → interpreted as `70+`
- If any prediction is missing → score is **0**, note: *"Missing FairFace prediction(s)"*

### Output Example

| Score | Notes                                            |
| ----- | ------------------------------------------------ |
| 10    | Gender match; Race match; Age match              |
| 6     | Gender match; Race mismatch; Age off by 1 bucket |
| 0     | Missing FairFace prediction(s)                   |

---

## 📁 Output Files

- `standard_labels.csv` — converted input for FairFace
- `test_outputs.csv` — raw FairFace predictions
- `scored_audit_results.xlsx` — final audit file with quality scores and notes

---

## 🧑‍💻 Author

Built by **Krithish Ayyappan** for fairness auditing of facial recognition systems.

