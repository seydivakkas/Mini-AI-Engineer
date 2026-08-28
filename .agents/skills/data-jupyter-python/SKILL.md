---
name: data-jupyter-python
description: "Master interactive data science, exploratory data analysis (EDA), and machine learning experimentation in Jupyter with Python: IPython magics (%timeit, %prun, %debug), interactive widgets (ipywidgets), automated batch execution (papermill), headless report exports (nbconvert), and memory/performance profiling."
risk: unknown
source: community
date_added: '2026-02-28'
---

# Data & Machine Learning with Jupyter and Python

Masterclass for interactive exploratory data analysis (EDA), notebook performance profiling, automated workflow execution, and interactive visualization in Jupyter environments.

## When to Use This Skill

Use this skill when:
- Conducting exploratory data analysis (EDA), statistical profiling, and hypothesis testing in Jupyter notebooks
- Profiling cell execution time, line-by-line bottlenecks, and memory consumption (`%timeit`, `%prun`, `%lprun`, `%memit`)
- Debugging unexpected runtime errors or exceptions interactively with post-mortem debugging (`%debug`, `%pdb`)
- Building interactive parameter sweep tools, sliders, and exploratory dashboards with `ipywidgets`
- Automating parameterized notebook execution in headless CI/CD data pipelines with `papermill`
- Exporting production research reports, HTML dashboards, or clean slides using `nbconvert`
- Structuring clean, reproducible notebooks with modular cell decomposition, environment versioning, and restart-and-run-all reliability

---

## Core Capabilities & Jupyter Mastery

### 1. Essential IPython Magic Commands

```python
# 1. Precise execution timing (statistical runs)
%timeit -n 100 -r 5 np.dot(matrix_a, matrix_b)

# 2. Line-by-line CPU Profiling
%prun -s cumulative train_step(model, batch)

# 3. Interactive Post-Mortem Debugging on Exception
%debug

# 4. Workspace Variable Inspection
%whos
```

---

### 2. Interactive Data Exploration with `ipywidgets`

```python
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np

@widgets.interact(
    kernel_size=widgets.IntSlider(min=3, max=21, step=2, value=5),
    sigma=widgets.FloatSlider(min=0.1, max=5.0, step=0.1, value=1.0)
)
def explore_gaussian_filtering(kernel_size: int, sigma: float):
    # Interactive visual adjustment without re-running entire notebook
    plt.figure(figsize=(6, 4))
    plt.title(f"Gaussian Kernel ({kernel_size}x{kernel_size}, sigma={sigma:.1f})")
    # Rendering logic...
    plt.show()
```

---

### 3. Headless Batch Automation with Papermill

Execute notebooks programmatically with injected hyperparameter dictionaries:

```bash
papermill \
  notebooks/model_evaluation_template.ipynb \
  ciktilar/eval_run_epoch_50.ipynb \
  -p learning_rate 0.001 \
  -p batch_size 64 \
  -p dataset_version "v2.1"
```

---

### 4. Automated Report & Document Export (`nbconvert`)

```bash
# Export clean HTML report without code input cells (for executive stakeholder review)
jupyter nbconvert --to html --no-input notebooks/customer_segmentation_analysis.ipynb

# Export PDF publication report
jupyter nbconvert --to pdf notebooks/deep_learning_benchmark.ipynb
```

---

## Best Practices & Engineering Rules

1. **Top-to-Bottom Execution Guarantee:** Always test notebooks with "Restart Kernel and Run All Cells" before committing to version control.
2. **Never Commit Large Binary Outputs / Datasets:** Clear bulky image/dataframe outputs or strip output cells using `nbstripout` before Git commits.
3. **Modular Cell Logic:** Keep cells focused (5-25 lines per cell). Separate data ingestion, preprocessing, modeling, and visualization into distinct labeled cells.
4. **Environment Specification:** Always specify Python environment dependencies at the top markdown cell with explicit package versioning.
