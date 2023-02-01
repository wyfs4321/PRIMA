# PRIMA

> This repository contains the source code of PRIMA.
****

# Environment
We implement all the approaches in Python v3.6.7. All experiments are conducted on an Intel Core i5 2.50GHz PC with 8GB RAM.

# Structure of program
In PRIMA, the code is stored in 9 files:
|File name|Usage|
|---|---
|prima.py|The main function of **PRIMA**.
|data_preprocessing.py|The code for pre-processing the data table.
|clasp.py|The code of **CLASP**, which contains the **double-sided clipping mechanism** and **SSVT**.
|data_rewriting.py|The code for rewriting (clipping or truncating) the data table.
|scope.py|The code of **SCOPE**.
|distribution_computing.py|The code for computing the distribution of high-dimensional noisy sum vector.
|cuboid_constructing.py|The code for constructing the base prefix-sum cuboid.
|sum_query_processing.py|The code for answering the multi-dimensional analytical queries.
|utils.py|The utilities that are used in other code.
# Get started with PRIMA
1. For starting the program, you can run .sh files as follows.
```bash
./run_prima.sh
```
2. You can change the parameters of PRIMA in the experiments by modifying the .sh files.

