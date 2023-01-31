# PRIMA

> This repository contains the source code of PRIMA.
****
# Structure of program
In PRIMA, the code is stored in 8 files:
|File name|Usage|
|---|---
|prefix_sum.py|The code of **SCOPE** and the code of baseline approaches.
|data_preprocess.py|The code for pre-processing of the raw data.
|auto_trip.py|The code of **CLASP**, which contains the **double-sided clipping mechanism** and **SSVT**.
|double_truncated.py|The code for clipping/truncating the data table.
|estimation.py|The code about the maximum entropy model, which is used when constructing a high-dimensional base prefix-sum cube.
|max_entropy.py|The code for computing the base prefix-sum cuboid and constructing the remaining cuboids.
|sum_query.py|The code for answering the multi-dimensional analytical queries.
|utils.py|The utilities that are used in other code.
# Get started with PRIMA
For starting the program, you should first set the parameters in .sh files and then run .sh files.

