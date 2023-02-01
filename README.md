# PRIMA

> This repository contains the source code of PRIMA.
****
# Structure of program
In PRIMA, the code is stored in 9 files:
|File name|Usage|
|---|---
|prima.py|The main function of PRIMA.
|clasp.py|The code of **CLASP**, which contains the **double-sided clipping mechanism** and **SSVT**.
|predicting.py|The code of **SCOPE**.
|data_preprocessing.py|The code for pre-processing the raw data.
|data_rewriting.py|The code for rewriting (clipping or truncating) the data table.
|distribution_computing.py|The code for computing the distribution of high-dimensional noisy sum vector.
|cuboid_constructing.py|The code for constructing the base prefix-sum cuboid.
|sum_query_processing.py|The code for answering the multi-dimensional analytical queries.
|utils.py|The utilities that are used in other code.
# Get started with PRIMA
For starting the program, you should first set the parameters in .sh files and then run .sh files.

