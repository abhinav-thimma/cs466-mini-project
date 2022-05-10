# CS466: Mini Project: Gibbs Sampling 

This repository contains code for running a multi-threaded Gibbs Sampling Motif finding algorithm.

## Submission by:
1. Abhinav Reddy Thimma: (```athimma2```)
2. Nikhitha Reddeddy (```nr25```)

## Instructions to run:

There are 3 seperate .py files which can be run as follows:\
1. Generating random datasets and plugging specific motifs into them

    > python3 project_benchmark_generation.py

2. Running Gibbs sampling algorithm on the generated datasets:
    > python3 project_gibbs_Sampling.py

3. Evaluating the results of Gibbs sampling using the metrics (Information content per column, Relative Entropy, Overlapping positions, Average run-time)
    > python3 project_evaluation.py

## Results:
The results from the above three steps would be captured in a file named ``` results.csv ``` generated in the root folder of the project