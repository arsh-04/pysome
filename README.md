# K-Means Clustering Project

## Overview
This project implements the K-Means clustering algorithm from scratch in Python. The goal is to cluster a given dataset into two and three groups using the K-Means algorithm and visualize the results.

## Dataset
The dataset used in this project is provided in the assignment and can be accessed [here](https://drive.google.com/file/d/1xHNXEdnyEcxMqlk4lA1JqVjKFr3sdV2Q/view?usp=drive_link).

## Requirements
To run this project, you need the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`

These libraries can be installed using the following command:
```bash
pip install numpy pandas matplotlib
```

## Implementation Details
- The K-Means algorithm is implemented from scratch without using libraries such as `scikit-learn`, `TensorFlow`, or `PyTorch`.
- The algorithm is tested for `k=2` and `k=3`.
- The dataset is normalized before applying clustering.
- Results are visualized using scatter plots with different colors representing different clusters.

## Usage
Run the Python script to perform clustering and visualize the results:
```bash
python kmeans_clustering.py
```

## Results
Two plots are generated:
1. Clustering result for `k=2`
2. Clustering result for `k=3`

Each plot shows the dataset with different colors representing distinct clusters.

## Notes
- Ensure that the dataset is correctly loaded before running the script.
- Label the plots properly for better understanding.

## Author
Arsh Aditey


---
This README file provides a structured overview of the project, guiding users on setup, execution, and results visualization.
