# CSK Performance Analysis Using K-Means Clustering

## Project overview
This Python project explores **200 synthetic CSK-themed cricket match records** using statistical analysis and K-Means clustering. It groups match records using five numerical performance features: CSK runs, opponent runs, CSK wickets, opponent wickets, and CSK sixes.

> **Dataset note:** The supplied file is named `CSK_200_Raw_Synthetic.csv`. The records are synthetic and should not be represented as verified official IPL match data.

## Objectives
- Explore and summarize cricket-themed match data.
- Handle missing numerical values using median imputation.
- Cap outliers using the interquartile range (IQR) rule.
- Standardize numerical features before distance-based clustering.
- Apply K-Means and select a cluster count using silhouette scores.
- Evaluate and visualize the clusters, including a PCA-based 2D view.

## Dataset
The raw dataset contains **200 rows** and **7 columns**.

Columns:
- `season`
- `opponent`
- `csk_runs`
- `opponent_runs`
- `csk_wickets`
- `opponent_wickets`
- `csk_sixes`

The five features used by K-Means are:
`csk_runs`, `opponent_runs`, `csk_wickets`, `opponent_wickets`, and `csk_sixes`.

`season` and `opponent` remain in the dataset but are not used as clustering inputs in the supplied script.

## Workflow
1. Load the raw CSV with Pandas.
2. Inspect the dataset shape, data types, and first records.
3. Fill missing values in the five numerical features with their respective medians.
4. Cap values outside the IQR limits (`Q1 - 1.5 × IQR` and `Q3 + 1.5 × IQR`).
5. Standardize the five features using `StandardScaler`.
6. Calculate descriptive statistics (including mean, variance, skewness, and kurtosis).
7. Test K-Means configurations for K = 2 through K = 8 and select the K with the highest silhouette score.
8. Evaluate the selected clustering and save the processed CSV.
9. Generate four plots: a CSK runs histogram, a CSK/opponent runs boxplot, an elbow curve, and a PCA cluster scatter plot.

## Results from the supplied processed CSV
The processed file contains **200 rows**. Its cluster counts are:

- Cluster 0: 110 records
- Cluster 1: 90 records

Cluster feature averages (rounded to 2 decimal places):

         csk_runs  opponent_runs  csk_wickets  opponent_wickets  csk_sixes
cluster                                                                   
0          154.94         180.15         6.36              4.39       5.96
1          198.48         152.33         4.46              6.80      10.38

The supplied Python script calculates the silhouette score when it is run. Interpret that score as a measure of cluster separation; a score nearer 1 indicates better separation, a score near 0 indicates overlapping clusters, and a negative score can indicate poor assignments. Do not treat the clusters as guaranteed win/loss labels.

## Visualizations
See [`CSK_KMeans_4_Graphs.png`](CSK_KMeans_4_Graphs.png) for:
1. CSK Runs Histogram
2. Runs Boxplot
3. K-Means Elbow Method (WCSS)
4. K-Means Cluster Visualization (PCA)

The K-Means model is fitted on all five standardized features. PCA is applied afterward to project the data into two dimensions for visualization.

## Requirements
- Python 3.9 or later recommended
- pandas
- numpy
- matplotlib
- scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the project
Keep the Python file and raw CSV in the same directory, then run:

```bash
python CSK_KMeans_Project.py
```

The script writes/overwrites:
- `CSK_200_Processed_Clusters.csv`
- `CSK_KMeans_4_Graphs.png`

The script expects the raw input filename to be exactly `CSK_200_Raw_Synthetic.csv`.

## Project files
- `CSK_KMeans_Project.py` — complete Python analysis and clustering script.
- `CSK_200_Raw_Synthetic.csv` — supplied raw synthetic dataset.
- `CSK_200_Processed_Clusters.csv` — supplied processed dataset with cluster labels.
- `CSK_KMeans_4_Graphs.png` — supplied four-panel visualization.
- `PROJECT_REPORT.md` — expanded project explanation and methodology.
- `requirements.txt` — Python dependencies.

## Limitations and future work
- The dataset is synthetic, so findings describe this dataset only.
- K-Means uses numerical distance and can be affected by feature selection, scaling, and outliers.
- Cluster labels are arbitrary identifiers, not rankings or outcome labels.
- Future work could use verified historical match data, add more match statistics, compare clustering algorithms, and build an interactive dashboard.
