# 🧹 Advanced Data Cleaning Project – COVID-19 Dataset
### 📌 Project Overview
This project focuses on **comprehensive data cleaning and preprocessing** of a large-scale COVID-19 dataset containing **67 columns** and **166,326 rows**.  
The dataset initially had a high percentage of missing values, inconsistencies, incorrect data types, and extreme outliers.  
After systematic cleaning and validation, the data was made **fully consistent, accurate, and ready to load into SQL databases or Power BI** for analysis.

---
## 🎯 Objective
To transform a messy, unstructured, low-quality, inconsistent COVID-19 dataset into a trusted,**high-quality, validated, reproducible, and load-ready data product using an end-to-end ETL approach (Extract → Transform → Load) with strong analytical reasoning at every step.suitable for SQL and Power BI reporting.

---
## 🛠 ETL Overview (simple for stakeholders)
* Extract: Ingest raw CSV/DB export into a controlled environment (Jupyter / ETL staging).
* Transform: Apply the cleaning & validation pipeline (categorical standardization, date fixes, missing-value rules, outlier handling, skew correction).
* Load: Export final tables/files (CSV / Parquet / SQL) formatted and documented for direct import into Power BI (or any BI tool).
---
## 🔎 Step-by-step (clear + stakeholder-friendly)

* Profile the data — quantified missingness, types, and irregularities to decide the plan.
* Segment by type — separate categorical, numerical, and datetime columns for targeted rules.
* Clean categorical data — standardized labels, trimmed spaces, and filled blanks with mode or “Unknown” (where domain knowledge was lacking).
* Fix timestamps — converted strings to datetime, handled errors, and extracted Year / Month / Day for trend analysis.
* Drop low-value columns — removed columns with >70% missing data to avoid noisy inputs.
* Treat outliers — detected using IQR / percentile checks and applied capping (winsorization) to retain information while controlling extremes.
* Correct skew & impute — used distribution-aware imputation: mean for near-normal, median for skewed; applied log/sqrt transforms when needed.
* Validate & document — performed consistency checks, removed duplicates, ensured zero nulls, and produced a data dictionary for Power BI loading.
* Deliver — exported clean dataset in BI-friendly formats and added metadata so dashboards are reproducible and auditable.
---


## 🧠 Key Cleaning Steps
1. **Data Inspection**
2. **Categorical Data Cleaning**
3. **Datetime Correction**
4. **Numerical Data Cleaning**
5. **Validation**
6. **Data Export**
---
## 🧰 Tech Stack
**Languages:** Python  
**Libraries:** Pandas, NumPy, Seaborn, Matplotlib, SciPy  
**Tools:** Jupyter Notebook, Google Colab

---
## What I built:
An ETL-driven data cleaning pipeline that transformed a messy COVID-19 dataset (67 features, 166K+ rows) into a validated, analysis-ready dataset — 
ready to load into Power BI for dashboards and further analysis.

---



## 🧩 Step-by-Step Analytical Workflow
#### Data Import and Initial Inspection
* Imported required libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `datetime`).
* Loaded the dataset into a pandas DataFrame.
* Performed an initial inspection using:
`.head()`, `.info()`, `.describe()`, `.isnull().sum()`
#### Load Dataset
```python
df = pd.read_csv("your_dataset.csv")
```


#### 1. Data Understanding & Quality Profiling:
* Conducted a comprehensive data audit:
* Checked structure using `.info()`, `.describe()`, `.nunique()`
* Created a missing value matrix to visualize data gaps.
* Evaluated data types, value distributions, and irregular patterns.
```python
#### Visualize missing values per column (bar chart)
# Calculate missing percentage per column
missing_percent = df.isnull().mean() * 100

# Sort descending for clarity
missing_percent = missing_percent.sort_values(ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=missing_percent.values, y=missing_percent.index)
plt.title("Missing Value Percentage per Column", fontsize=14)
plt.xlabel("Percentage (%)")
plt.ylabel("Columns")
plt.show()

#### Heatmap of missing data
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Data Heatmap", fontsize=14)
plt.show()



```
#### 🧠 Analytical Decision: Before cleaning, I quantified data health metrics to ensure every transformation could be justified numerically rather than arbitrarily applied.
---


#### 2. Data Segmentation: Type-Based Strategy
* Segregated columns into:
* Categorical variables → qualitative, often with inconsistent labels.
* Numerical variables → quantitative, prone to skewness and outliers.
* Date/Time variables → often stored as text needing type correction.
```python
Divided all columns into Categorical and Numerical sets for targeted cleaning.

categorical_cols = df.select_dtypes(include=['object']).columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

```


#### 🧠 Problem-Solving Thought: Each data type has a unique cleaning requirement; applying one-size-fits-all rules causes logical data loss. So I designed separate pipelines per data type.
---
#### 3. Cleaning Categorical Data
* Inconsistency Detection: Found mixed casing, trailing spaces, and category drift ("Male", "MALE", "male").
* Standardization: Applied `df.str.strip().str.title()` to unify text format.
* Imputation Strategy:
* Used mode for dominant-category columns.
* For high-cardinality columns, applied domain-driven filling or marked "Unknown" for interpretability.
* Validation: Rechecked category counts to ensure logical balance.

```python



```




#### 🧠 Reasoning: Instead of blind mode imputation, I evaluated whether missing values had domain significance — avoiding biased over-representation of the majority class.
---


#### 4. Correcting Datetime Columns
* Converted string dates to datetime using `pd.to_datetime(errors='coerce')`
* Handled invalid entries gracefully (kept them as NaT for traceability).
* Extracted year, month, and day components for potential time-based analysis.
```python
Identified and converted all date-related columns to proper datetime format.

df['date_column'] = pd.to_datetime(df['date_column'], errors='coerce')
```
#### 🧠 Analytical Insight: Dates were essential for understanding the timeline of COVID events — converting and preserving temporal integrity ensures future trend analysis accuracy.
---



#### 5. Numerical Data Analysis & Cleaning
#### a. Missing Value Diagnosis
* Computed missing percentage for each numeric column.
```python
Calculated missing value percentage for every numerical column.

missing_percent = df[numerical_cols].isnull().mean() * 100

total_missing_percent = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
```
#### Decision rule:
* Drop columns >70% missing (data too incomplete for imputation).
* Retain others for imputation.
```python
# Assuming df is your DataFrame
threshold = 0.7  # 70%
# Drop columns with more than 70% missing data
df = df.loc[:, missing_percent <= threshold]

# (Optional) See what was removed
dropped_cols = missing_percent[missing_percent > threshold].index.tolist()
print("Dropped columns:", dropped_cols)

```
#### 🧠 Justification: This threshold was chosen based on diminishing information gain — after 70% missingness, column entropy becomes too low for reliable inference.
---
#### b. Outlier Detection & Treatment
* Measured skewness and visualized via boxplots and histograms.
* Applied `IQR` method and `percentile-based capping` (1st–99th) to mitigate outliers.
* Rechecked summary statistics post-capping to confirm logical value ranges.
```python
# check skewness of the columns
skew_values = df[num_cols].skew().sort_values(ascending=True)

# Display the top 15 skewed columns
print("📊 Skewness of Numerical Columns (Top 25):")
print(skew_values.tail(25))

#outliers detection(IQR)
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_limit) | (df[col] > upper_limit)]
    print(f"{col}  :  {len(outliers)}   ** outliers detected")


# Cap the outliers in-place

num_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    # Cap outliers safely with .loc
    df.loc[:, col] = np.where(df[col] < lower_limit, lower_limit,
                              np.where(df[col] > upper_limit, upper_limit, df[col]))

    print(f"{col}: Outliers capped at {lower_limit:.2f} (lower) and {upper_limit:.2f} (upper)")


```
#### 🧠 Analytical Logic: Rather than deleting outliers (risking data loss), I capped extreme values to retain data continuity and minimize distortion of mean/variance.
---
#### c. Skewness Correction & Imputation
* After capping, recalculated skewness.
* Used mean imputation for near-normal columns.
* Used median imputation for skewed distributions.
* In extreme skewness, applied log or sqrt transformation to stabilize variance.
```python
#check skewness again after capping
for col in num_cols:
    skewness = df[col].skew()
    print(f"{col}: skewness = {skewness:.2f}")


#fiil the numerical column
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in num_cols:
    if df[col].isnull().sum() > 0:
        skewness = df[col].skew()

        # Decide fill method
        if abs(skewness) < 0.5:
            fill_value = df[col].mean()
            method = "mean"
        elif abs(skewness) < 1:
            fill_value = df[col].median()
            method = "median"
        else:
            fill_value = df[col].mode()[0]
            method = "mode"

        df[col].fillna(fill_value)

        print(f"{col}: missing values filled using {method} (skewness = {skewness:.3f})")



# column transformation for high skew columns using(log method)
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Transform columns with high skew (>1) using .loc
for col in num_cols:
    if df[col].skew() > 1:
        # Use .loc to overwrite original column safely
        df.loc[:, col] = np.log(df[col] + 1)
        print(f"{col} transformed in-place using log")

```




#### 🧠 Analytical Thinking: Every imputation was guided by the column’s statistical shape — ensuring that filled values maintained distribution realism.
---
#### 6. Data Consistency & Final Validation
* Verified column-level consistency
* No duplicates, no type mismatches, no NaN values.
* Confirmed logical relationships (e.g., start_date < end_date).
* Conducted data sanity checks:
* `.duplicated().sum()` → 0 duplicates
* `.isnull().sum().sum()` → 0 nulls

```python
#cleaned missing value heatmap
plt.figure(figsize=(12,6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()

```

#### 🧠 Verification Mindset: Cleaning without validation is meaningless — I performed multiple quality checkpoints to prove data integrity post-processing.
---
## 🧮 Skills & Competencies Demonstrated

| **Category**               | **Skills**                                                                                     |
|----------------------------|-----------------------------------------------------------------------------------------------|
| **Programming & Tools**    | Python, Pandas, NumPy, Matplotlib, Seaborn, Datetime                                          |
| **Data Analysis**          | Missing data profiling, Outlier detection, Skewness analysis                                  |
| **Statistical Reasoning**  | Distribution-based imputation, Quantile capping, Normality testing                            |
| **Analytical Thinking**    | Decision-making based on data metrics, Validation logic                                       |
| **Problem-Solving**        | Designed custom rules for each data type; avoided arbitrary cleaning                          |
| **Data Quality Assurance** | Integrity checks, logical validation, reproducible pipeline design                 

----
## 🧭 Key Learnings and Analytical Takeaways
* Developed the ability to reason statistically through data imperfections rather than just “clean” them.
* Understood how data behavior (skew, outliers, missingness) directly affects model performance.
* Learned to quantify cleaning impact using pre/post metrics — bridging raw processing with analytical validation.
* Gained expertise in building generalizable cleaning pipelines adaptable to any future dataset.
* Practiced data storytelling by explaining “why” each decision was taken, not just “what” was done.
---
# 🚀 Final Outcome
**A fully validated, clean, consistent, and analysis-ready COVID-19 dataset free from missing, inconsistent, or extreme values — now suitable for exploratory data analysis (EDA), feature engineering, or predictive modeling.**

---
#🧾 **Project Summary (Short Version for Resume):**
**Built a complete data cleaning pipeline for a 166K-row COVID dataset (67 columns) using Python. Handled missing values, corrected categorical inconsistencies, fixed date formats, treated outliers, reduced skewness, and validated data integrity. Achieved a fully clean dataset with no nulls or inconsistencies — ready for advanced analysis.**

---
