import pandas as pd

# Load CSVs
model1_csv = "french_baseline.csv"  # French-only model
model2_csv = "french_tl.csv"  # French + Spanish model (without CL)
cl_csv = "french_cl.csv"    # French + Spanish model (with CL)

df_model1 = pd.read_csv(model1_csv)
df_model2 = pd.read_csv(model2_csv)
df_cl = pd.read_csv(cl_csv)

# Identify forgotten points in Model 2 compared to Model 1
forgotten_in_model2 = df_model1[
    (df_model1["Predicted Label"] == df_model1["Actual Label"]) & 
    (df_model2["Predicted Label"] != df_model2["Actual Label"])
]

# Identify retained points in Continual Learning model compared to Model 2
retained_in_cl = df_model2[
    (df_model2["Predicted Label"] != df_model2["Actual Label"]) & 
    (df_cl["Predicted Label"] == df_cl["Actual Label"])
]

# Save the results to CSVs for further analysis
forgotten_in_model2.to_csv("forgotten_in_model2.csv", index=False)
retained_in_cl.to_csv("retained_in_cl.csv", index=False)

# Optional: Analyze patterns in forgotten/retained points
def analyze_patterns(df, column="Review"):
    # Example: Analyze length of reviews or frequent words
    df["Review Length"] = df[column].apply(len)
    print("Average review length:", df["Review Length"].mean())
    print("Most frequent words:", pd.Series(" ".join(df[column]).split()).value_counts().head(10))

print("Patterns in forgotten points:")
analyze_patterns(forgotten_in_model2)

print("Patterns in retained points:")
analyze_patterns(retained_in_cl)
