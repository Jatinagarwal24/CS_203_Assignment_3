import pandas as pd
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa

# Step 1: Read the JSON files
file1 = pd.read_json(r'C:\Users\jatin\Downloads\csv_A.json')
file2 = pd.read_json(r'C:\Users\jatin\Downloads\csv_j.json')
file3 = pd.read_json(r'C:\Users\jatin\Downloads\csv_H.json')

# Step 2: Extract the 'truck_label' column
labels_1 = file1['truck_label'].map({'Truck': 1, 'No Truck': 0})
labels_2 = file2['truck_label'].map({'Truck': 1, 'No Truck': 0})
labels_3 = file3['truck_label'].map({'Truck': 1, 'No Truck': 0})

# Step 3: Combine the labels into a single DataFrame
data = pd.DataFrame({
    'Rater1': labels_1,
    'Rater2': labels_2,
    'Rater3': labels_3
})

# Step 4: Create the matrix for Fleiss' kappa
# Convert each row into the count of occurrences for each category
category_counts = data.apply(pd.Series.value_counts, axis=1).fillna(0).astype(int)

# Ensure columns are in the correct order (0, 1)
category_counts = category_counts.reindex(columns=[0, 1], fill_value=0)

# Step 5: Calculate Fleiss' kappa
kappa = fleiss_kappa(category_counts.values)
print(f"Fleiss' kappa: {kappa}")
