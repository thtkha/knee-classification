import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

metadata = pd.DataFrame(columns=["Name", "Path", "KL"])
main_dir = "./OSAIL_KL_Dataset/Labeled"

for grade in range(5):
    folder_path = os.path.join(main_dir, str(grade))
    
    for image in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image)
        metadata = metadata._append({
            "Name": image,
            "Path": image_path,
            "KL": grade
        }, ignore_index=True)

metadata.to_csv("metadata.csv", index=False)

# Hold out 20% for test
train_val_df, test_df = train_test_split(
    metadata, 
    test_size=0.2, 
    stratify=metadata['KL'], 
    random_state=42
)

print(f"Test set size: {len(test_df)}")

# Create CSVs folder if it doesn't exist
os.makedirs("data/CSVs", exist_ok=True)

# Save test set
test_df.to_csv("CSVs/test.csv", index=False)

# Stratified 5-fold cross-validation on remaining 80%

## Ensure KL is integer
train_val_df['KL'] = train_val_df['KL'].astype(int)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 0

for train_index, val_index in skf.split(train_val_df, train_val_df['KL']):
    train_fold = train_val_df.iloc[train_index]
    val_fold = train_val_df.iloc[val_index]
    
    print(f"Fold {fold}: Train={len(train_fold)}, Validation={len(val_fold)}")
    
    # Save train and validation CSV files directly in CSVs folder
    train_fold.to_csv(f"CSVs/fold_{fold}_train.csv", index=False)
    val_fold.to_csv(f"CSVs/fold_{fold}_val.csv", index=False)
    
    fold += 1

print("\nCSV files saved in 'CSVs' folder:")
print("- CSVs/test.csv")
for i in range(5):
    print(f"- CSVs/fold_{i}_train.csv")
    print(f"- CSVs/fold_{i}_val.csv")

