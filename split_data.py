import pandas as pd
from sklearn.model_selection import train_test_split
# Read the dataframe from a CSV file (replace 'your_file.csv' with your actual file)
df = pd.read_csv('../../main_notebooks/gen_images_for_CNN/crop_metadata.csv')

print(df)

# Split the dataframe into 80% train and 20% test with stratification on 'condition_index'
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['condition_index'])

# Check the proportions of the 'condition_index' entries in the original dataframe
print("Original dataframe proportions:")
print(df['condition_index'].value_counts(normalize=True))

# Check the proportions of the 'condition_index' entries in the train dataframe
print("\nTrain dataframe proportions:")
print(train_df['condition_index'].value_counts(normalize=True))

# Check the proportions of the 'condition_index' entries in the test dataframe
print("\nTest dataframe proportions:")
print(test_df['condition_index'].value_counts(normalize=True))

# Save the train and test datasets to CSV files
train_df.to_csv('train_dataset.csv', index=False)
test_df.to_csv('test_dataset.csv', index=False)

