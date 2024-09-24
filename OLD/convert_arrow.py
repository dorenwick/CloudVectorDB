import pandas as pd

# Read the parquet file
df = pd.read_parquet(r'C:\Users\doren\PycharmProjects\CloudVectorDB\NGRAM_BUILDER\filtered_small_ShortUnigramProcessor.parquet')

# Print the first 100 rows
print("First 100 rows:")
print(df.head(100).to_string())

print("\n" + "="*50 + "\n")

# Print the last 100 rows
print("Last 100 rows:")
print(df.tail(100).to_string())

print("\n" + "="*50 + "\n")

# Sort by 'count' column
df_sorted = df.sort_values('count', ascending=False)

# Print the first 100 rows of the sorted dataframe
print("First 100 rows after sorting by 'count':")
print(df_sorted.head(100).to_string())

print("\n" + "="*50 + "\n")

# Print the last 100 rows of the sorted dataframe
print("Last 100 rows after sorting by 'count':")
print(df_sorted.tail(100).to_string())