import pandas as pd

# Read the hyphenize dataset
df = pd.read_csv('curated_dataset_hyphenize.csv')

new_prompts = []

for index, row in df.iterrows():
    # Keep the exact prompt text
    original_prompt = str(row['prompt'])
    
    # Wrap it to match the format
    formatted_prompt = f"[{{'role': 'user', 'content': {repr(original_prompt)}}}]"
    new_prompts.append(formatted_prompt)

# Make a new DataFrame with just the prompt column
new_df = pd.DataFrame({'prompt': new_prompts})
new_df.to_csv('curated_dataset_hyphenize_formatted.csv', index=False)
print("Done! Saved to curated_dataset_hyphenize_formatted.csv")