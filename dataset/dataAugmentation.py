import json
import pandas as pd
from transformers import pipeline

# First turn the data into dataframe
with open('dataToAugment.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

output = df['output'].apply(pd.Series)
df = pd.concat([df.drop(columns=['output']), output], axis=1)

# Initialize the HuggingFace text generation pipeline
generator = pipeline("text-generation", model="facebook/bart-large-cnn", device=0)

def augment_dataset(generator, data, num_variations=5):
    print("Start the process")
    augmented_data = []
    
    for sample in data:
        prompt = ""         '''import the prompt from prompt.txt'''
        
        Generate {num_variations} variations:"""

        generated_variations = generator(prompt, max_length=100, truncation=True, num_return_sequences=num_variations, num_beams=5)

        # Process generated responses
        for variation in generated_variations:
            augmented_data.append({
                "input": variation["generated_text"].strip(),
                "output": sample["output"]  # Keep output structure unchanged
            })
    
    return augmented_data

# Generate more data based on the existing dataset
new_data = augment_dataset(generator, data, num_variations=5)

# Save the augmented data to a file
with open('augmented_data.json', 'w') as f:
    json.dump(new_data, f, indent=4)

print(f"Generated {len(new_data)} new augmented samples.")