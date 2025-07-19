from datasets import load_dataset
import numpy as np
import pandas as pd
from collections import Counter
import csv

# Load the WikiArt dataset
dataset = load_dataset("huggan/wikiart", split="train")

# Style mapping
style_mapping = {
    0: "Abstract_Expressionism",
    1: "Action_painting",
    2: "Analytical_Cubism",
    3: "Art_Nouveau",
    4: "Baroque",
    5: "Color_Field_Painting",
    6: "Contemporary_Realism",
    7: "Cubism",
    8: "Early_Renaissance",
    9: "Expressionism",
    10: "Fauvism",
    11: "High_Renaissance",
    12: "Impressionism",
    13: "Mannerism_Late_Renaissance",
    14: "Minimalism",
    15: "Naive_Art_Primitivism",
    16: "New_Realism",
    17: "Northern_Renaissance",
    18: "Pointillism",
    19: "Pop_Art",
    20: "Post_Impressionism",
    21: "Realism",
    22: "Rococo",
    23: "Romanticism", 
    24: "Symbolism",
    25: "Synthetic_Cubism",
    26: "Ukiyo_e"
}

style_names = [style_mapping[style] for style in dataset["style"]]

# Count the occurrences of each style
style_counts = Counter(style_names)

# Total number of samples
total_samples = sum(style_counts.values())

# Print the distribution with percentages
print("Style distribution (with percentages):")
output_rows = []
for style, count in style_counts.items():
    percentage = (count / total_samples) * 100
    print(f"{style}: {count} ({percentage:.2f}%)")
    output_rows.append({"Style": style, "Count": count, "Percentage": f"{percentage:.2f}%"})


output_file = "style_distribution.csv"
with open(output_file, mode="w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["Style", "Count", "Percentage"])
    writer.writeheader()
    writer.writerows(output_rows)

print(f"Style distribution saved to {output_file}")

#missing_styles = sum(1 for item in dataset if item["style"] is None)
#print("Number of missing styles:", missing_styles)

# Display a few examples
# print(dataset[0])
# example = dataset[0]
# example['image'].show()
# image_array = np.array(example['image'])
# print(image_array.shape)
