import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def load_top10_classes(top10_file):
    """Load top 10 classes from top10_style_classes.txt."""
    top10_classes = []
    with open(top10_file, 'r') as f:
        for line in f:
            class_id, style_name = line.strip().split(maxsplit=1)
            top10_classes.append(int(class_id))
    return top10_classes


def filter_dataset_by_classes(csv_file, top10_classes):
    """Filter a dataset to include only rows with the top 10 classes."""
    df = pd.read_csv(csv_file, header=None, names=["filename", "class_id"])
    return df[df['class_id'].isin(top10_classes)]


def plot_class_distribution(df, title, output_plot):
    """Plot class distribution for the filtered dataset."""
    class_distribution = df['class_id'].value_counts()
    plt.figure(figsize=(12, 6))
    plt.bar(class_distribution.index.astype(str), class_distribution.values, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel("Class ID")
    plt.ylabel("Number of Samples")
    plt.tight_layout()

    # Save plot
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")

    # Show plot
    plt.show()


def eda_on_top10(input_dir, top10_file, output_dir):
    """Perform EDA on filtered datasets."""
    # Load top 10 classes
    top10_classes = load_top10_classes(top10_file)

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for dataset in ["filtered_style_train.csv", "filtered_style_val.csv"]:
        print(f"\nProcessing {dataset}...")

        # Filter dataset by top 10 classes
        df = filter_dataset_by_classes(os.path.join(input_dir, dataset), top10_classes)

        # Plot class distribution
        plot_title = f"Top 10 Class Distribution in {dataset.split('_')[1].capitalize()} Dataset"
        plot_path = os.path.join(output_dir, f"{dataset.split('_')[1]}_top10_distribution.png")
        plot_class_distribution(df, plot_title, plot_path)


if __name__ == "__main__":
    # python metadata_eda.py ../../data/metadata ../../data/metadata/style_class.txt ../../output/images/full_metadata
    parser = argparse.ArgumentParser(description="Perform EDA on filtered datasets for top 10 classes.")
    parser.add_argument("input_dir", help="Directory containing filtered datasets.")
    parser.add_argument("top10_file", help="Path to the top10_style_classes.txt file.")
    parser.add_argument("output_dir", help="Directory to save output plots.")
    args = parser.parse_args()

    eda_on_top10(args.input_dir, args.top10_file, args.output_dir)
