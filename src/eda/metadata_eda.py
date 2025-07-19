import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def create_output_directory(output_dir):
    """Create the output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")


def load_style_classes(style_class_file):
    """Load style classes from style_class.txt into a dictionary."""
    style_classes = {}
    with open(style_class_file, 'r') as f:
        for line in f:
            class_id, style_name = line.strip().split(maxsplit=1)
            style_classes[int(class_id)] = style_name
    return style_classes


def create_class_table(df, style_classes):
    """Create a table with class, style, count, and percentage."""
    class_counts = df['class_id'].value_counts()
    table = pd.DataFrame({
        'class': class_counts.index,
        'count': class_counts.values,
        'style': class_counts.index.map(style_classes),
    })
    table['percentage'] = (table['count'] / table['count'].sum()) * 100
    return table


def plot_class_distribution(table, title, output_plot):
    """Plot class distribution and save the plot."""
    plt.figure(figsize=(12, 6))
    plt.bar(table['style'], table['count'], color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel("Style")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")

    # Show the plot
    plt.show()


def eda_on_metadata(input_dir, style_class_file, output_dir):
    """
    Perform EDA for style_train.csv and style_val.csv.
    Args:
        input_dir (str): Directory containing style_train.csv and style_val.csv.
        style_class_file (str): Path to the style_class.txt file.
        output_dir (str): Directory to save output plots and tables.
    """
    # Load style classes
    style_classes = load_style_classes(style_class_file)

    # Create output directory
    create_output_directory(output_dir)

    for dataset in ["style_train.csv", "style_val.csv"]:
        print(f"\nProcessing {dataset}...")

        # Load dataset
        df = pd.read_csv(os.path.join(input_dir, dataset), header=None, names=["filename", "class_id"])

        # Create class distribution table
        table = create_class_table(df, style_classes)

        # Save class distribution table
        table_path = os.path.join(output_dir, f"{dataset.split('.')[0]}_class_distribution.csv")
        table.to_csv(table_path, index=False)
        print(f"Class distribution table saved to {table_path}")

        # Plot class distribution
        plot_title = f"Class Distribution in {dataset.split('_')[1].capitalize()} Dataset"
        plot_path = os.path.join(output_dir, f"{dataset.split('.')[0]}_class_distribution.png")
        plot_class_distribution(table, plot_title, plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform EDA on style_train.csv and style_val.csv.")
    parser.add_argument("input_dir", help="Directory containing style_train.csv and style_val.csv.")
    parser.add_argument("style_class_file", help="Path to the style_class.txt file.")
    parser.add_argument("output_dir", help="Directory to save output plots and tables.")
    args = parser.parse_args()

    eda_on_metadata(args.input_dir, args.style_class_file, args.output_dir)
