import os
import pandas as pd
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


def filter_top10_classes(train_csv, val_csv, style_classes, output_dir):
    """
    Filter the top 10 classes from train and val datasets and save to output directory.
    Args:
        train_csv (str): Path to style_train.csv.
        val_csv (str): Path to style_val.csv.
        style_classes (dict): Mapping of class IDs to style names.
        output_dir (str): Directory to save filtered datasets and top 10 classes.
    """
    # Load train dataset
    train_df = pd.read_csv(train_csv, header=None, names=["filename", "class_id"])
    class_counts = train_df['class_id'].value_counts()
    top10_classes = class_counts.head(10).index.tolist()

    # Save filtered train dataset
    filtered_train_df = train_df[train_df['class_id'].isin(top10_classes)]
    filtered_train_path = os.path.join(output_dir, "filtered_style_train.csv")
    filtered_train_df.to_csv(filtered_train_path, index=False, header=False)
    print(f"Filtered training dataset saved to {filtered_train_path}")

    # Load val dataset
    val_df = pd.read_csv(val_csv, header=None, names=["filename", "class_id"])
    filtered_val_df = val_df[val_df['class_id'].isin(top10_classes)]
    filtered_val_path = os.path.join(output_dir, "filtered_style_val.csv")
    filtered_val_df.to_csv(filtered_val_path, index=False, header=False)
    print(f"Filtered validation dataset saved to {filtered_val_path}")

    # Save top 10 classes to a file
    top10_classes_dict = {class_id: style_classes[class_id] for class_id in top10_classes}
    top10_classes_path = os.path.join(output_dir, "top10_style_classes.txt")
    with open(top10_classes_path, 'w') as f:
        for class_id, style_name in top10_classes_dict.items():
            f.write(f"{class_id} {style_name}\n")
    print(f"Top 10 classes saved to {top10_classes_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter top 10 classes from style_train.csv and style_val.csv.")
    parser.add_argument("input_dir", help="Directory containing style_train.csv and style_val.csv.")
    parser.add_argument("style_class_file", help="Path to the style_class.txt file.")
    parser.add_argument("output_dir", help="Directory to save filtered datasets and top 10 classes.")
    args = parser.parse_args()

    # Create output directory
    create_output_directory(args.output_dir)

    # Load style classes
    style_classes = load_style_classes(args.style_class_file)

    # Filter top 10 classes
    filter_top10_classes(
        os.path.join(args.input_dir, "style_train.csv"),
        os.path.join(args.input_dir, "style_val.csv"),
        style_classes,
        args.output_dir,
    )
