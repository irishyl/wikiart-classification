#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 22:37:53 2024

@author: lvxinyuan
"""

import os
from pyspark.sql import SparkSession

# Create a Spark session
spark = SparkSession.builder \
    .appName("Process Parquet Files") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()


# Path to the folder containing Parquet files
input_folder = "/scratch/hl5679/wikiart/data"
output_folder = "/scratch/hl5679/wikiart/data_cleaned"

file_list = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.parquet')]
# print(file_list)

for file in file_list:
    df = spark.read.parquet(input_folder)

    # Remove specific columns
    columns_to_drop = ['artist', 'genre']
    transformed_df = df.drop(*columns_to_drop)

    # Filter specific classes
    filter_values = [12, 21, 23, 9, 20, 24, 3, 4, 0, 17, 15, 7, 22]
    filtered_df = transformed_df.filter(transformed_df['style'].isin(filter_values))

    # Write the transformed DataFrame back to new Parquet files
    filtered_df.write.mode("overwrite").parquet(output_folder)

# Stop the Spark session
spark.stop()