import csv
import re

def clean_special_characters(text):
    return re.sub(r'[^a-zA-Z0-9\s-]', '', text)

def clean_csv(input_file, output_file):
    with open(input_file, mode='r', encoding = 'utf-8', newline = '') as infile, \
         open(output_file, mode='w', encoding = 'utf-8', newline = '') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            cleaned_row = []
            for cell in row:
                if isinstance(cell, str):
                    cleaned_row.append(clean_special_characters(cell))
                else:
                    cleaned_row.append(cell)
            writer.writerow(cleaned_row)

    print(f"Cleaned CSV saved to '{output_file}'.")

input_csv = 'madlad-400_not_cleaned.csv'
output_csv = 'madlad-400_cleaned.csv'
clean_csv(input_csv, output_csv)