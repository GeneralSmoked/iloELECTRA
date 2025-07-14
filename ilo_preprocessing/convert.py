import os
import pandas as pd

def csv_to_txt_batch(input_root, output_root):
    os.makedirs(output_root, exist_ok=True)

    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.endswith('.csv'):
                input_csv_path = os.path.join(root, file)

                relative_path = os.path.relpath(root, input_root)
                output_dir = os.path.join(output_root, relative_path)
                os.makedirs(output_dir, exist_ok=True)

                base_name = os.path.splitext(file)[0]
                output_txt_path = os.path.join(output_dir, base_name + ".txt")

                try:
                    df = pd.read_csv(input_csv_path)
                    text_column = df.columns[0]
                    lines = df[text_column].dropna().astype(str).str.strip()
                    lines = lines[lines !=""]

                    with open(output_txt_path, 'w', encoding='utf-8') as f:
                        for line in lines:
                            f.write(line + '\n')

                        print(f"Converted: {input_csv_path} -> {output_txt_path} ({len(lines)} lines)")
                except Exception as e:
                    print(f"Failed to process {input_csv_path}: {e}")


csv_to_txt_batch("IloELECTRA Corpus", "IloELECTRA Corpus txt files")