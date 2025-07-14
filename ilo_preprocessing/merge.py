import os 

def merge_txt_files(input_root, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(input_root):
            for file in files:
                if file.endswith('.txt'):
                    txt_path = os.path.join(root, file)
                    with open(txt_path, 'r', encoding='utf-8') as infile:
                        for line in infile:
                            clean_line = line.strip()
                            if clean_line:
                                outfile.write(clean_line + '\n')
                    print(f" Merged: {txt_path}")
    print(f"All text files merged into: {output_file}")

merge_txt_files("IloELECTRA Corpus txt files", "iloELECTRA_pretrain.txt")