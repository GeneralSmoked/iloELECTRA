# iloELECTRA

**iloELECTRA** is a pipeline for building an Ilocano ELECTRA-based language model, including preprocessing scripts, a full corpus, and training modules.

---

## 📁 Folder Structure
```bash
iloELECTRA/
├── trainingScripts/ # Model training and evaluation
│ ├── training.py
│ └── eval_metrics.py
│
├── ilo_preprocessing/ # Corpus + preprocessing scripts
│ ├── IloELECTRA Corpus/ # Raw CSV files by domain
│ ├── IloELECTRA Corpus txt files/ # Converted TXT files
│ ├── clean.py # Cleans corpus
│ ├── convert.py # Converts CSVs to plain text
│ ├── merge.py # Merges all text into one
│ └── iloELECTRA_pretrain.txt # Final cleaned corpus
│
├── iloELECTRA_Multilabel.py # Multilabel classification task
├── iloELECTRA_NER.py # Named entity recognition task
├── requirements.txt # Python dependencies
```
## 🧹 How to Preprocess the Corpus

```bash
cd ilo_preprocessing
python clean.py
python convert.py
python merge.py
```

## 🧠 How to Train and Evaluate
```bash
cd trainingScripts
python training.py
python eval_metrics.py
```

## 📌 Notes
All data used is Ilocano text from various domains.

This repo includes the full corpus to ensure reproducibility.

Model checkpoints are excluded due to GitHub size limits.

##⚙️ Requirements
Install dependencies:

```bash
pip install -r requirements.txt
