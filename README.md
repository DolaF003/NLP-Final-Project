# NLP Final Project: Predicting NBA Draft Outcomes from Scouting Reports

Natural Language Processing Final Project (Spring 2026). We trained text classifiers on pre-draft scouting reports to predict NBA draft outcomes across four tasks of increasing granularity:

1. Drafted vs. Undrafted
2. First Round (picks 1-30) vs. Not First Round (second round or undrafted)
3. Top 10 picks vs. Picks 51-60 (the two extremes of the draft)
4. Pick-range prediction (six classes, drafted players only)

## Directory structure

```
NLP-Final-Project/
├── README.md
├── NLP_Project_Code.ipynb              # main notebook (all code, end-to-end)
├── Edited scouting_reports_master.csv  # 240 scout reports, scraped from HoopsProspects
└── nba_drafts/
    ├── nba_draft_2018.csv
    ├── nba_draft_2019.csv
    ├── nba_draft_2020.csv
    ├── nba_draft_2021.csv
    ├── nba_draft_2022.csv
    ├── nba_draft_2023.csv
    ├── nba_draft_2024.csv
    └── nba_draft_2025.csv
```

## Where to look for code

Everything lives in `NLP_Project_Code.ipynb`. The notebook is organized by section:

- Cells 0-17: Google Drive connection and the BeautifulSoup-based scraping pipeline used to build `Edited scouting_reports_master.csv` from HoopsProspects.
- Cells 18-23: Load the CSV files, normalize player names, merge scout reports with draft outcomes, and apply manual fixes for name mismatches.
- Cells 24-32: 80/20 train/test splits for all four tasks, plus synthetic augmentation of the Task 1 training set to address the 197 drafted / 43 undrafted class imbalance.
- Cells 33-35: Baseline classifiers (`sklearn.dummy.DummyClassifier` with majority-class and stratified strategies).
- Cells 36-39: CNN model built in TensorFlow/Keras (`Embedding` → `Conv1D` → `GlobalMaxPooling1D` → dense layers).
- Cells 40-74: DistilBERT (`distilbert-base-uncased`) fine-tuned separately for each of the four tasks via the Hugging Face `Trainer` API.
- Cells 75-77: TF-IDF and n-gram features fed into Logistic Regression and Multinomial Naive Bayes.
- Cells 78-82: Cross-experiment comparison plots.

## Where to look for data files

- `Edited scouting_reports_master.csv` is at the repo root. It has 240 rows, one per scout report, with columns for player name, report year, school, position, measurables (height, weight, wingspan, vertical, hand), and the full report text in `details_text`. The `report_url` column links back to the original HoopsProspects page for each report.
- `nba_drafts/` contains one CSV per draft year from 2018 to 2025. Each file lists the 60 drafted players that year along with their team, college/international affiliation, and overall pick number.

## How to run the code

The notebook was developed in Google Colab and assumes you mount Google Drive at `/content/drive`. Before running:

1. Create a folder called `NLP Final Project` in your Drive's `MyDrive`.
2. Inside it, create two subfolders: `hoopsprospects_output/` and `nba_drafts/`.
3. Put `Edited scouting_reports_master.csv` into `hoopsprospects_output/`.
4. Put all eight `nba_draft_YYYY.csv` files into `nba_drafts/`.

> Heads up: the repo flattens the scout reports CSV to the root, but the notebook's `SCOUTS_PATH` variable expects it inside `hoopsprospects_output/`. Either keep the Drive layout as described above, or edit `SCOUTS_PATH` in cell 19 to point at wherever you put the file.

To run:

1. Open `NLP_Project_Code.ipynb` in Google Colab.
2. Switch the runtime to GPU (Runtime → Change runtime type → T4 GPU). The DistilBERT fine-tuning needs this to finish in a reasonable time.
3. Run the cells top-to-bottom. Pip installs for `transformers`, `datasets`, and `evaluate` happen inside the notebook at cell 41.

## What order to run things in

Top to bottom. The preprocessing cells (18-23) must run before any of the train/test split cells (24-32), and those must run before any modeling cells. Within the modeling sections, each task is self-contained, so you can rerun only the BERT-Task-2 cells without rerunning BERT-Task-1.

The scraping cells at the top (1-17) do not need to run if you already have `Edited scouting_reports_master.csv`. They are kept in the notebook for reproducibility of how the dataset was built.

## Where we got the data

- Scout reports: scraped from [HoopsProspects](https://hoopsprospects.com/scouting-reports/) using `requests` and BeautifulSoup. The scraping code is in cells 1-17 of the notebook.
- NBA draft results: compiled from [Basketball Reference's draft pages](https://www.basketball-reference.com/draft/) for years 2018-2025, cleaned into one CSV per year.

## Non-standard libraries

The notebook uses pandas, numpy, scikit-learn, matplotlib, and seaborn (standard). Beyond that:

- [`wordcloud`](https://github.com/amueller/word_cloud) by Andreas Mueller, used for the word cloud visualizations.

Colab has TensorFlow pre-installed. The Hugging Face packages are installed in cell 41 via `!pip install`.

Transformers (Hugging Face) — used for DistilBERT fine-tuning
https://huggingface.co/docs/transformers/index

Datasets (Hugging Face) — used to create DatasetDict objects for BERT training
https://huggingface.co/docs/datasets/index

Evaluate (Hugging Face) — used to compute accuracy and F1 during BERT training
https://huggingface.co/docs/evaluate/index

TensorFlow / Keras — used to build and train the CNN
https://www.tensorflow.org/api_docs/python/tf/keras


## Models we used

- DistilBERT (distilbert-base-uncased) — pre-trained model fine-tuned for sequence classification
  (https://huggingface.co/distilbert/distilbert-base-uncased)

## Notebooks and tutorials we referenced

Beyond the official scikit-learn, seaborn, Hugging Face, and TensorFlow docs:

- [drakearch/kaggle-courses, `02-text-classification.ipynb`](https://github.com/drakearch/kaggle-courses/blob/master/natural_language_processing/02-text-classification.ipynb) on GitHub. Notebook version of the Kaggle NLP course; we adapted the word-frequency-between-groups comparison pattern from it.
- [WordCloud "Using frequency" example](https://amueller.github.io/word_cloud/auto_examples/frequency.html). Reference for the `generate_from_frequencies()` pattern we used for the TF-IDF weighted word clouds.
- [Seaborn "Annotated heatmaps" example](https://seaborn.pydata.org/examples/spreadsheet_heatmap.html). Reference for the annotated heatmap layout in the visualizations.
- DataCamp, [*Seaborn Heatmaps: A Guide to Data Visualization*](https://www.datacamp.com/tutorial/seaborn-heatmaps). Reference for `annot_kws` and `cbar_kws` parameters.
- DataCamp, [*Python Boxplots: A Comprehensive Guide for Beginners*](https://www.datacamp.com/tutorial/python-boxplots). Reference for the two-group boxplot comparison.

Much of our code was adapted from notebooks/examples we went over in class or for homework assignments.

## Data augmentation:

ChatGPT (GPT-4o) — used to generate 30 synthetic undrafted player scouting reports for Task 1 training data augmentation
https://openai.com/chatgpt

## Team

Abraham Gonzalez Diaz, Osaruese (Wesay) Okungbowa, Troy Healey, Adedola (Dola) Fakeye
