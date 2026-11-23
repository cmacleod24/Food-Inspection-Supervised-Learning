# Food-Inspection-Supervised-Learning

Predicting Chicago Food Inspection Failures

A supervised-learning notebook (Food_Inspection_Module_5.ipynb) that predicts whether a Chicago food establishment will fail a health inspection, using only information available before the visit. 

Can an establishment's risk category, facility type, inspection type, and inspection history predict whether its next inspection will be a Fail? A model that ranks inspections by failure probability could help the Chicago Department of Public Health (CDPH) decide which establishments to visit first.

Data
Source: City of Chicago Food Inspections dataset from Kaggle: https://www.kaggle.com/datasets/tjkyner/chicago-food-inspections
Unit of data: one row per inspection (221,468 raw rows).
Labels: the Results field, recorded by CDPH inspectors. The notebook turns it into a binary target: fail = 1 for "Fail", fail = 0 for "Pass" or "Pass w/ Conditions".
Raw columns used: Results, License #, Inspection Date, Inspection ID, Facility Type, Risk, Inspection Type, and DBA Name (display only, in the error table).
Raw columns not used: AKA Name, Address, City, State, Zip, Violations, Latitude, Longitude, Location.

What the notebook does
Step (cell)	Description
Setup (0)	Imports and plotting style
Load and clean (1)	Keeps completed outcomes (Pass, Pass w/ Conditions, Fail); drops inspections from March 2020 onward (COVID-era practice changes); drops missing/zero license numbers and rows without a risk level; groups facility types; standardizes inspection types; builds history features; removes re-inspections from the modeled rows
Split (2)	Time-based split: train = 2010-2017 (112,282 rows), test = 2018 to Feb 2020 (25,974 rows)
Model and baseline (3)	Logistic regression with class_weight="balanced", compared against an "always Pass" dummy baseline
Model comparison (4)	LazyPredict on a 5,000-row training subsample, as a rough check against 27 other classifiers
Cross-validation (5)	5-fold GroupKFold on the training set, grouped by license
Diagnostics (6)	Confusion matrix and top coefficients
Error analysis (7)	Five misclassified test inspections (3 false negatives, 2 false positives) with the features driving each prediction
Engineered features

None of these are columns in the original CSV; all are derived in the notebook.

Feature	Type	Built from	Meaning
facility	categorical	Facility Type	Keyword-grouped into 8 categories: Restaurant, Grocery Store, School, Daycare / Children's, Bakery, Care facility, Catering / Mobile, Other
insp_type	categorical	Inspection Type	Cleaned text; the six most common types are kept and the rest become "Other"
month	categorical	Inspection Date	Month of the inspection
risk_level	numeric	Risk	1 (High), 2 (Medium), 3 (Low)
prior_inspections	numeric	License #, Inspection Date	Count of earlier inspections for that license
prior_fail_rate	numeric	License #, Results	Share of earlier inspections that failed (0 if none)
days_since_last	numeric	License #, Inspection Date	Days since the previous inspection, capped at 730; missing values filled with the training-years median
first_in_data	binary	License #, Inspection Date	1 if no earlier inspection exists in the data

History features count only earlier inspections (the current result is subtracted), so the model never sees its own label. They are computed before re-inspections are dropped, so an establishment's full history is counted even though re-inspections are not themselves predicted.

Results

Modeling set: 138,256 inspections, 25.9% failures (77,666 Pass, 24,845 Pass w/ Conditions, 35,745 Fail).

Model	Accuracy	Precision (Fail)	Recall (Fail)	F1 (Fail)
Logistic regression, test set	0.613	0.340	0.520	0.411
Logistic regression, 5-fold CV (mean ± sd)	0.616 ± 0.005	0.349 ± 0.006	0.559 ± 0.007	0.429 ± 0.005
Dummy baseline (always Pass), test set	0.740	0.000	0.000	0.000

Accuracy is misleading on this imbalanced target: the do-nothing baseline scores higher than the model. The model is judged on precision and recall for the Fail class.

Strongest coefficients:

Toward Fail: Facility = Other (+0.43), Inspection type = Complaint (+0.36), Inspection type = License (+0.32), first inspection (+0.29), prior fail rate (+0.25)
Toward Pass: Facility = Daycare / Children's (-0.38), Inspection type = Other (-0.36), Inspection type = Canvass (-0.29), Facility = Care facility (-0.25), Month = December (-0.18)

The LazyPredict comparison found no model clearly better (best ROC AUC about 0.61, versus about 0.60 for logistic regression), which suggests the limit is the information in the features.

How to run
Download the Food Inspections CSV from the Chicago Data Portal.
Put it at /content/Food_Inspections.csv (the Google Colab path used in cell 1), or change the path in cell 1.
Install dependencies: numpy, pandas, matplotlib (3.6+ for the seaborn-v0_8-whitegrid style), scikit-learn (1.1+ for OneHotEncoder(min_frequency=...)), and lazypredict.
Run the cells in order. Cell 4 uses !pip install lazypredict, which works in Colab/Jupyter; run pip install lazypredict yourself elsewhere.

Cell 4 subsamples the training data to 5,000 rows because of memory limits. Results are for comparison only and use a fixed random_state=42.

Known limitations and gotchas
Inspection-type grouping: the top six types are chosen before re-inspection rows are removed, so some of the six may be re-inspection types that never appear in the modeling data. Check df["insp_type"].value_counts() before and after the drop. Rare types such as "Suspected Food Poisoning" end up in "Other".
Imputed gaps: first inspections receive the training-years median for days_since_last (284 days); it is a filled-in value, not an observation. The first_in_data flag is meant to compensate.
"Other" is coarse: the facility category mixes very different businesses.
Error-analysis filter: false positives are selected with Results == "Pass", which excludes "Pass w/ Conditions" even though both are class 0. Use fail == 0 for consistency.
Cross-validation folds are not time-ordered: the grouped folds mix years, unlike the time-based train/test split.
Label noise and selection bias: outcomes depend on the inspector and the day, and complaint-driven visits are not a random sample of establishments.
Time frame: the test period ends before COVID, so results may not hold under current practices.
Modest performance: about two of every three flagged inspections are false alarms. The model is a rough triage aid, not a basis for enforcement decisions.
Files
Food_Inspection_Module_5.ipynb: the analysis
Food_Inspections.csv: input data from Kaggle
