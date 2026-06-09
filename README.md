# Electrical Load Fingerprinting using NILM

> *Your electricity meter sees one number. But every appliance inside your home has left a unique fingerprint in that signal. This project reads them.*

---

## The Problem

Every home has a single electricity meter. It records total power consumption — the sum of everything running at once. A fridge, a fan, a washing machine, a phone charger — all collapsed into one number.

But here's what's interesting from an electrical engineering standpoint: **every appliance has a distinct electrical signature**. A motor has inrush current spikes. A resistive heater has a clean rectangular ON/OFF transition. A switching power supply creates harmonic distortion. A compressor has a characteristic ramp-up curve.

This field is called **NILM — Non-Intrusive Load Monitoring**. The idea: disaggregate that single aggregate signal back into individual appliances using signal features and machine learning. No extra sensors. No hardware changes. Just smarter reading of data that already exists.

This is used in:
- Smart meters and energy auditing
- Demand-side management in power grids
- Detecting faulty or energy-wasting appliances
- Building automation systems

---

## Why I built this

I'm a third-year Electrical Engineering student at NIT Rourkela. In our circuits and machines labs, I noticed that different loads behave completely differently on an oscilloscope — a motor looks nothing like a heater. I started wondering: if these signatures are so visually distinct, can a machine learn to tell them apart from aggregate data?

This project is my attempt to answer that.

---

## Dataset

**REFIT — Household Electricity Dataset**

For this project, I use House 1 low-frequency data from the REFIT dataset.

- UK residential electricity consumption dataset
- Aggregate mains power along with appliance-level sub-meter readings
- 1 Hz sampling rate
- Multiple monitored household appliances
- Appliance-level ground truth available through sub-metering

**What each row contains:**
- Unix timestamp
- Aggregate power reading (watts)
- Individual appliance power readings (ground truth labels)

The project uses a cleaned REFIT House 1 CSV containing aggregate mains power and appliance-level measurements.

---

## Project Structure

```text
load-fingerprinting/
│
├── data/
│   ├── raw/                      # Downloaded REFIT files go here
│   └── processed/                # Cleaned, feature-engineered CSVs
│
├── notebooks/
│   ├── 01_data_exploration.ipynb       # Understanding the raw signal
│   ├── 02_signal_analysis.ipynb        # Extracting electrical features
│   ├── 03_event_detection.ipynb        # Detecting ON/OFF transitions
│   ├── 04_feature_engineering.ipynb    # Building the feature matrix
│   └── 05_modelling.ipynb             # Training and evaluating classifiers
│
├── utils/
│   └── signal_utils.py               # Reusable signal processing functions
│
├── outputs/                          # All saved plots
├── requirements.txt
└── README.md
```

---

## Approach — Step by Step

### Step 1: Understand the raw signal

Plot aggregate power over time. Visually identify when appliances switch ON/OFF. Get comfortable with the data before touching any ML.

### Step 2: Signal-level analysis

For each appliance, study its individual power trace:

- What does its steady-state wattage look like?
- How sharp or gradual is its ON transition?
- Does it have periodic behavior (compressor cycling)?
- What's the variance during operation?

This is where EE knowledge directly applies. A pure CS student would skip this step. You shouldn't.

### Step 3: Event detection

Detect state-change events in the aggregate signal using edge detection on the power trace — essentially finding significant step-changes (ΔP > threshold). Each detected event is a candidate appliance switching ON or OFF.

### Step 4: Feature engineering

For each detected event, extract features:

- `delta_power`: magnitude of the power change
- `delta_power_abs`: absolute magnitude of the power change
- `power_before_mean`: average power before the event
- `power_before_std`: variation before the event
- `steady_state_mean`: average power after the event
- `steady_state_std`: stability after the event
- `steady_state_max`: peak power after the event
- `steady_state_min`: minimum power after the event
- `rise_time_seconds`: how quickly did power reach steady state?
- `time_of_day_hour`: hour of day
- `is_nighttime`: nighttime usage indicator
- `direction`: ON/OFF event indicator

### Step 5: Train classifiers

Label each event with its appliance (from ground truth). Train:

- Logistic Regression (baseline)
- Decision Tree (interpretable)
- Random Forest (best performer)

Evaluate per-appliance F1 score. Accuracy alone is not enough to judge multi-class appliance classification.

---

## Key Results

| Model | Accuracy | Macro F1 |
|--------|----------|----------|
| Logistic Regression | 0.45 | 0.43 |
| Decision Tree | 0.49 | 0.46 |
| Random Forest | 0.59 | 0.56 |

Additional observations:

- Random Forest achieved the best overall performance.
- Computer events were classified most reliably.
- Fridge and Washer_Dryer events were more difficult to separate because of overlapping power signatures.
- Feature importance analysis showed that steady-state power characteristics were the most informative predictors.
- 5-fold cross-validation produced an average accuracy of approximately **0.64 ± 0.08**, indicating reasonable generalization across different train-test splits.

---

## What I found interesting

- Computer events were the easiest to identify because of their distinctive power signatures
- Fridge and Washer_Dryer were harder to separate — overlapping power characteristics caused frequent confusion
- Time-of-day turned out to be less important than the electrical features themselves
- Steady-state power features contributed most to appliance classification

---

## What I'd do differently

- Use high-frequency data for harmonic feature extraction — would dramatically improve classification performance
- Try a Hidden Markov Model approach — NILM researchers use these for state-based appliance modeling
- Apply to Indian household data — load profiles can differ significantly from REFIT

---

## How to Run

### 1. Prepare the dataset

Place the cleaned REFIT House 1 CSV file inside:

```text
data/raw/
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run notebooks in order

```bash
jupyter notebook
```

Open:

```text
01_data_exploration.ipynb
→ 02_signal_analysis.ipynb
→ 03_event_detection.ipynb
→ 04_feature_engineering.ipynb
→ 05_modelling.ipynb
```

---

## Requirements

See `requirements.txt`.

Core packages:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- scipy
- jupyter

---

## References

- Hart, G.W., "Nonintrusive Appliance Load Monitoring", Proceedings of the IEEE, 1992
- Kelly & Knottenbelt, "Neural NILM", ACM BuildSys, 2015
- REFIT: Electrical Load Measurements for UK Homes

---

## About

Built by **Soumya** — 3rd year Electrical Engineering, NIT Rourkela.

Combining signal processing knowledge from EE coursework with ML to solve a real power systems problem.

LinkedIn: *[your link]*  
GitHub: https://github.com/Som0111/electrical-load-fingerprinting
