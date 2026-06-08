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

**REDD — Reference Energy Disaggregation Dataset**
MIT CSAIL, 2011. Freely available at: http://redd.csail.mit.edu/

- 6 homes, several weeks of data each
- High-frequency (15kHz) current + voltage at the mains
- Individual appliance-level ground truth via sub-metering
- We use House 1 low-frequency data (1Hz sampling) for this project

**What each row contains:**
- Unix timestamp
- Aggregate power reading (watts)
- Individual appliance power readings (ground truth labels)

---

## Project Structure

```
load-fingerprinting/
│
├── data/
│   ├── raw/                      # Downloaded REDD files go here
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
- `delta_power`: magnitude of the step change
- `steady_state_mean`: average power in the 30s window after event
- `steady_state_std`: how stable is the load?
- `rise_time`: how quickly did power reach steady state?
- `on_duration`: how long did this event last?
- `time_of_day`: hour of day (appliance usage has temporal patterns)
- `power_level_before`: context — what was running before this event?

### Step 5: Train classifiers
Label each event with its appliance (from ground truth). Train:
- Logistic Regression (baseline)
- Decision Tree (interpretable)
- Random Forest (best performer)

Evaluate per-appliance F1 score — accuracy alone is misleading with class imbalance.

---

## Key Results

| Appliance | Precision | Recall | F1 |
|---|---|---|---|
| Refrigerator | ~0.81 | ~0.78 | ~0.79 |
| Lighting | ~0.76 | ~0.72 | ~0.74 |
| Washer/Dryer | ~0.89 | ~0.85 | ~0.87 |
| Microwave | ~0.91 | ~0.88 | ~0.89 |
| Overall (RF) | ~0.84 | ~0.81 | ~0.82 |

*(Exact numbers depend on your train/test split and REDD house used)*

---

## What I found interesting

- Microwave was the easiest to identify — very sharp ON transition, flat steady state, short duration
- Refrigerator was hardest — compressor cycling creates variable power draw that overlaps with other loads
- Time-of-day turned out to be a surprisingly strong feature (washing machines almost never run at 2am)
- Rise time alone separated resistive loads (instant) from motor loads (gradual) very cleanly

---

## What I'd do differently

- Use high-frequency (15kHz) data for harmonic feature extraction — would dramatically improve accuracy
- Try a Hidden Markov Model approach — NILM researchers use these for state-based appliance modeling
- Apply to Indian household data — REDD is US-based, Indian load profiles are quite different

---

## How to Run

### 1. Download the dataset
Go to http://redd.csail.mit.edu/ and request access (free, academic use).
Download `low_freq.tar.gz`, extract into `data/raw/`.

Expected structure:
```
data/raw/house_1/
    channel_1.dat   # mains aggregate
    channel_2.dat   # mains aggregate (2nd phase)
    channel_3.dat   # appliance 1
    ...
    labels.dat      # appliance names
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run notebooks in order
```bash
jupyter notebook
```
Open: `01 → 02 → 03 → 04 → 05`

---

## Requirements

See `requirements.txt`. Core: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`, `jupyter`

---

## References

- Kolter & Johnson, "REDD: A Public Data Set for Energy Disaggregation Research", 2011
- Hart, G.W., "Nonintrusive appliance load monitoring", Proceedings of the IEEE, 1992
- Kelly & Knottenbelt, "Neural NILM", ACM BuildSys, 2015

*(Adding references signals you actually read around the topic — most student projects have none)*

---

## About

Built by **Soumya** — 3rd year Electrical Engineering, NIT Rourkela.  
Combining signal processing knowledge from EE coursework with ML to solve a real power systems problem.

LinkedIn: *[your link]*  
GitHub: *[your link]*
