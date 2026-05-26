# Independent Study: Extraction of the Top Quark Branching Ratio $(R_b)$ at √s = 8 TeV

This repository contains the Monte Carlo generation cards, event selection algorithms, and statistical inference scripts for an independent phenomenological study based on the CMS collaboration's measurement framework (arXiv:2209.01222). 

The primary objective is the extraction of the ratio of top-quark branching fractions $[ R_b = B(t → Wb) / Σ B(t → Wq) ]$ in the dileptonic tt-bar decay channel. By testing the unitarity of the 3x3 CKM matrix, this parameter serves as a strict probe for Beyond the Standard Model (BSM) physics. This study validates the original paper's hypothesis: utilizing an orthogonal light-quark tagger (q-tagger) significantly constrains the combinatorial misassignment background during maximum likelihood optimization.

## 🔬 Monte Carlo & Analysis Framework
* **Event Generation:** MadGraph5_aMC@NLO (Exact matrix element calculation for tt-bar + 1j and backgrounds)
* **Parton Shower & Hadronization:** Pythia 8 (Implemented with k_T-MLM matching to prevent QCD double-counting, and utilizing the 5-Flavor Scheme to resolve tW initial-state anomalies)
* **Detector Simulation:** Delphes 3 (Parameterized fast-simulation of the CMS detector geometry and tracking resolution)
* **Data Analysis:** uproot and awkward (Event loop processing of ROOT trees, applying offline kinematic cuts and phase-space isolation)
* **Statistical Inference:** iminuit (Minuit2 engine performing a binned Profile Likelihood Ratio fit with continuous nuisance parameter profiling)

## 📂 Repository Structure
```text
Rb-TopQuark-Measurement/
├── README.md               
├── requirements.txt        
├── cards/                  # Configuration cards for the Monte Carlo generation pipeline
│   ├── ttbar_signal/       
│   ├── tw_background/
│   └── dy_background/
├── scripts/                # Analysis and statistical inference pipeline
│   ├── read_delphes.py     # Evaluates ROOT trees, applies MET/Z-veto cuts, and extracts tagging multiplicities
│   ├── model.py            # Defines the combinatorial probability matrix and binomial efficiency trials
│   ├── fit.py              # Executes the Minuit2 Profile Likelihood Ratio (PLR) fit
│   └── plot_mlj.py         # Derives and plots the lepton-jet invariant mass (m_lj) to model misassignment
└── report/
    ├── report.pdf          # Full 17-page theoretical methodology, derivations, and physics results
    └── report.tex



The inclusion of the orthogonal q-tagger successfully validated the theoretical framework: introducing mutually exclusive tracking parameters mathematically constrains the Profile Likelihood fit, drastically shrinking the 95% Confidence Intervals.

Extraction from Pseudo-Data Benchmark (R_b = 0.9):
PLR fit utilizing b-tag only {n_b}: R_b = 0.9039 (95% CI: [0.845, 1.500])
PLR fit utilizing dual taggers {n_b, n_q}: R_b = 0.9080 (95% CI: [0.879, 0.948])

Phase-Space Isolation & Event Yields
By applying a strict Z-mass resonance veto and imposing a Missing Transverse Energy (MET) threshold, the Drell-Yan background was successfully suppressed in the ee and μμ channels, cleanly isolating the tt-bar dileptonic signal region across all jet multiplicities (n_j ≥ 2).
(Note: Ensure this path points to your actual pushed image file in the repo)

Execution Instructions
1. Environment Setup
Bash
pip install -r requirements.txt


2. Offline Selection & Histogramming
(Note: Raw .root files are excluded from version control due to size constraints. The analysis scripts expect them in a local data/ directory).
Bash
python scripts/read_delphes.py --input data/tag_2_delphes_events.root --output counts.pkl


3. Profile Likelihood Fit
Executes the combinatorial likelihood model over the expected yields, profiling systematic uncertainties to extract the global best fit for R_b.
Bash
python scripts/fit.py --ttbar counts_ttbar.pkl --tW counts_tW.pkl --DY counts_DY.pkl
