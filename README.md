# Earnings Call NLP Risk Signals

**Detecting Hedging Language in Corporate Communications**

An NLP framework for analyzing earnings call transcripts to detect uncertainty and hedging patterns. Uses the Loughran-McDonald finance-specific lexicon to quantify linguistic risk signals and correlate them with financial performance.

## Why This Matters

Earnings calls contain signals beyond the numbers. Executives hedge when they're uncertain - but *when* they hedge depends on company context. This framework surfaces those patterns systematically, enabling:

- **Investment analysis:** Detect linguistic shifts before they show up in financials
- **Risk assessment:** Quantify uncertainty in management communication
- **Competitive intelligence:** Compare communication strategies across companies

## Key Finding

Growth context determines hedging behavior:

| Company Profile | Pattern | Correlation |
|-----------------|---------|-------------|
| **High-growth (NVIDIA)** | Hedging increases as growth peaks become harder to sustain | r = +0.959 |
| **Mature (Microsoft)** | Hedging increases when growth slows | r = -0.584 |

High-growth companies hedge when performance pressure mounts. Mature companies hedge when momentum stalls.

## Methodology

1. **Data Collection:** Q&A sections from earnings call transcripts (where unscripted language reveals more)
2. **Text Processing:** Speaker segmentation, tokenization, normalization
3. **Uncertainty Detection:** Pattern matching against Loughran-McDonald uncertainty lexicon (297 terms)
4. **Correlation Analysis:** Uncertainty density vs. quarterly revenue performance

## Repository Structure
```
├── data/
│   ├── transcripts/              # Raw earnings call transcripts
│   └── processed/                # Cleaned Q&A sections
├── src/
│   ├── extract_qa.py             # Q&A section extraction
│   ├── uncertainty_analysis.py   # L-M dictionary matching
│   └── correlation.py            # Statistical analysis
├── results/
│   └── figures/                  # Generated visualizations
├── lm_uncertainty_terms.txt      # Loughran-McDonald uncertainty word list
└── README.md
```

## Quick Start
```python
from src.uncertainty_analysis import calculate_uncertainty_density

# Load Loughran-McDonald uncertainty terms
with open('lm_uncertainty_terms.txt', 'r') as f:
    uncertainty_terms = set(f.read().split())

# Analyze transcript
text = "We believe revenue could potentially exceed expectations..."
density, count, total = calculate_uncertainty_density(text, uncertainty_terms)
print(f"Uncertainty density: {density}%")
```

## Sample Results

| Company   | Q2 Growth | Q3 Growth | Q4 Growth | Uncertainty Correlation |
|-----------|-----------|-----------|-----------|-------------------------|
| NVIDIA    | 15.4%     | 17.0%     | 12.0%     | +0.959                  |
| Microsoft | 6.1%      | 0.7%      | 9.0%      | -0.584                  |

## Data Sources

- **Transcripts:** [Seeking Alpha](https://seekingalpha.com)
- **Uncertainty Dictionary:** [Loughran-McDonald Master Dictionary](https://sraf.nd.edu/loughranmcdonald-master-dictionary/)
- **Financial Data:** Company investor relations filings

## Applications

- Earnings call monitoring and alerting
- Pre-earnings sentiment analysis
- Management credibility scoring
- Sector-wide linguistic benchmarking

## Roadmap

- [ ] Expand to full S&P 500 coverage
- [ ] Add real-time transcript ingestion
- [ ] Build uncertainty trend dashboards
- [ ] Integrate additional L-M categories (litigious, negative, constraining)

## License

MIT License

## Author

**Ahdithan Uthayakumar**  
[LinkedIn](https://linkedin.com/in/ahdithan) · [GitHub](https://github.com/ahdithanu)
```

---

**Changes made:**
- Removed all Northwestern/MSDS references
- Added "Why This Matters" with business applications
- Added "Applications" section showing real-world use cases
- Added "Roadmap" to signal this is ongoing work, not a finished assignment
- Reframed as "framework" not "project"
- Citation section removed entirely
- Positioning: builder creating a tool, not student submitting homework

---

**Also update the GitHub description field to:**
```
NLP framework for detecting hedging language in earnings call transcripts. Correlates uncertainty patterns with financial performance using the Loughran-McDonald lexicon.
