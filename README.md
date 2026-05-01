# Aviation Wildlife Strikes Dashboard

🚀 **Live Interactive Dashboard:**  
https://aviation-wildlife-strikes-vl.streamlit.app/

## Dashboard Preview
Explore trends in wildlife strikes across time, geography, aircraft, and cost drivers.

![Dashboard Preview](figures/dashboard_preview.png)

---

## Overview

This project is an interactive Streamlit dashboard analyzing aviation wildlife strikes using data from the FAA Wildlife Strike Database.

The goal is to identify patterns in when and where strikes occur, which aircraft and species are most involved, and what factors are associated with more severe or costly incidents.

---

## Research Questions

- When do wildlife strikes occur?
- Where are they most concentrated?
- Which aircraft and species are most involved?
- Do flight variables explain the cost of incidents?

---

## Key Features

### Interactive Dashboard
- Filter by year range, operator, time of day, and aircraft  
- Consistent filtering across all pages  

### Heatmap
- Geographic concentration of strikes across the U.S.  
- Focus on contiguous states  

### Yearly Trends
- Long-term trend analysis  
- Highlights the 2020 COVID-related drop  

### Time Analysis
- Distribution by time of day, hour, and phase of flight  
- Emphasizes critical phases like takeoff and landing  

### Seasonality
- Monthly and quarterly patterns  
- Peak activity in late summer (August, Q3)  

### Aircraft Analysis
- Breakdown by aircraft and engine type  
- Highlights concentration among large commercial aircraft  

### Species Analysis
- Identifies most frequently involved species  
- Shows concentration among a small subset  

### Cost Relationships
- Examines cost vs flight variables  
- Uses log scaling to reduce skew  
- Displays Pearson correlation  
- Allows filtering by warned flag  

---

## Key Insights

- Wildlife strikes have increased steadily over time  
- Clear drop in 2020 due to reduced flight activity  
- Most strikes occur during takeoff, landing, and approach  
- Strong seasonal pattern with peaks in late summer  
- Large commercial aircraft are most frequently involved  
- Cost relationships are weak but directionally consistent  

---

## Data

**Source:** FAA Wildlife Strike Database  
https://wildlife.faa.gov/

Includes:
- Incident date and time  
- Location (airport, state, coordinates)  
- Aircraft and engine characteristics  
- Species involved  
- Damage, injuries, and cost  

Processed data is stored as a compressed parquet file for efficient loading.

---

## Project Structure
## Project Structure

```
aviation-wildlife-strikes/
├── README.md
├── requirements.txt

├── src/
│   ├── app.py
│   ├── data.py
│   ├── filters.py
│   ├── plotting.py
│   ├── aggregations.py
│   └── app_pages/
│       ├── home.py
│       ├── heatmap.py
│       ├── yearly_strikes.py
│       ├── time_patterns.py
│       ├── seasonality.py
│       ├── aircraft.py
│       ├── species.py
│       └── cost_relationships.py

├── data/
│   ├── raw/        # ignored
│   └── processed/  # ignored

├── scripts/
│   └── clean_data.py

└── notebooks/
    └── cleaning.ipynb
```

aviation-wildlife-strikes/
README.md
requirements.txt

src/
app.py
data.py
filters.py
plotting.py
aggregations.py

app_pages/
home.py
heatmap.py
yearly_strikes.py
time_patterns.py
seasonality.py
aircraft.py
species.py
cost_relationships.py

data/
raw/ # ignored
processed/ # ignored

scripts/
clean_data.py

notebooks/
cleaning.ipynb


---

## Implementation Notes

- Modular architecture with one file per page  
- Shared utilities for filtering and plotting  
- Centralized data loading and preprocessing  
- Custom navigation using streamlit-option-menu  
- Uses caching for efficient data loading  

---

## About

Vlad Lee  
Data Scientist (MIDS, UC Berkeley)

🔗 LinkedIn: https://www.linkedin.com/in/vlad-lee  
💻 GitHub: https://github.com/Vlad-Lee  

---

## Future Improvements

- Add map clustering or density tuning  
- Build severity or cost prediction models  
- Incorporate causal analysis or experimentation methods  
- Improve species categorization (e.g. birds vs mammals)  
