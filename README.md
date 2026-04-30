# Aviation Wildlife Strikes Dashboard

🚀 **Live Interactive Dashboard:**  
https://aviation-wildlife-strikes-vl.streamlit.app/

---

## Overview
...

This project is an interactive Streamlit dashboard that explores wildlife strikes in aviation using data from the FAA Wildlife Strike Database. The goal is to identify patterns in when, where, and how wildlife strikes occur, and to understand factors associated with more severe incidents.

Users can filter and explore trends across time, geography, aircraft characteristics, species, and cost relationships.

Research Questions
When do wildlife strikes occur?
Where are they most concentrated?
Which aircraft and species are most involved?
Do flight variables explain the cost of incidents?
Key Features
Interactive Dashboard
Filter by year range, operator, time of day, and aircraft
Consistent filtering across all pages
Heatmap
Geographic concentration of strikes across the U.S.
Focus on contiguous states
Yearly Trends
Long-term trend analysis
Highlights the 2020 COVID-related drop
Time Analysis
Distribution by time of day, hour, and phase of flight
Emphasizes critical phases like takeoff and landing
Seasonality
Monthly and quarterly patterns
Peak activity in late summer (August, Q3)
Aircraft Analysis
Breakdown by aircraft type and engine type
Focus on large commercial aircraft and turbofan engines
Species Analysis
Identifies species most frequently involved
Shows concentration among a small subset of species
Cost Relationships
Examines relationship between cost and flight variables
Uses log scaling to reduce skew
Displays Pearson correlation
Allows filtering by warned flag
Project Structure

aviation-wildlife-strikes/
README.md
requirements.txt

src/
app.py # Streamlit entry point and routing
data.py # Data loading
filters.py # Sidebar filters and filtering logic
plotting.py # Shared chart styling utilities
aggregations.py # Data transformations and aggregations

app_pages/            # Individual dashboard pages  
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
clean_data.py # Data cleaning pipeline

notebooks/
cleaning.ipynb # Exploratory work

Data

Source: FAA Wildlife Strike Database
https://wildlife.faa.gov/

The dataset includes:

Incident date and time
Location (airport, state, coordinates)
Aircraft and engine characteristics
Species involved
Damage, injuries, and cost

Processed data is stored as a compressed parquet file for efficient loading.

Implementation Notes
Modular architecture with one file per page
Shared utilities for filtering and plotting
Centralized data loading and preprocessing
Custom navigation using streamlit-option-menu
Disabled default Streamlit pages system to avoid duplicate navigation
Uses caching for efficient data loading
Key Insights
Wildlife strikes have increased steadily over time
Clear drop in 2020 due to reduced flight activity
Most strikes occur during takeoff, landing, and approach
Strong seasonal pattern with peaks in late summer
Large commercial aircraft are most frequently involved
Cost relationships with flight variables are weak but directionally consistent
About

Vlad Lee
Data Scientist (MIDS, UC Berkeley)

LinkedIn: https://www.linkedin.com/in/vlad-lee

GitHub: https://github.com/Vlad-Lee

Future Improvements
Add map clustering or density tuning
Build severity or cost prediction models
Incorporate causal analysis or experimentation frameworks
Improve species categorization (for example birds vs mammals)
