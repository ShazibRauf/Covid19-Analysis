# COVID-19 Analysis Dashboard

Welcome to my COVID-19 Analysis Dashboard project! This dashboard provides an interactive way to visualize COVID-19 data globally, using Plotly Dash.

## Features

- **Data Visualization:** View infection, recovery, and death rates for selected countries.
- **SIR Model Simulation:** Simulate the SIR (Susceptible, Infected, Recovered) model for chosen countries.
- **Interactive Interface:** Select multiple countries and different timelines for detailed analysis.

### Prerequisites

- Python 3.8
- pip

### Required Libraries

Install the necessary Python libraries:

```bash
pip install pandas numpy dash plotly scipy
```

### Usage

- Clone the repository or download the files to your local machine.
- Navigate to the directory containing the script and run python dashboard.py
- Open a web browser and navigate to http://127.0.0.1:8050/. The dashboard interface will be displayed.

### How to Use the Dashboard

**Selecting Countries**
Use the "Select Countries" dropdown menu to choose the countries you want to visualize. Multiple selections are allowed.

**Selecting Timeline**
Use the "Select Timeline" dropdown menu to choose the type of timeline to display:

- **Timeline Confirmed:** Shows the confirmed infection rates.
- **Timeline Confirmed Filtered:** Shows filtered confirmed infection rates.
- **Timeline Doubling Rate:** Shows the doubling rate of infections.
- **Timeline Doubling Rate Filtered:** Shows the filtered doubling rate of infections.

### Viewing the SIR Model
Select a country from the "SIR Model" dropdown to view the SIR model simulation for that country.
