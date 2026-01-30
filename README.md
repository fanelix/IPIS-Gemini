# IPI Cumulative Displacement Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/cloud)

A professional Streamlit web application for visualizing In-Place Inclinometer (IPI) monitoring data used in geotechnical engineering.

![Dashboard Preview](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 Features

- **Auto-Detection**: Automatically parses Campbell Scientific TOA5 format files
- **Flexible Configuration**: Adjustable gauge length and sensor depth settings
- **Base Reading Correction**: Applies initial reading subtraction for accurate displacement
- **Bottom-Up Summation**: Calculates cumulative displacement assuming stable toe
- **Interactive Visualizations**: Four chart types with Plotly interactivity

## 📊 Dashboard Views

| View | Description |
|------|-------------|
| **Profile Plot** | Cumulative displacement vs depth (compare multiple timestamps) |
| **Trend Plot** | Displacement time history at selected depths |
| **Vector Plot** | A-axis vs B-axis showing movement direction |
| **Temperature Plot** | Thermistor readings over time |

## 🧮 Calculation Methods

### Incremental Displacement
```
d_inc = L × sin(θ)
```
Where `L` is gauge length and `θ` is tilt angle.

### Cumulative Displacement (Bottom-Up)
```
D_i = Σ(d_j) for j = i to n
```
Where `n` is the bottom sensor (assumed stable).

### Resultant Displacement
```
R = √(A² + B²)
```

## 📁 Supported File Formats

- Campbell Scientific TOA5 (.dat, .csv)
- Generic CSV with timestamp and sensor columns

### Expected Column Patterns

| Data Type | Column Pattern Example |
|-----------|----------------------|
| Timestamp | `TIMESTAMP` |
| A-Axis Tilt | `IPI_1_Tilt_A(1)`, `IPI_1_Tilt_A(2)`, ... |
| B-Axis Tilt | `IPI_1_Tilt_B(1)`, `IPI_1_Tilt_B(2)`, ... |
| A-Axis Deflection | `IPI_1_Def_A(1)`, `IPI_1_Def_A(2)`, ... |
| B-Axis Deflection | `IPI_1_Def_B(1)`, `IPI_1_Def_B(2)`, ... |
| Temperature | `IPI_1_Therm(1)`, `IPI_1_Therm(2)`, ... |

## 🚀 Quick Start

### Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ipi-dashboard.git
cd ipi-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Access via Browser

Open `http://localhost:8501` in your web browser.

## 📦 Dependencies

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
```

## 🖥️ Usage

1. **Upload Data**: Use the sidebar to upload your IPI data file (.dat or .csv)
2. **Configure Parameters**:
   - Set gauge length (default: 1.0m)
   - Set top sensor depth
   - Choose data source (raw tilt or pre-calculated deflection)
3. **Select Base Reading**: Choose the initial reading for correction
4. **Explore Charts**: Navigate between Profile, Trend, Vector, and Temperature views
5. **Export Data**: Download processed data as CSV

## 📐 Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| Gauge Length | 1.0 m | Distance between sensor nodes |
| Top Sensor Depth | 1.0 m | Depth of topmost sensor from ground |
| Data Source | Raw Tilt | Use sin(θ) values or pre-calculated deflection |
| Axis Selection | A-Axis | Display A, B, or Resultant displacement |

## 🏗️ Project Structure

```
ipi-dashboard/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── DEPLOYMENT_GUIDE.md # Step-by-step deployment instructions
```

## 📈 Sample Output

The dashboard processes IPI data to produce:
- Maximum displacement values per axis
- Depth location of maximum displacement
- Time-series trends for monitoring
- Vector direction analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Author

Geotechnical Data Analysis Team

## 🙏 Acknowledgments

- Campbell Scientific for TOA5 data format documentation
- Streamlit team for the excellent web framework
- Plotly for interactive visualization capabilities
