"""
Multi-Point In-Place Inclinometer (IPI) Dashboard
==================================================
A comprehensive Streamlit web application for visualizing multiple IPI monitoring points.

Features:
- Support for multiple IPIS points (up to 20)
- Auto-detection of Campbell Scientific TOA5 format
- Per-point gauge length configuration (1m, 2m, 3m)
- Independent processing per IPIS point
- Comparative visualization across points
- Base reading correction and cumulative displacement calculation

Author: Geotechnical Data Analysis Team
Version: 2.0 - Multi-Point Support
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import hashlib

# =============================================================================
# CONSTANTS
# =============================================================================
MAX_IPIS_POINTS = 20
GAUGE_LENGTH_OPTIONS = [1.0, 2.0, 3.0]
DEFAULT_GAUGE_LENGTH = 3.0
DEFAULT_TOP_DEPTH = 1.0

# Default axis range settings
DEFAULT_AXIS_RANGE = {'auto': True, 'min': -50.0, 'max': 50.0}

# High contrast colors for data series
CHART_COLORS = [
    '#2563eb', '#dc2626', '#16a34a', '#9333ea', '#ea580c',
    '#0891b2', '#c026d3', '#4f46e5', '#059669', '#d97706',
    '#7c3aed', '#db2777', '#0d9488', '#ca8a04', '#6366f1',
    '#e11d48', '#14b8a6', '#f59e0b', '#8b5cf6', '#f43f5e'
]

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Multi-Point IPI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
    /* Main app - ensure dark text on light background */
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
    }
    
    /* Force dark text color globally */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div {
        color: #1e293b !important;
    }
    
    /* Main header styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1e40af !important;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem;
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-radius: 8px;
        border-bottom: 3px solid #2563eb;
    }
    
    /* Sub header */
    .sub-header {
        font-size: 1.1rem;
        color: #475569 !important;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    /* Point card styling */
    .point-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2563eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 0.5rem;
    }
    
    /* Sidebar styling - dark sidebar for contrast */
    section[data-testid="stSidebar"] {
        background-color: #1e293b !important;
    }
    
    section[data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span,
    section[data-testid="stSidebar"] .stMarkdown label,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f1f5f9 !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #60a5fa !important;
    }
    
    /* Main content area text */
    .main .block-container {
        color: #1e293b !important;
    }
    
    .main .block-container p,
    .main .block-container span,
    .main .block-container label,
    .main .block-container li {
        color: #374151 !important;
    }
    
    .main .block-container h1,
    .main .block-container h2,
    .main .block-container h3 {
        color: #1e40af !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #e2e8f0;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 6px;
        color: #1e293b !important;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #e2e8f0 !important;
        color: #1e293b !important;
        border-radius: 6px;
    }
    
    .streamlit-expanderHeader p {
        color: #1e293b !important;
        font-weight: 600;
    }
    
    .streamlit-expanderContent {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        color: #374151 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: none;
        border-radius: 6px;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #1d4ed8 !important;
        color: #ffffff !important;
    }
    
    /* Delete button */
    .delete-btn > button {
        background-color: #dc2626 !important;
    }
    
    /* Selectbox and input styling */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stDateInput > div > div > input {
        background-color: #ffffff !important;
        color: #1e293b !important;
        border: 1px solid #cbd5e1;
    }
    
    /* Multiselect */
    .stMultiSelect > div > div {
        background-color: #ffffff !important;
        color: #1e293b !important;
    }
    
    /* Info box */
    .stAlert {
        background-color: #dbeafe !important;
        color: #1e40af !important;
        border: 1px solid #93c5fd;
    }
    
    /* Metric values */
    [data-testid="stMetricValue"] {
        color: #1e40af !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #475569 !important;
    }
    
    /* Point counter badge */
    .point-counter {
        background-color: #2563eb;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    .point-counter-full {
        background-color: #dc2626;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class IPISPoint:
    """Data class representing a single IPIS monitoring point."""
    point_id: str
    name: str
    raw_df: pd.DataFrame
    metadata: Dict
    gauge_lengths: np.ndarray
    top_depth: float = DEFAULT_TOP_DEPTH
    base_reading_idx: int = 0
    num_sensors: int = 0
    detected_cols: Dict = field(default_factory=dict)
    processed_df: Optional[pd.DataFrame] = None
    color: str = '#2563eb'
    
    def __post_init__(self):
        """Initialize after dataclass creation."""
        if self.detected_cols:
            self.num_sensors = self.detected_cols.get('num_sensors', 0)


# =============================================================================
# DATA PARSING FUNCTIONS
# =============================================================================
def clean_and_split_lines(file_content: str) -> list:
    """Clean file content and handle concatenated lines."""
    content = file_content.replace('\r\n', '\n').replace('\r', '\n')
    lines = content.split('\n')
    cleaned_lines = []
    timestamp_pattern = r'"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"'
    
    for line in lines:
        timestamps = list(re.finditer(timestamp_pattern, line))
        if len(timestamps) > 1:
            last_end = 0
            for i, match in enumerate(timestamps):
                if i == 0:
                    continue
                split_point = match.start()
                segment = line[last_end:split_point].strip()
                if segment:
                    cleaned_lines.append(segment)
                last_end = split_point
            if last_end < len(line):
                segment = line[last_end:].strip()
                if segment:
                    cleaned_lines.append(segment)
        else:
            if line.strip():
                cleaned_lines.append(line.strip())
    
    return cleaned_lines


def parse_csv_line(line: str) -> list:
    """
    Parse a CSV line properly handling quoted fields with commas.
    This is critical for column names like "Tilt_A(1,1)" which contain commas.
    """
    fields = []
    current_field = ""
    in_quotes = False
    
    for char in line:
        if char == '"':
            in_quotes = not in_quotes
        elif char == ',' and not in_quotes:
            fields.append(current_field.strip().strip('"'))
            current_field = ""
        else:
            current_field += char
    
    fields.append(current_field.strip().strip('"'))
    return fields


def parse_wrapped_column_header(line: str) -> list:
    """
    Parse a column header line that may be wrapped in outer quotes with 
    double-escaped inner quotes.
    
    Format 3 (Newest): The entire header row is wrapped in outer quotes
    with inner quotes doubled for escaping (e.g., Tilt_A columns).
    
    This function detects this format and properly parses it.
    """
    line = line.strip()
    
    # Check if line is wrapped in outer quotes (Format 3)
    if line.startswith('"') and line.endswith('"') and '""' in line:
        # Remove outer quotes
        inner = line[1:-1]
        # Replace doubled quotes with single quotes
        inner = inner.replace('""', '"')
        # Now parse normally
        return parse_csv_line(inner)
    else:
        # Standard format - parse directly
        return parse_csv_line(line)


def parse_toa5_file(file_content: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Parse Campbell Scientific TOA5 format file.
    
    Supports multiple column naming formats:
    - Format 1 (Old): IPIS_Tilt_A(N) with simple (N) indexing
    - Format 2 (New): Tilt_A(1,N) with 2D array indexing, standard CSV
    - Format 3 (Newest): Tilt_A(1,N) with wrapped-quote header line
    """
    lines = clean_and_split_lines(file_content)
    
    if len(lines) < 5:
        raise ValueError("File appears to be too short or corrupted")
    
    # Parse header using proper CSV parsing (commas inside quotes)
    header_info = parse_csv_line(lines[0])
    metadata = {
        'format': header_info[0] if len(header_info) > 0 else 'Unknown',
        'station_name': header_info[1] if len(header_info) > 1 else 'Unknown',
        'logger_model': header_info[2] if len(header_info) > 2 else 'Unknown',
        'serial_number': header_info[3] if len(header_info) > 3 else 'Unknown',
        'program_name': header_info[5] if len(header_info) > 5 else 'Unknown',
        'table_name': header_info[7] if len(header_info) > 7 else 'Unknown'
    }
    
    # Parse column names - handle both standard and wrapped-quote formats
    # This correctly handles columns like "Tilt_A(1,1)" which contain commas
    columns = parse_wrapped_column_header(lines[1])
    expected_fields = len(columns)
    
    # Parse data (skip header rows)
    data_lines = lines[4:]
    valid_rows = []
    skipped_rows = 0
    
    for line in data_lines:
        try:
            fields = parse_csv_line(line)
            
            if len(fields) == expected_fields:
                valid_rows.append(fields)
            elif len(fields) > expected_fields:
                valid_rows.append(fields[:expected_fields])
                skipped_rows += 1
            else:
                fields.extend([np.nan] * (expected_fields - len(fields)))
                valid_rows.append(fields)
                skipped_rows += 1
        except Exception:
            skipped_rows += 1
            continue
    
    if not valid_rows:
        raise ValueError("No valid data rows found in file")
    
    df = pd.DataFrame(valid_rows, columns=columns)
    
    # Convert numeric columns
    for col in df.columns:
        if col not in ['TIMESTAMP']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Parse timestamp
    if 'TIMESTAMP' in df.columns:
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce')
        df = df.dropna(subset=['TIMESTAMP'])
        df = df.sort_values('TIMESTAMP').reset_index(drop=True)
    
    metadata['skipped_rows'] = skipped_rows
    metadata['total_rows'] = len(valid_rows)
    
    return df, metadata


def detect_ipi_columns(df: pd.DataFrame) -> Dict:
    """
    Auto-detect IPI sensor columns in dataframe.
    
    Supports two column naming formats:
    
    Format 1 (Old): IPIS_Tilt_A(N), IPIS_Tilt_B(N), IPIS_Therm(N), IPIS_Def_A(N), IPIS_Def_B(N)
                    where N = 1, 2, 3, ...
    
    Format 2 (New): Tilt_A(1,N), Tilt_B(1,N), IPI_Temp(1,N), IPI_Def_A(1,N), IPI_Def_B(1,N)
                    where N = 1, 2, 3, ... (uses 2D array notation with first index = 1)
    """
    columns = df.columns.tolist()
    
    detected = {
        'timestamp': None,
        'tilt_a': [],
        'tilt_b': [],
        'def_a': [],
        'def_b': [],
        'therm': [],
        'battery': None,
        'panel_temp': None,
        'int_temp': None,
        'num_sensors': 0,
        'format_type': 'unknown'  # Track which format was detected
    }
    
    # Detect format type first
    has_2d_notation = any('(1,' in col for col in columns)
    detected['format_type'] = 'new_2d' if has_2d_notation else 'old_1d'
    
    for col in columns:
        col_lower = col.lower()
        
        # Timestamp detection
        if 'timestamp' in col_lower or col_lower == 'ts':
            detected['timestamp'] = col
        
        # Battery voltage
        elif 'batt' in col_lower and 'volt' in col_lower:
            detected['battery'] = col
        elif col_lower == 'battv':
            detected['battery'] = col
        
        # Panel/Internal temperature
        elif 'ptemp' in col_lower:
            detected['panel_temp'] = col
        elif col_lower == 'int_temp':
            detected['int_temp'] = col
        
        # Tilt A columns - both formats
        # Old: IPIS_Tilt_A(N) or tilt_a(N)
        # New: Tilt_A(1,N)
        elif 'tilt_a' in col_lower or 'tilt_a(' in col_lower:
            detected['tilt_a'].append(col)
        
        # Tilt B columns
        elif 'tilt_b' in col_lower or 'tilt_b(' in col_lower:
            detected['tilt_b'].append(col)
        
        # Deflection A columns
        # Old: IPIS_Def_A(N) or def_a(N)
        # New: IPI_Def_A(1,N)
        elif 'def_a' in col_lower or 'ipi_def_a' in col_lower:
            detected['def_a'].append(col)
        
        # Deflection B columns
        elif 'def_b' in col_lower or 'ipi_def_b' in col_lower:
            detected['def_b'].append(col)
        
        # Temperature columns
        # Old: IPIS_Therm(N) or therm(N)
        # New: IPI_Temp(1,N)
        elif ('therm' in col_lower and 'ptemp' not in col_lower) or 'ipi_temp' in col_lower:
            detected['therm'].append(col)
    
    # Sort columns by sensor number - handle both formats
    def extract_number(col_name):
        """Extract sensor number from column name, handling both formats."""
        # Try 2D notation first: (1,N)
        match_2d = re.search(r'\(1,(\d+)\)', col_name)
        if match_2d:
            return int(match_2d.group(1))
        # Fall back to 1D notation: (N)
        match_1d = re.search(r'\((\d+)\)', col_name)
        if match_1d:
            return int(match_1d.group(1))
        return 0
    
    for key in ['tilt_a', 'tilt_b', 'def_a', 'def_b', 'therm']:
        detected[key] = sorted(detected[key], key=extract_number)
    
    detected['num_sensors'] = max(
        len(detected['tilt_a']),
        len(detected['tilt_b']),
        len(detected['def_a']),
        len(detected['def_b'])
    )
    
    return detected


def generate_point_id(content: str) -> str:
    """Generate unique ID for IPIS point based on file content."""
    return hashlib.md5(content.encode()).hexdigest()[:8]


# =============================================================================
# DISPLACEMENT CALCULATIONS
# =============================================================================
def calculate_incremental_displacement(tilt_sin: float, gauge_length: float) -> float:
    """Calculate incremental displacement from tilt (sin θ) and gauge length."""
    if pd.isna(tilt_sin):
        return np.nan
    return tilt_sin * gauge_length * 1000  # Convert to mm


def calculate_cumulative_displacement(incremental_displacements: np.ndarray, from_bottom: bool = True) -> np.ndarray:
    """Calculate cumulative displacement from incremental values."""
    if from_bottom:
        return np.flip(np.nancumsum(np.flip(incremental_displacements)))
    else:
        return np.nancumsum(incremental_displacements)


def process_ipis_point(point: IPISPoint, use_raw_tilt: bool = True) -> pd.DataFrame:
    """Process a single IPIS point to calculate cumulative displacements."""
    df = point.raw_df
    detected_cols = point.detected_cols
    gauge_lengths = point.gauge_lengths
    top_depth = point.top_depth
    base_reading_idx = point.base_reading_idx
    
    num_sensors = detected_cols['num_sensors']
    
    # Ensure gauge_lengths matches num_sensors
    if len(gauge_lengths) != num_sensors:
        if len(gauge_lengths) < num_sensors:
            gauge_lengths = np.concatenate([
                gauge_lengths, 
                np.full(num_sensors - len(gauge_lengths), gauge_lengths[-1] if len(gauge_lengths) > 0 else DEFAULT_GAUGE_LENGTH)
            ])
        else:
            gauge_lengths = gauge_lengths[:num_sensors]
    
    # Calculate depths based on cumulative gauge lengths
    depths = np.zeros(num_sensors)
    depths[0] = top_depth
    for i in range(1, num_sensors):
        depths[i] = depths[i-1] + gauge_lengths[i-1]
    
    results = []
    
    for idx, row in df.iterrows():
        timestamp = row[detected_cols['timestamp']]
        
        # Extract tilt data
        if use_raw_tilt and detected_cols['tilt_a'] and detected_cols['tilt_b']:
            tilt_a = np.array([row[col] for col in detected_cols['tilt_a']])
            tilt_b = np.array([row[col] for col in detected_cols['tilt_b']])
            
            inc_a = np.array([calculate_incremental_displacement(tilt_a[i], gauge_lengths[i]) 
                            for i in range(min(len(tilt_a), num_sensors))])
            inc_b = np.array([calculate_incremental_displacement(tilt_b[i], gauge_lengths[i]) 
                            for i in range(min(len(tilt_b), num_sensors))])
        elif detected_cols['def_a'] and detected_cols['def_b']:
            inc_a = np.array([row[col] for col in detected_cols['def_a']])
            inc_b = np.array([row[col] for col in detected_cols['def_b']])
        else:
            continue
        
        # Get temperature
        if detected_cols['therm']:
            temps = np.array([row[col] for col in detected_cols['therm'][:num_sensors]])
        else:
            temps = np.full(num_sensors, np.nan)
        
        for i in range(num_sensors):
            results.append({
                'point_id': point.point_id,
                'point_name': point.name,
                'timestamp': timestamp,
                'record_idx': idx,
                'sensor_num': i + 1,
                'depth': depths[i],
                'gauge_length': gauge_lengths[i],
                'inc_disp_a': inc_a[i] if i < len(inc_a) else np.nan,
                'inc_disp_b': inc_b[i] if i < len(inc_b) else np.nan,
                'temperature': temps[i] if i < len(temps) else np.nan
            })
    
    processed_df = pd.DataFrame(results)
    
    if processed_df.empty:
        return processed_df
    
    # Apply base reading correction
    base_data = processed_df[processed_df['record_idx'] == base_reading_idx].copy()
    base_data = base_data.set_index('sensor_num')[['inc_disp_a', 'inc_disp_b']].rename(
        columns={'inc_disp_a': 'base_a', 'inc_disp_b': 'base_b'}
    )
    
    processed_df = processed_df.merge(base_data, left_on='sensor_num', right_index=True, how='left')
    processed_df['inc_disp_a_corr'] = processed_df['inc_disp_a'] - processed_df['base_a']
    processed_df['inc_disp_b_corr'] = processed_df['inc_disp_b'] - processed_df['base_b']
    
    # Calculate cumulative displacement
    cum_disp_a_list = []
    cum_disp_b_list = []
    
    for timestamp in processed_df['timestamp'].unique():
        mask = processed_df['timestamp'] == timestamp
        inc_a = processed_df.loc[mask, 'inc_disp_a_corr'].values
        inc_b = processed_df.loc[mask, 'inc_disp_b_corr'].values
        
        cum_a = calculate_cumulative_displacement(inc_a, from_bottom=True)
        cum_b = calculate_cumulative_displacement(inc_b, from_bottom=True)
        
        cum_disp_a_list.extend(cum_a)
        cum_disp_b_list.extend(cum_b)
    
    processed_df['cum_disp_a'] = cum_disp_a_list
    processed_df['cum_disp_b'] = cum_disp_b_list
    processed_df['cum_disp_resultant'] = np.sqrt(
        processed_df['cum_disp_a']**2 + processed_df['cum_disp_b']**2
    )
    
    return processed_df


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def create_profile_plot_single(processed_df: pd.DataFrame, selected_timestamps: list, point_name: str,
                                axis_range: Optional[Dict] = None) -> go.Figure:
    """
    Create profile plot for a single IPIS point.
    
    Args:
        processed_df: Processed dataframe with displacement data
        selected_timestamps: List of timestamps to plot
        point_name: Name of the IPIS point
        axis_range: Optional dict with 'auto', 'min', 'max' for X-axis range
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('<b>A-Axis Displacement</b>', '<b>B-Axis Displacement</b>'),
        shared_yaxes=True,
        horizontal_spacing=0.10
    )
    
    for i, timestamp in enumerate(selected_timestamps):
        mask = processed_df['timestamp'] == timestamp
        data = processed_df[mask].sort_values('depth')
        
        color = CHART_COLORS[i % len(CHART_COLORS)]
        ts_str = pd.Timestamp(timestamp).strftime('%Y-%m-%d %H:%M')
        
        fig.add_trace(
            go.Scatter(
                x=data['cum_disp_a'], y=data['depth'],
                mode='lines+markers', name=f'{ts_str}',
                line=dict(color=color, width=2.5),
                marker=dict(size=7),
                legendgroup=f'group{i}', showlegend=True,
                hovertemplate='<b>Depth:</b> %{y:.2f} m<br><b>A-Axis:</b> %{x:.3f} mm<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data['cum_disp_b'], y=data['depth'],
                mode='lines+markers', name=f'{ts_str}',
                line=dict(color=color, width=2.5),
                marker=dict(size=7),
                legendgroup=f'group{i}', showlegend=False,
                hovertemplate='<b>Depth:</b> %{y:.2f} m<br><b>B-Axis:</b> %{x:.3f} mm<extra></extra>'
            ),
            row=1, col=2
        )
    
    fig.add_vline(x=0, line_dash="dash", line_color="#64748b", line_width=1.5, row=1, col=1)
    fig.add_vline(x=0, line_dash="dash", line_color="#64748b", line_width=1.5, row=1, col=2)
    
    fig.update_layout(
        title=dict(
            text=f'<b>{point_name} - Cumulative Displacement Profile</b>',
            font=dict(size=16, color='#1e293b'),
            x=0.5, xanchor='center', y=0.95
        ),
        legend=dict(
            orientation='h', yanchor='top', y=-0.12,
            xanchor='center', x=0.5,
            title=dict(text='<b>Timestamp:</b> ', font=dict(size=10)),
            bgcolor='#f8fafc', bordercolor='#cbd5e1', borderwidth=1,
            font=dict(size=9, color='#1e293b')
        ),
        plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
        height=600, margin=dict(t=60, b=80, l=70, r=50)
    )
    
    for annotation in fig['layout']['annotations']:
        annotation['y'] = 1.02
        annotation['font'] = dict(size=12, color='#1e293b')
    
    axis_style = dict(
        title_font=dict(size=11, color='#374151'),
        tickfont=dict(size=9, color='#4b5563'),
        gridcolor='#e5e7eb', linecolor='#d1d5db',
        linewidth=1, showline=True, mirror=True
    )
    
    # Apply axis range if specified (not auto)
    x_axis_config = dict(title_text='Displacement (mm)', zeroline=True, zerolinecolor='#9ca3af', **axis_style)
    if axis_range and not axis_range.get('auto', True):
        x_axis_config['range'] = [axis_range['min'], axis_range['max']]
    
    fig.update_xaxes(**x_axis_config, row=1, col=1)
    fig.update_xaxes(**x_axis_config, row=1, col=2)
    fig.update_yaxes(title_text='Depth (m)', autorange='reversed', **axis_style, row=1, col=1)
    fig.update_yaxes(autorange='reversed', **axis_style, row=1, col=2)
    
    return fig


def create_profile_plot_comparison(points_data: Dict[str, pd.DataFrame], selected_timestamp, axis: str = 'A',
                                    axis_range: Optional[Dict] = None) -> go.Figure:
    """
    Create comparative profile plot across multiple IPIS points.
    
    Args:
        points_data: Dict mapping point names to their processed dataframes
        selected_timestamp: Timestamp to compare
        axis: 'A' or 'B' axis
        axis_range: Optional dict with 'auto', 'min', 'max' for X-axis range
    """
    fig = go.Figure()
    
    disp_col = 'cum_disp_a' if axis == 'A' else 'cum_disp_b'
    
    for i, (point_name, df) in enumerate(points_data.items()):
        mask = df['timestamp'] == selected_timestamp
        data = df[mask].sort_values('depth')
        
        if data.empty:
            continue
        
        color = CHART_COLORS[i % len(CHART_COLORS)]
        
        fig.add_trace(go.Scatter(
            x=data[disp_col], y=data['depth'],
            mode='lines+markers', name=point_name,
            line=dict(color=color, width=2.5),
            marker=dict(size=7),
            hovertemplate=f'<b>{point_name}</b><br>Depth: %{{y:.2f}} m<br>{axis}-Axis: %{{x:.3f}} mm<extra></extra>'
        ))
    
    fig.add_vline(x=0, line_dash="dash", line_color="#64748b", line_width=1.5)
    
    ts_str = pd.Timestamp(selected_timestamp).strftime('%Y-%m-%d %H:%M')
    
    # Build x-axis config
    x_axis_config = dict(
        title='Cumulative Displacement (mm)',
        gridcolor='#e5e7eb', linecolor='#d1d5db',
        zeroline=True, zerolinecolor='#9ca3af'
    )
    if axis_range and not axis_range.get('auto', True):
        x_axis_config['range'] = [axis_range['min'], axis_range['max']]
    
    fig.update_layout(
        title=dict(
            text=f'<b>Multi-Point {axis}-Axis Comparison</b><br><sub>{ts_str}</sub>',
            font=dict(size=16, color='#1e293b'),
            x=0.5, xanchor='center'
        ),
        xaxis=x_axis_config,
        yaxis=dict(
            title='Depth (m)', autorange='reversed',
            gridcolor='#e5e7eb', linecolor='#d1d5db'
        ),
        legend=dict(
            orientation='h', yanchor='top', y=-0.15,
            xanchor='center', x=0.5,
            bgcolor='#f8fafc', bordercolor='#cbd5e1', borderwidth=1
        ),
        plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
        height=600, margin=dict(t=80, b=100, l=70, r=50)
    )
    
    return fig


def create_trend_plot_single(processed_df: pd.DataFrame, selected_depths: list, point_name: str) -> go.Figure:
    """Create trend plot for a single IPIS point."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('<b>A-Axis Time History</b>', '<b>B-Axis Time History</b>'),
        horizontal_spacing=0.10
    )
    
    all_depths = sorted(processed_df['depth'].unique())
    
    for i, depth in enumerate(selected_depths):
        closest_depth = min(all_depths, key=lambda x: abs(x - depth))
        mask = processed_df['depth'] == closest_depth
        data = processed_df[mask].sort_values('timestamp')
        
        color = CHART_COLORS[i % len(CHART_COLORS)]
        
        fig.add_trace(
            go.Scatter(
                x=data['timestamp'], y=data['cum_disp_a'],
                mode='lines+markers', name=f'{closest_depth:.1f}m',
                line=dict(color=color, width=2),
                marker=dict(size=4),
                legendgroup=f'depth{i}', showlegend=True,
                hovertemplate='<b>Time:</b> %{x}<br><b>A-Axis:</b> %{y:.3f} mm<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data['timestamp'], y=data['cum_disp_b'],
                mode='lines+markers', name=f'{closest_depth:.1f}m',
                line=dict(color=color, width=2),
                marker=dict(size=4),
                legendgroup=f'depth{i}', showlegend=False,
                hovertemplate='<b>Time:</b> %{x}<br><b>B-Axis:</b> %{y:.3f} mm<extra></extra>'
            ),
            row=1, col=2
        )
    
    fig.add_hline(y=0, line_dash="dash", line_color="#64748b", line_width=1.5, row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#64748b", line_width=1.5, row=1, col=2)
    
    fig.update_layout(
        title=dict(
            text=f'<b>{point_name} - Displacement Time History</b>',
            font=dict(size=16, color='#1e293b'),
            x=0.5, xanchor='center', y=0.95
        ),
        legend=dict(
            orientation='h', yanchor='top', y=-0.15,
            xanchor='center', x=0.5,
            title=dict(text='<b>Depth:</b> ', font=dict(size=10)),
            bgcolor='#f8fafc', bordercolor='#cbd5e1', borderwidth=1
        ),
        plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
        height=450, margin=dict(t=60, b=80, l=70, r=50),
        hovermode='x unified'
    )
    
    for annotation in fig['layout']['annotations']:
        annotation['y'] = 1.02
        annotation['font'] = dict(size=12, color='#1e293b')
    
    axis_style = dict(
        title_font=dict(size=11, color='#374151'),
        tickfont=dict(size=9, color='#4b5563'),
        gridcolor='#e5e7eb', linecolor='#d1d5db'
    )
    
    fig.update_xaxes(title_text='Date/Time', **axis_style, row=1, col=1)
    fig.update_xaxes(title_text='Date/Time', **axis_style, row=1, col=2)
    fig.update_yaxes(title_text='Displacement (mm)', zeroline=True, zerolinecolor='#9ca3af', **axis_style, row=1, col=1)
    fig.update_yaxes(title_text='Displacement (mm)', zeroline=True, zerolinecolor='#9ca3af', **axis_style, row=1, col=2)
    
    return fig


def create_trend_comparison(points_data: Dict[str, pd.DataFrame], selected_depth: float, axis: str = 'A') -> go.Figure:
    """Create comparative trend plot across multiple points at a specific depth."""
    fig = go.Figure()
    
    disp_col = 'cum_disp_a' if axis == 'A' else 'cum_disp_b'
    
    for i, (point_name, df) in enumerate(points_data.items()):
        all_depths = sorted(df['depth'].unique())
        closest_depth = min(all_depths, key=lambda x: abs(x - selected_depth))
        
        mask = df['depth'] == closest_depth
        data = df[mask].sort_values('timestamp')
        
        color = CHART_COLORS[i % len(CHART_COLORS)]
        
        fig.add_trace(go.Scatter(
            x=data['timestamp'], y=data[disp_col],
            mode='lines+markers', name=f'{point_name} ({closest_depth:.1f}m)',
            line=dict(color=color, width=2),
            marker=dict(size=4),
            hovertemplate=f'<b>{point_name}</b><br>Time: %{{x}}<br>{axis}-Axis: %{{y:.3f}} mm<extra></extra>'
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="#64748b", line_width=1.5)
    
    fig.update_layout(
        title=dict(
            text=f'<b>Multi-Point {axis}-Axis Comparison @ ~{selected_depth:.1f}m Depth</b>',
            font=dict(size=16, color='#1e293b'),
            x=0.5, xanchor='center'
        ),
        xaxis=dict(title='Date/Time', gridcolor='#e5e7eb'),
        yaxis=dict(
            title='Cumulative Displacement (mm)',
            gridcolor='#e5e7eb',
            zeroline=True, zerolinecolor='#9ca3af'
        ),
        legend=dict(
            orientation='h', yanchor='top', y=-0.15,
            xanchor='center', x=0.5,
            bgcolor='#f8fafc', bordercolor='#cbd5e1', borderwidth=1
        ),
        plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
        height=450, margin=dict(t=60, b=100, l=70, r=50),
        hovermode='x unified'
    )
    
    return fig


# =============================================================================
# SESSION STATE MANAGEMENT
# =============================================================================
def init_session_state():
    """Initialize session state variables."""
    if 'ipis_points' not in st.session_state:
        st.session_state.ipis_points = {}  # Dict[point_id, IPISPoint]
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = {}  # Dict[point_id, pd.DataFrame]
    # Axis range settings for profile plots
    if 'axis_range' not in st.session_state:
        st.session_state.axis_range = {
            'auto': True,
            'min': -50.0,
            'max': 50.0
        }


def add_ipis_point(file_content: str, filename: str) -> Tuple[bool, str]:
    """Add a new IPIS point from file content."""
    # Check limit
    if len(st.session_state.ipis_points) >= MAX_IPIS_POINTS:
        return False, f"Maximum limit of {MAX_IPIS_POINTS} IPIS points reached. Please remove a point before adding new ones."
    
    try:
        # Parse file
        df, metadata = parse_toa5_file(file_content)
        
        # Detect columns
        detected_cols = detect_ipi_columns(df)
        
        if detected_cols['num_sensors'] == 0:
            return False, f"Could not detect IPI sensor columns in {filename}"
        
        # Generate unique ID
        point_id = generate_point_id(file_content)
        
        # Check for duplicates
        if point_id in st.session_state.ipis_points:
            return False, f"This file appears to already be loaded (duplicate detected)"
        
        # Create point name from metadata
        point_name = metadata.get('station_name', filename.replace('.dat', '').replace('.DAT', ''))
        
        # Assign color
        color_idx = len(st.session_state.ipis_points) % len(CHART_COLORS)
        
        # Initialize gauge lengths
        num_sensors = detected_cols['num_sensors']
        gauge_lengths = np.full(num_sensors, DEFAULT_GAUGE_LENGTH)
        
        # Create IPIS point
        point = IPISPoint(
            point_id=point_id,
            name=point_name,
            raw_df=df,
            metadata=metadata,
            gauge_lengths=gauge_lengths,
            detected_cols=detected_cols,
            num_sensors=num_sensors,
            color=CHART_COLORS[color_idx]
        )
        
        # Store point
        st.session_state.ipis_points[point_id] = point
        
        # Format info for user
        format_desc = "2D array format" if detected_cols.get('format_type') == 'new_2d' else "standard format"
        return True, f"Successfully loaded: {point_name} ({num_sensors} sensors, {len(df)} records, {format_desc})"
        
    except Exception as e:
        return False, f"Error parsing {filename}: {str(e)}"


def remove_ipis_point(point_id: str):
    """Remove an IPIS point."""
    if point_id in st.session_state.ipis_points:
        del st.session_state.ipis_points[point_id]
    if point_id in st.session_state.processed_data:
        del st.session_state.processed_data[point_id]


def process_all_points(use_raw_tilt: bool = True):
    """Process all IPIS points."""
    for point_id, point in st.session_state.ipis_points.items():
        processed_df = process_ipis_point(point, use_raw_tilt)
        st.session_state.processed_data[point_id] = processed_df
        point.processed_df = processed_df


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    init_session_state()
    
    # Header
    st.markdown('<div class="main-header">📊 Multi-Point IPI Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">In-Place Inclinometer Monitoring - Multiple Points Analysis</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Point counter
        num_points = len(st.session_state.ipis_points)
        counter_class = "point-counter-full" if num_points >= MAX_IPIS_POINTS else ""
        st.markdown(f'<span class="point-counter {counter_class}">{num_points} / {MAX_IPIS_POINTS} Points</span>', unsafe_allow_html=True)
        
        st.divider()
        
        # File upload section
        st.subheader("1. Upload IPIS Data Files")
        
        if num_points >= MAX_IPIS_POINTS:
            st.error(f"⚠️ Maximum limit of {MAX_IPIS_POINTS} points reached!")
            st.info("Remove existing points to add new ones.")
        else:
            uploaded_files = st.file_uploader(
                "Upload .DAT Files",
                type=['dat', 'csv'],
                accept_multiple_files=True,
                help=f"Upload Campbell Scientific TOA5 format files. Max {MAX_IPIS_POINTS} total points.",
                key="file_uploader"
            )
            
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    if len(st.session_state.ipis_points) >= MAX_IPIS_POINTS:
                        st.warning(f"Skipped {uploaded_file.name}: Maximum points reached")
                        continue
                    
                    file_content = uploaded_file.read().decode('utf-8')
                    success, message = add_ipis_point(file_content, uploaded_file.name)
                    
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
        
        st.divider()
        
        # Processing options
        st.subheader("2. Processing Options")
        
        data_source = st.radio(
            "Data Source",
            options=['Raw Tilt (sin θ)', 'Pre-calculated Deflection'],
            index=0,
            help="Select data source for displacement calculation"
        )
        use_raw_tilt = data_source == 'Raw Tilt (sin θ)'
        
        st.divider()
        
        # Axis Range Configuration
        st.subheader("3. Plot Axis Settings")
        
        auto_range = st.checkbox(
            "Auto X-Axis Range",
            value=st.session_state.axis_range.get('auto', True),
            help="Automatically scale X-axis based on data"
        )
        st.session_state.axis_range['auto'] = auto_range
        
        if not auto_range:
            col_min, col_max = st.columns(2)
            with col_min:
                axis_min = st.number_input(
                    "X-Axis Min (mm)",
                    value=float(st.session_state.axis_range.get('min', -50.0)),
                    step=5.0,
                    key="axis_min"
                )
                st.session_state.axis_range['min'] = axis_min
            with col_max:
                axis_max = st.number_input(
                    "X-Axis Max (mm)",
                    value=float(st.session_state.axis_range.get('max', 50.0)),
                    step=5.0,
                    key="axis_max"
                )
                st.session_state.axis_range['max'] = axis_max
            
            # Quick presets
            st.caption("Quick Presets:")
            preset_cols = st.columns(4)
            with preset_cols[0]:
                if st.button("±25", key="preset_25", use_container_width=True):
                    st.session_state.axis_range['min'] = -25.0
                    st.session_state.axis_range['max'] = 25.0
                    st.rerun()
            with preset_cols[1]:
                if st.button("±50", key="preset_50", use_container_width=True):
                    st.session_state.axis_range['min'] = -50.0
                    st.session_state.axis_range['max'] = 50.0
                    st.rerun()
            with preset_cols[2]:
                if st.button("±100", key="preset_100", use_container_width=True):
                    st.session_state.axis_range['min'] = -100.0
                    st.session_state.axis_range['max'] = 100.0
                    st.rerun()
            with preset_cols[3]:
                if st.button("±200", key="preset_200", use_container_width=True):
                    st.session_state.axis_range['min'] = -200.0
                    st.session_state.axis_range['max'] = 200.0
                    st.rerun()
        
        st.divider()
        
        # Loaded points management
        st.subheader("4. Loaded Points")
        
        if st.session_state.ipis_points:
            for point_id, point in list(st.session_state.ipis_points.items()):
                with st.expander(f"📍 {point.name}", expanded=False):
                    # Show format type and basic info
                    format_type = point.detected_cols.get('format_type', 'unknown')
                    format_badge = "🔷 2D Array" if format_type == 'new_2d' else "🔶 Standard"
                    st.caption(f"{format_badge} | Sensors: {point.num_sensors} | Records: {len(point.raw_df)}")
                    
                    # =============================================
                    # GAUGE LENGTH CONFIGURATION
                    # =============================================
                    st.markdown("**⚙️ Gauge Length Configuration**")
                    
                    # Show current gauge lengths summary
                    unique_gauges = np.unique(point.gauge_lengths)
                    if len(unique_gauges) == 1:
                        gauge_summary = f"All sensors: `{unique_gauges[0]:.0f} m`"
                    else:
                        gauge_summary = f"Mixed: {', '.join([f'{g:.0f}m' for g in unique_gauges])}"
                    st.caption(f"Current: {gauge_summary}")
                    
                    # Quick set ALL sensors buttons
                    st.markdown("**Set ALL Sensors:**")
                    qcol1, qcol2, qcol3 = st.columns(3)
                    with qcol1:
                        if st.button("All 1m", key=f"all1_{point_id}", use_container_width=True):
                            point.gauge_lengths = np.full(point.num_sensors, 1.0)
                            if point_id in st.session_state.processed_data:
                                del st.session_state.processed_data[point_id]
                            st.rerun()
                    with qcol2:
                        if st.button("All 2m", key=f"all2_{point_id}", use_container_width=True):
                            point.gauge_lengths = np.full(point.num_sensors, 2.0)
                            if point_id in st.session_state.processed_data:
                                del st.session_state.processed_data[point_id]
                            st.rerun()
                    with qcol3:
                        if st.button("All 3m", key=f"all3_{point_id}", use_container_width=True):
                            point.gauge_lengths = np.full(point.num_sensors, 3.0)
                            if point_id in st.session_state.processed_data:
                                del st.session_state.processed_data[point_id]
                            st.rerun()
                    
                    # Per-sensor gauge length configuration
                    st.markdown("**Per-Sensor Gauge Length:**")
                    
                    # Calculate depths for display
                    depths = np.zeros(point.num_sensors)
                    depths[0] = point.top_depth
                    for i in range(1, point.num_sensors):
                        depths[i] = depths[i-1] + point.gauge_lengths[i-1]
                    
                    # Create a compact grid for sensor gauge lengths
                    # Display in rows of 4 sensors each
                    sensors_per_row = 4
                    num_rows = (point.num_sensors + sensors_per_row - 1) // sensors_per_row
                    
                    gauge_changed = False
                    new_gauge_lengths = point.gauge_lengths.copy()
                    
                    for row_idx in range(num_rows):
                        cols = st.columns(sensors_per_row)
                        for col_idx, col in enumerate(cols):
                            sensor_idx = row_idx * sensors_per_row + col_idx
                            if sensor_idx < point.num_sensors:
                                with col:
                                    current_gauge = point.gauge_lengths[sensor_idx]
                                    # Compact sensor label with depth
                                    sensor_label = f"S{sensor_idx + 1}"
                                    depth_label = f"({depths[sensor_idx]:.1f}m)"
                                    
                                    new_gauge = st.selectbox(
                                        f"{sensor_label} {depth_label}",
                                        options=GAUGE_LENGTH_OPTIONS,
                                        index=GAUGE_LENGTH_OPTIONS.index(current_gauge) if current_gauge in GAUGE_LENGTH_OPTIONS else 2,
                                        format_func=lambda x: f"{int(x)}m",
                                        key=f"gs_{point_id}_{sensor_idx}",
                                        label_visibility="visible"
                                    )
                                    
                                    if new_gauge != current_gauge:
                                        new_gauge_lengths[sensor_idx] = new_gauge
                                        gauge_changed = True
                    
                    # Apply changes if any gauge was modified
                    if gauge_changed:
                        point.gauge_lengths = new_gauge_lengths
                        if point_id in st.session_state.processed_data:
                            del st.session_state.processed_data[point_id]
                    
                    st.divider()
                    
                    # Top depth
                    point.top_depth = st.number_input(
                        "Top Depth (m)",
                        min_value=0.0, max_value=100.0,
                        value=float(point.top_depth),
                        step=0.5,
                        key=f"td_{point_id}"
                    )
                    
                    # Base reading selection - show ALL available timestamps
                    timestamps = point.raw_df[point.detected_cols['timestamp']].sort_values().unique()
                    base_options = [pd.Timestamp(ts).strftime('%Y-%m-%d %H:%M') for ts in timestamps]
                    
                    st.markdown(f"**Base Reading** ({len(base_options)} timestamps available)")
                    
                    selected_base = st.selectbox(
                        "Base Reading",
                        options=base_options,
                        index=0,
                        key=f"base_{point_id}",
                        label_visibility="collapsed"
                    )
                    point.base_reading_idx = base_options.index(selected_base)
                    
                    # Remove button
                    if st.button("🗑️ Remove", key=f"del_{point_id}", type="secondary"):
                        remove_ipis_point(point_id)
                        st.rerun()
        else:
            st.info("No points loaded. Upload .DAT files above.")
    
    # Main content
    if not st.session_state.ipis_points:
        st.info("👆 Upload IPIS data files using the sidebar to begin analysis.")
        
        with st.expander("📖 About This Dashboard", expanded=True):
            st.markdown("""
            ### Multi-Point IPI Dashboard Features
            
            - **Multiple IPIS Points**: Upload up to 20 different monitoring points
            - **Independent Processing**: Each point has its own gauge length and base reading settings
            - **Per-IPIS Gauge Length**: Configure 1m, 2m, or 3m gauge lengths individually per dataset
            - **Configurable Axis Range**: Set custom X-axis limits for displacement plots
            - **Dual Format Support**: Supports both standard and 2D array column formats
            - **Comparative Analysis**: Compare displacement profiles across multiple points
            - **Auto-detection**: Automatically detects Campbell Scientific TOA5 format
            
            ### How to Use
            
            1. Upload one or more `.DAT` files using the sidebar
            2. Configure gauge length (1m, 2m, 3m) for each IPIS point
            3. Adjust X-axis range settings if needed (auto or manual)
            4. Set base reading and top depth for each point
            5. Use the tabs to view individual or comparative plots
            
            ### Supported File Formats
            - Campbell Scientific TOA5 (.dat, .csv)
            - Standard format: `IPIS_Tilt_A(N)` column naming
            - 2D Array format: `Tilt_A(1,N)` column naming
            """)
        return
    
    # Process all points
    process_all_points(use_raw_tilt)
    
    # Check if we have processed data
    if not st.session_state.processed_data:
        st.error("No data could be processed. Please check your files.")
        return
    
    # Point selector
    st.subheader("📍 Select Points to Display")
    
    all_point_names = {pid: p.name for pid, p in st.session_state.ipis_points.items()}
    selected_point_ids = st.multiselect(
        "Select IPIS Points",
        options=list(all_point_names.keys()),
        default=list(all_point_names.keys())[:5],  # Default first 5
        format_func=lambda x: all_point_names[x],
        help="Select which points to display in the visualizations"
    )
    
    if not selected_point_ids:
        st.warning("Please select at least one IPIS point to display.")
        return
    
    # Get selected data
    selected_points_data = {
        st.session_state.ipis_points[pid].name: st.session_state.processed_data[pid]
        for pid in selected_point_ids
        if pid in st.session_state.processed_data
    }
    
    st.divider()
    
    # Display tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Individual Profiles",
        "📊 Compare Profiles", 
        "📉 Individual Trends",
        "📋 Compare Trends"
    ])
    
    # Tab 1: Individual Profile Plots
    with tab1:
        st.subheader("Individual Displacement Profiles")
        
        # Select specific point
        point_for_profile = st.selectbox(
            "Select Point",
            options=selected_point_ids,
            format_func=lambda x: all_point_names[x],
            key="profile_point"
        )
        
        point = st.session_state.ipis_points[point_for_profile]
        df = st.session_state.processed_data[point_for_profile]
        
        # Timestamp selection
        available_timestamps = sorted(df['timestamp'].unique())
        
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_timestamps = st.multiselect(
                "Select Timestamps",
                options=available_timestamps,
                default=[available_timestamps[0], available_timestamps[-1]] if len(available_timestamps) > 1 else available_timestamps[:1],
                format_func=lambda x: pd.Timestamp(x).strftime('%Y-%m-%d %H:%M'),
                max_selections=8,
                key="profile_timestamps"
            )
        with col2:
            if st.button("Latest", key="latest_btn"):
                selected_timestamps = [available_timestamps[-1]]
        
        if selected_timestamps:
            fig = create_profile_plot_single(df, selected_timestamps, point.name, 
                                             axis_range=st.session_state.axis_range)
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Comparative Profile Plot
    with tab2:
        st.subheader("Compare Profiles Across Points")
        
        if len(selected_points_data) < 2:
            st.info("Select at least 2 points above to compare profiles.")
        else:
            # Find common timestamp range
            all_timestamps = set()
            for df in selected_points_data.values():
                all_timestamps.update(df['timestamp'].unique())
            common_timestamps = sorted(all_timestamps)
            
            compare_timestamp = st.select_slider(
                "Select Timestamp for Comparison",
                options=common_timestamps,
                value=common_timestamps[-1],
                format_func=lambda x: pd.Timestamp(x).strftime('%Y-%m-%d %H:%M'),
                key="compare_timestamp"
            )
            
            axis_choice = st.radio(
                "Select Axis",
                options=['A', 'B'],
                horizontal=True,
                key="compare_axis"
            )
            
            fig = create_profile_plot_comparison(selected_points_data, compare_timestamp, axis_choice,
                                                  axis_range=st.session_state.axis_range)
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Individual Trend Plot
    with tab3:
        st.subheader("Individual Displacement Trends")
        
        point_for_trend = st.selectbox(
            "Select Point",
            options=selected_point_ids,
            format_func=lambda x: all_point_names[x],
            key="trend_point"
        )
        
        point = st.session_state.ipis_points[point_for_trend]
        df = st.session_state.processed_data[point_for_trend]
        
        available_depths = sorted(df['depth'].unique())
        
        selected_depths = st.multiselect(
            "Select Depths",
            options=available_depths,
            default=[available_depths[0], available_depths[len(available_depths)//2], available_depths[-1]],
            format_func=lambda x: f"{x:.1f} m",
            max_selections=6,
            key="trend_depths"
        )
        
        if selected_depths:
            fig = create_trend_plot_single(df, selected_depths, point.name)
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Comparative Trend Plot
    with tab4:
        st.subheader("Compare Trends Across Points")
        
        if len(selected_points_data) < 2:
            st.info("Select at least 2 points above to compare trends.")
        else:
            # Find depth range
            all_depths = set()
            for df in selected_points_data.values():
                all_depths.update(df['depth'].unique())
            depth_range = sorted(all_depths)
            
            compare_depth = st.select_slider(
                "Select Depth for Comparison",
                options=depth_range,
                value=depth_range[len(depth_range)//2],
                format_func=lambda x: f"{x:.1f} m",
                key="compare_depth"
            )
            
            axis_choice_trend = st.radio(
                "Select Axis",
                options=['A', 'B'],
                horizontal=True,
                key="compare_axis_trend"
            )
            
            fig = create_trend_comparison(selected_points_data, compare_depth, axis_choice_trend)
            st.plotly_chart(fig, use_container_width=True)
    
    # Summary section
    st.divider()
    st.subheader("📊 Summary Statistics")
    
    summary_data = []
    for point_id in selected_point_ids:
        point = st.session_state.ipis_points[point_id]
        df = st.session_state.processed_data[point_id]
        
        latest = df[df['timestamp'] == df['timestamp'].max()]
        
        summary_data.append({
            'Point': point.name,
            'Sensors': point.num_sensors,
            'Records': len(point.raw_df),
            'Max A (mm)': f"{latest['cum_disp_a'].abs().max():.2f}",
            'Max B (mm)': f"{latest['cum_disp_b'].abs().max():.2f}",
            'Max Resultant (mm)': f"{latest['cum_disp_resultant'].max():.2f}",
            'Date Range': f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
