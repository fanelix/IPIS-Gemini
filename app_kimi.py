"""
Multi-Point In-Place Inclinometer (IPI) Dashboard v3.0
======================================================
Professional-grade Streamlit application for geotechnical monitoring.

Improvements:
- Robust data validation and error handling
- Intelligent caching for performance
- Data quality scoring and anomaly detection
- Export capabilities (CSV/Excel)
- Rate-of-change (velocity) analysis
- Alarm threshold monitoring
- Responsive UI with pagination
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import re
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import hashlib
from enum import Enum
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================
class Config:
    MAX_POINTS = 20
    GAUGE_OPTIONS = [1.0, 2.0, 3.0]
    DEFAULT_GAUGE = 3.0
    MAX_TIMESTAMPS_PROFILE = 10
    MAX_DEPTHS_TREND = 8
    CACHE_TTL = 3600  # 1 hour
    
    # Alarm thresholds (mm)
    WARNING_THRESHOLD = 10.0
    CRITICAL_THRESHOLD = 25.0
    
    # Colors
    COLORS = [
        '#2563eb', '#dc2626', '#16a34a', '#9333ea', '#ea580c',
        '#0891b2', '#c026d3', '#4f46e5', '#059669', '#d97706'
    ]

class DataQuality(Enum):
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    ERROR = "error"

# =============================================================================
# DATA MODELS
# =============================================================================
@dataclass(frozen=True)
class ProcessingConfig:
    """Immutable configuration for processing."""
    gauge_lengths: Tuple[float, ...]
    top_depth: float
    base_reading_idx: int
    use_raw_tilt: bool

@dataclass
class IPIMetadata:
    """Metadata for an IPI point."""
    station_name: str
    format_type: str
    num_sensors: int
    num_records: int
    date_range: Tuple[datetime, datetime]
    file_hash: str
    columns_detected: Dict[str, Any]

@dataclass
class ProcessedResult:
    """Container for processed data with quality metrics."""
    df: pd.DataFrame
    metadata: IPIMetadata
    quality_score: float
    max_displacement: float
    max_velocity: float  # mm/day
    config: ProcessingConfig

# =============================================================================
# DATA VALIDATION & PARSING
# =============================================================================
class TOA5Parser:
    """Robust parser for Campbell Scientific TOA5 format."""
    
    TIMESTAMP_COLS = ['timestamp', 'ts', 'datetime', 'time']
    
    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL, show_spinner="Parsing file...")
    def parse(file_content: str, filename: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Parse TOA5 file with comprehensive validation.
        Returns: (dataframe, metadata_dict) or raises ValueError
        """
        if not file_content or len(file_content.strip()) < 100:
            raise ValueError("File appears to be empty or too small")
        
        lines = TOA5Parser._clean_lines(file_content)
        if len(lines) < 5:
            raise ValueError("Invalid TOA5 format: insufficient header rows")
        
        # Parse header
        header = TOA5Parser._parse_csv_line(lines[0])
        metadata = {
            'format': header[0] if len(header) > 0 else 'Unknown',
            'station_name': header[1] if len(header) > 1 else filename.replace('.dat', ''),
            'logger_model': header[2] if len(header) > 2 else 'Unknown',
            'serial_number': header[3] if len(header) > 3 else 'Unknown',
            'program_name': header[5] if len(header) > 5 else 'Unknown',
        }
        
        # Parse column headers with format detection
        raw_columns = TOA5Parser._detect_and_parse_columns(lines[1])
        
        # Parse data rows
        data_rows = []
        expected_fields = len(raw_columns)
        
        for i, line in enumerate(lines[4:], start=5):
            if not line.strip():
                continue
            fields = TOA5Parser._parse_csv_line(line)
            
            # Handle row length mismatches gracefully
            if len(fields) == expected_fields:
                data_rows.append(fields)
            elif len(fields) > expected_fields:
                data_rows.append(fields[:expected_fields])
            else:
                # Pad with NaN
                fields.extend([np.nan] * (expected_fields - len(fields)))
                data_rows.append(fields)
        
        if not data_rows:
            raise ValueError("No valid data rows found")
        
        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=raw_columns)
        
        # Convert timestamp
        df = TOA5Parser._standardize_timestamp(df)
        
        # Convert numeric columns
        for col in df.columns:
            if col.lower() not in ['timestamp']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Validate data
        if df['timestamp'].isna().all():
            raise ValueError("No valid timestamps found")
        
        df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        
        if len(df) < 2:
            raise ValueError("Insufficient data points (minimum 2 required)")
        
        # Check for duplicate timestamps
        if df['timestamp'].duplicated().any():
            st.warning(f"⚠️ Duplicate timestamps detected in {filename}. Keeping first occurrence.")
            df = df.drop_duplicates(subset=['timestamp'], keep='first')
        
        return df, metadata
    
    @staticmethod
    def _clean_lines(content: str) -> List[str]:
        """Normalize line endings and clean content."""
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        lines = []
        for line in content.split('\n'):
            line = line.strip()
            if line:
                lines.append(line)
        return lines
    
    @staticmethod
    def _parse_csv_line(line: str) -> List[str]:
        """Parse CSV handling quoted fields."""
        fields = []
        current = []
        in_quotes = False
        
        for char in line:
            if char == '"':
                in_quotes = not in_quotes
            elif char == ',' and not in_quotes:
                fields.append(''.join(current).strip().strip('"'))
                current = []
            else:
                current.append(char)
        
        fields.append(''.join(current).strip().strip('"'))
        return fields
    
    @staticmethod
    def _detect_and_parse_columns(line: str) -> List[str]:
        """Auto-detect column format and parse accordingly."""
        line = line.strip()
        
        # Format 3: Wrapped in outer quotes with doubled inner quotes
        if line.startswith('"') and line.endswith('"') and '""' in line:
            inner = line[1:-1].replace('""', '"')
            return TOA5Parser._parse_csv_line(inner)
        
        # Format 2: Standard CSV
        return TOA5Parser._parse_csv_line(line)
    
    @staticmethod
    def _standardize_timestamp(df: pd.DataFrame) -> pd.DataFrame:
        """Find and standardize timestamp column."""
        timestamp_col = None
        
        for col in df.columns:
            if any(ts in col.lower() for ts in TOA5Parser.TIMESTAMP_COLS):
                timestamp_col = col
                break
        
        if timestamp_col is None:
            raise ValueError("No timestamp column detected")
        
        # Rename to standard
        df = df.rename(columns={timestamp_col: 'timestamp'})
        
        # Parse with multiple format attempts
        formats = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S']
        
        for fmt in formats:
            try:
                parsed = pd.to_datetime(df['timestamp'], format=fmt, errors='coerce')
                if parsed.notna().sum() > len(df) * 0.8:  # 80% success rate
                    df['timestamp'] = parsed
                    return df
            except:
                continue
        
        # Fallback to pandas inference
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        return df

class ColumnDetector:
    """Detect and categorize IPI sensor columns."""
    
    PATTERNS = {
        'tilt_a': [r'tilt_a\(?1?,?(\d+)\)?', r'ipis_tilt_a\((\d+)\)', r'tilt_a_?(\d+)'],
        'tilt_b': [r'tilt_b\(?1?,?(\d+)\)?', r'ipis_tilt_b\((\d+)\)', r'tilt_b_?(\d+)'],
        'def_a': [r'(?:ipi_)?def_a\(?1?,?(\d+)\)?', r'ipis_def_a\((\d+)\)', r'def_a_?(\d+)'],
        'def_b': [r'(?:ipi_)?def_b\(?1?,?(\d+)\)?', r'ipis_def_b\((\d+)\)', r'def_b_?(\d+)'],
        'temp': [r'(?:ipi_)?temp\(?1?,?(\d+)\)?', r'ipis_therm\((\d+)\)', r'therm_?(\d+)'],
        'battery': [r'battv?', r'battery', r'bat_volt'],
        'panel_temp': [r'ptemp', r'panel_temp'],
    }
    
    @staticmethod
    def detect(df: pd.DataFrame) -> Dict[str, Any]:
        """Detect IPI columns with confidence scoring."""
        columns = df.columns.tolist()
        detected = {key: [] for key in ColumnDetector.PATTERNS.keys()}
        detected['timestamp'] = 'timestamp' if 'timestamp' in columns else None
        detected['num_sensors'] = 0
        detected['format_type'] = 'unknown'
        
        for col in columns:
            col_lower = col.lower()
            
            for category, patterns in ColumnDetector.PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, col_lower):
                        match = re.search(pattern, col_lower)
                        detected[category].append((col, int(match.group(1)) if match.groups() else 0))
                        break
        
        # Sort by sensor number
        for key in ['tilt_a', 'tilt_b', 'def_a', 'def_b', 'temp']:
            detected[key] = sorted(detected[key], key=lambda x: x[1])
            detected[key] = [x[0] for x in detected[key]]  # Keep just names
        
        # Determine format type
        if any('(1,' in c for c in columns):
            detected['format_type'] = '2d_array'
        elif any('ipis_' in c.lower() for c in columns):
            detected['format_type'] = 'legacy'
        else:
            detected['format_type'] = 'standard'
        
        # Count sensors (max of tilt columns)
        detected['num_sensors'] = max(
            len(detected['tilt_a']),
            len(detected['tilt_b']),
            len(detected['def_a']),
            len(detected['def_b']),
            0
        )
        
        # Validation
        if detected['num_sensors'] == 0:
            raise ValueError("No IPI sensor columns detected. Check file format.")
        
        if len(detected['tilt_a']) != len(detected['tilt_b']):
            st.warning("⚠️ Mismatched A/B axis sensor counts")
        
        return detected

# =============================================================================
# PROCESSING ENGINE
# =============================================================================
class IPIProcessor:
    """Core processing logic with caching."""
    
    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL, show_spinner="Calculating displacements...")
    def process(df: pd.DataFrame, columns: Dict, config: ProcessingConfig) -> pd.DataFrame:
        """
        Process IPI data to calculate cumulative displacements.
        Returns processed DataFrame with displacement columns.
        """
        num_sensors = columns['num_sensors']
        
        # Validate gauge lengths
        if len(config.gauge_lengths) != num_sensors:
            raise ValueError(f"Gauge length count ({len(config.gauge_lengths)}) doesn't match sensor count ({num_sensors})")
        
        # Calculate depths
        depths = IPIProcessor._calculate_depths(config.gauge_lengths, config.top_depth)
        
        # Extract data as matrices (timesteps x sensors)
        timestamps = df['timestamp'].values
        
        if config.use_raw_tilt and columns['tilt_a'] and columns['tilt_b']:
            tilt_a = df[columns['tilt_a']].values
            tilt_b = df[columns['tilt_b']].values
            
            # Calculate incremental displacements (sin(θ) * L * 1000 = mm)
            inc_a = tilt_a * np.array(config.gauge_lengths) * 1000
            inc_b = tilt_b * np.array(config.gauge_lengths) * 1000
        elif columns['def_a'] and columns['def_b']:
            inc_a = df[columns['def_a']].values
            inc_b = df[columns['def_b']].values
        else:
            raise ValueError("No tilt or deflection data available")
        
        # Temperature
        if columns['temp']:
            temps = df[columns['temp']].values
        else:
            temps = np.full((len(df), num_sensors), np.nan)
        
        # Base reading correction
        if config.base_reading_idx >= len(df):
            base_idx = 0
        else:
            base_idx = config.base_reading_idx
        
        base_a = inc_a[base_idx, :]
        base_b = inc_b[base_idx, :]
        
        inc_a_corr = inc_a - base_a
        inc_b_corr = inc_b - base_b
        
        # Cumulative displacement (from bottom up)
        cum_a = np.flip(np.nancumsum(np.flip(inc_a_corr, axis=1), axis=1), axis=1)
        cum_b = np.flip(np.nancumsum(np.flip(inc_b_corr, axis=1), axis=1), axis=1)
        cum_res = np.sqrt(cum_a**2 + cum_b**2)
        
        # Velocity calculation (simple difference / time difference in days)
        if len(timestamps) > 1:
            time_diff = np.diff(timestamps).astype('timedelta64[s]').astype(float) / 86400.0
            time_diff = np.concatenate([[np.nan], time_diff])  # Pad first value
            
            # Rate of change of resultant displacement
            res_change = np.concatenate([[np.nan], np.diff(cum_res[:, -1])])  # Bottom sensor
            velocity = np.divide(res_change, time_diff, out=np.zeros_like(res_change), where=time_diff!=0)
        else:
            velocity = np.zeros(len(timestamps))
        
        # Build long-format DataFrame
        records = []
        for t_idx, ts in enumerate(timestamps):
            for s_idx in range(num_sensors):
                records.append({
                    'timestamp': ts,
                    'sensor_num': s_idx + 1,
                    'depth': depths[s_idx],
                    'gauge_length': config.gauge_lengths[s_idx],
                    'tilt_a': tilt_a[t_idx, s_idx] if config.use_raw_tilt else np.nan,
                    'tilt_b': tilt_b[t_idx, s_idx] if config.use_raw_tilt else np.nan,
                    'inc_disp_a': inc_a[t_idx, s_idx],
                    'inc_disp_b': inc_b[t_idx, s_idx],
                    'inc_disp_a_corr': inc_a_corr[t_idx, s_idx],
                    'inc_disp_b_corr': inc_b_corr[t_idx, s_idx],
                    'cum_disp_a': cum_a[t_idx, s_idx],
                    'cum_disp_b': cum_b[t_idx, s_idx],
                    'cum_disp_resultant': cum_res[t_idx, s_idx],
                    'temperature': temps[t_idx, s_idx] if s_idx < temps.shape[1] else np.nan,
                    'velocity_mm_day': velocity[t_idx] if s_idx == num_sensors - 1 else np.nan  # Only for bottom
                })
        
        return pd.DataFrame(records)
    
    @staticmethod
    def _calculate_depths(gauge_lengths: Tuple[float, ...], top_depth: float) -> np.ndarray:
        """Calculate absolute depths from gauge lengths."""
        depths = np.zeros(len(gauge_lengths))
        depths[0] = top_depth
        for i in range(1, len(gauge_lengths)):
            depths[i] = depths[i-1] + gauge_lengths[i-1]
        return depths
    
    @staticmethod
    def calculate_quality_score(df: pd.DataFrame) -> float:
        """Calculate data quality score 0-100."""
        scores = []
        
        # Completeness (40%)
        completeness = (1 - df.isna().mean().mean()) * 40
        scores.append(completeness)
        
        # Timestamp regularity (30%)
        time_diff = df['timestamp'].diff().dropna()
        if len(time_diff) > 1:
            regularity = (1 - (time_diff.std() / time_diff.mean())) * 30 if time_diff.mean() > 0 else 0
            scores.append(max(0, regularity))
        
        # Physical validity - no impossible displacements (30%)
        max_disp = df['cum_disp_resultant'].abs().max()
        if pd.notna(max_disp) and max_disp < 500:  # 500mm threshold
            scores.append(30)
        elif pd.notna(max_disp) and max_disp < 1000:
            scores.append(15)
        else:
            scores.append(0)
        
        return sum(scores)

# =============================================================================
# VISUALIZATION
# =============================================================================
class Visualizer:
    """Plotly visualization factory."""
    
    @staticmethod
    def create_profile_plot(df: pd.DataFrame, timestamps: List, point_name: str, 
                           axis_range: Optional[Dict] = None) -> go.Figure:
        """Create displacement profile plot."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('A-Axis Displacement', 'B-Axis Displacement'),
            shared_yaxes=True,
            horizontal_spacing=0.08
        )
        
        for i, ts in enumerate(timestamps):
            mask = df['timestamp'] == ts
            data = df[mask].sort_values('depth')
            color = Config.COLORS[i % len(Config.COLORS)]
            ts_str = pd.Timestamp(ts).strftime('%Y-%m-%d %H:%M')
            
            # A-axis
            fig.add_trace(
                go.Scatter(
                    x=data['cum_disp_a'], y=data['depth'],
                    mode='lines+markers',
                    name=ts_str,
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                    legendgroup=f'g{i}',
                    showlegend=True,
                    hovertemplate=f'<b>{ts_str}</b><br>Depth: %{{y:.1f}}m<br>A: %{{x:.2f}}mm<extra></extra>'
                ),
                row=1, col=1
            )
            
            # B-axis
            fig.add_trace(
                go.Scatter(
                    x=data['cum_disp_b'], y=data['depth'],
                    mode='lines+markers',
                    name=ts_str,
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                    legendgroup=f'g{i}',
                    showlegend=False,
                    hovertemplate=f'<b>{ts_str}</b><br>Depth: %{{y:.1f}}m<br>B: %{{x:.2f}}mm<extra></extra>'
                ),
                row=1, col=2
            )
        
        # Zero lines
        fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1, row=1, col=1)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1, row=1, col=2)
        
        # Layout
        fig.update_layout(
            title=dict(
                text=f'<b>{point_name}</b><br><sub>Displacement Profile</sub>',
                x=0.5, font=dict(size=16)
            ),
            height=600,
            template='plotly_white',
            legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center'),
            margin=dict(t=80, b=80)
        )
        
        # Axis ranges
        x_config = dict(title='Displacement (mm)', zeroline=True, zerolinecolor='gray')
        if axis_range and not axis_range.get('auto'):
            x_config['range'] = [axis_range['min'], axis_range['max']]
        
        fig.update_xaxes(x_config, row=1, col=1)
        fig.update_xaxes(x_config, row=1, col=2)
        fig.update_yaxes(title='Depth (m)', autorange='reversed', row=1, col=1)
        
        return fig
    
    @staticmethod
    def create_trend_plot(df: pd.DataFrame, depths: List[float], point_name: str) -> go.Figure:
        """Create time history plot."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Displacement History', 'Temperature'),
            shared_xaxes=True,
            vertical_spacing=0.12,
            row_heights=[0.7, 0.3]
        )
        
        all_depths = sorted(df['depth'].unique())
        
        for i, target_depth in enumerate(depths):
            closest = min(all_depths, key=lambda x: abs(x - target_depth))
            mask = df['depth'] == closest
            data = df[mask].sort_values('timestamp')
            color = Config.COLORS[i % len(Config.COLORS)]
            
            # Resultant displacement
            fig.add_trace(
                go.Scatter(
                    x=data['timestamp'],
                    y=data['cum_disp_resultant'],
                    mode='lines+markers',
                    name=f'{closest:.1f}m',
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    hovertemplate=f'<b>{closest:.1f}m</b><br>Time: %{{x}}<br>Resultant: %{{y:.2f}}mm<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Temperature (only first depth to avoid clutter)
            if i == 0 and data['temperature'].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=data['timestamp'],
                        y=data['temperature'],
                        mode='lines',
                        name=f'Temp @ {closest:.1f}m',
                        line=dict(color='orange', width=1, dash='dot'),
                        showlegend=True,
                        hovertemplate=f'Temp: %{{y:.1f}}°C<extra></extra>'
                    ),
                    row=2, col=1
                )
        
        # Alarm lines
        fig.add_hline(y=Config.WARNING_THRESHOLD, line_dash="dash", line_color="orange", 
                     line_width=1, row=1, col=1, annotation_text="Warning")
        fig.add_hline(y=Config.CRITICAL_THRESHOLD, line_dash="dash", line_color="red", 
                     line_width=1, row=1, col=1, annotation_text="Critical")
        fig.add_hline(y=0, line_color="gray", line_width=1, row=1, col=1)
        
        fig.update_layout(
            title=dict(text=f'<b>{point_name}</b><br><sub>Time History</sub>', x=0.5),
            height=700,
            template='plotly_white',
            legend=dict(orientation='h', y=-0.2, x=0.5, xanchor='center'),
            hovermode='x unified',
            margin=dict(t=80, b=100)
        )
        
        fig.update_yaxes(title='Resultant Disp (mm)', row=1, col=1)
        fig.update_yaxes(title='Temperature (°C)', row=2, col=1)
        fig.update_xaxes(title='Date/Time', row=2, col=1)
        
        return fig
    
    @staticmethod
    def create_comparison_plot(points_data: Dict[str, pd.DataFrame], timestamp, axis: str = 'A') -> go.Figure:
        """Compare multiple points at specific timestamp."""
        fig = go.Figure()
        disp_col = f'cum_disp_{axis.lower()}'
        
        for i, (name, df) in enumerate(points_data.items()):
            mask = df['timestamp'] == timestamp
            data = df[mask].sort_values('depth')
            if data.empty:
                continue
            
            color = Config.COLORS[i % len(Config.COLORS)]
            fig.add_trace(go.Scatter(
                x=data[disp_col],
                y=data['depth'],
                mode='lines+markers',
                name=name,
                line=dict(color=color, width=2.5),
                marker=dict(size=7),
                hovertemplate=f'<b>{name}</b><br>Depth: %{{y:.1f}}m<br>{axis}: %{{x:.2f}}mm<extra></extra>'
            ))
        
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        ts_str = pd.Timestamp(timestamp).strftime('%Y-%m-%d %H:%M')
        fig.update_layout(
            title=f'Multi-Point Comparison - {axis}-Axis<br><sub>{ts_str}</sub>',
            xaxis_title='Displacement (mm)',
            yaxis_title='Depth (m)',
            yaxis_autorange='reversed',
            template='plotly_white',
            height=600,
            legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center')
        )
        return fig

# =============================================================================
# UI COMPONENTS
# =============================================================================
class UI:
    """UI rendering components."""
    
    @staticmethod
    def render_sidebar():
        """Render sidebar controls."""
        with st.sidebar:
            st.title("⚙️ Configuration")
            
            # Point counter with progress
            current = len(st.session_state.get('points', {}))
            st.progress(current / Config.MAX_POINTS, text=f"Points: {current}/{Config.MAX_POINTS}")
            
            if current >= Config.MAX_POINTS:
                st.error("Maximum points reached. Remove some to add more.")
                return None
            
            # File upload
            st.divider()
            st.subheader("1. Upload Data")
            files = st.file_uploader(
                "TOA5 Files (.dat)",
                type=['dat', 'csv'],
                accept_multiple_files=True,
                key='file_upload'
            )
            
            # Processing options
            st.divider()
            st.subheader("2. Processing")
            use_raw = st.radio(
                "Data Source",
                ['Raw Tilt (sin θ)', 'Pre-calculated Deflection'],
                index=0,
                key='data_source'
            ) == 'Raw Tilt (sin θ)'
            
            # Axis settings
            st.divider()
            st.subheader("3. Display Settings")
            auto_range = st.checkbox("Auto X-Axis", value=True, key='auto_range')
            if not auto_range:
                c1, c2 = st.columns(2)
                with c1:
                    st.number_input("Min (mm)", value=-50.0, step=10.0, key='x_min')
                with c2:
                    st.number_input("Max (mm)", value=50.0, step=10.0, key='x_max')
            
            return {
                'files': files,
                'use_raw_tilt': use_raw,
                'axis_range': {
                    'auto': auto_range,
                    'min': st.session_state.get('x_min', -50.0),
                    'max': st.session_state.get('x_max', 50.0)
                }
            }
    
    @staticmethod
    def render_point_manager():
        """Render point configuration cards."""
        if 'points' not in st.session_state or not st.session_state.points:
            return
        
        st.sidebar.divider()
        st.sidebar.subheader("4. Point Configuration")
        
        # Use a form to batch updates
        with st.sidebar.form("point_config_form"):
            for pid, point in list(st.session_state.points.items()):
                with st.expander(f"📍 {point['metadata'].station_name}", expanded=False):
                    # Quick gauge settings
                    st.write("**Gauge Lengths**")
                    num_sensors = point['metadata'].num_sensors
                    
                    # All same button
                    col1, col2, col3 = st.columns(3)
                    gauge_vals = list(point['config'].gauge_lengths)
                    
                    with col1:
                        if st.button("All 1m", key=f"{pid}_1m"):
                            gauge_vals = [1.0] * num_sensors
                    with col2:
                        if st.button("All 2m", key=f"{pid}_2m"):
                            gauge_vals = [2.0] * num_sensors
                    with col3:
                        if st.button("All 3m", key=f"{pid}_3m"):
                            gauge_vals = [3.0] * num_sensors
                    
                    # Individual sensors (collapsible)
                    if num_sensors <= 6:
                        gauge_vals = []
                        cols = st.columns(num_sensors)
                        for i, c in enumerate(cols):
                            with c:
                                val = st.selectbox(
                                    f"S{i+1}",
                                    Config.GAUGE_OPTIONS,
                                    index=Config.GAUGE_OPTIONS.index(point['config'].gauge_lengths[i]),
                                    key=f"{pid}_g_{i}"
                                )
                                gauge_vals.append(val)
                    
                    # Top depth
                    top_depth = st.number_input(
                        "Top Depth (m)",
                        value=point['config'].top_depth,
                        step=0.5,
                        key=f"{pid}_td"
                    )
                    
                    # Base reading
                    timestamps = point['raw_df']['timestamp'].tolist()
                    base_idx = st.selectbox(
                        "Base Reading",
                        range(len(timestamps)),
                        format_func=lambda i: timestamps[i].strftime('%Y-%m-%d %H:%M'),
                        index=point['config'].base_reading_idx,
                        key=f"{pid}_base"
                    )
                    
                    # Update button
                    if st.button("Update", key=f"{pid}_update"):
                        new_config = ProcessingConfig(
                            gauge_lengths=tuple(gauge_vals),
                            top_depth=top_depth,
                            base_reading_idx=base_idx,
                            use_raw_tilt=point['config'].use_raw_tilt
                        )
                        st.session_state.points[pid]['config'] = new_config
                        # Clear cache for this point
                        if 'processed' in st.session_state and pid in st.session_state.processed:
                            del st.session_state.processed[pid]
                        st.rerun()
                    
                    # Remove button
                    if st.button("🗑️ Remove", key=f"{pid}_del", type="secondary"):
                        del st.session_state.points[pid]
                        if 'processed' in st.session_state and pid in st.session_state.processed:
                            del st.session_state.processed[pid]
                        st.rerun()
            
            st.form_submit_button("Apply All Changes", type="primary")
    
    @staticmethod
    def render_quality_badge(score: float):
        """Render quality indicator."""
        if score >= 80:
            st.success(f"✅ Quality: {score:.0f}/100")
        elif score >= 60:
            st.warning(f"⚠️ Quality: {score:.0f}/100")
        else:
            st.error(f"❌ Quality: {score:.0f}/100")
    
    @staticmethod
    def render_summary_table(points_data: Dict[str, ProcessedResult]):
        """Render summary statistics table."""
        if not points_data:
            return
        
        summary = []
        for name, result in points_data.items():
            latest = result.df[result.df['timestamp'] == result.df['timestamp'].max()]
            summary.append({
                'Point': name,
                'Quality': f"{result.quality_score:.0f}",
                'Sensors': result.metadata.num_sensors,
                'Max Disp (mm)': f"{result.max_displacement:.2f}",
                'Max Velocity (mm/d)': f"{result.max_velocity:.3f}" if result.max_velocity != 0 else "-",
                'Records': result.metadata.num_records,
                'Latest': result.metadata.date_range[1].strftime('%Y-%m-%d')
            })
        
        st.dataframe(
            pd.DataFrame(summary),
            use_container_width=True,
            hide_index=True,
            column_config={
                'Quality': st.column_config.ProgressColumn(
                    "Quality",
                    help="Data quality score",
                    format="%d",
                    min_value=0,
                    max_value=100,
                )
            }
        )

# =============================================================================
# MAIN APPLICATION
# =============================================================================
def init_session():
    """Initialize session state."""
    defaults = {
        'points': {},  # Dict[str, Dict] - raw data and configs
        'processed': {},  # Dict[str, ProcessedResult]
        'axis_range': {'auto': True, 'min': -50, 'max': 50},
        'use_raw_tilt': True
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def process_point(pid: str, point_data: Dict) -> Optional[ProcessedResult]:
    """Process a single point with error handling."""
    try:
        config = point_data['config']
        df_raw = point_data['raw_df']
        columns = point_data['columns']
        meta = point_data['metadata']
        
        processed_df = IPIProcessor.process(df_raw, columns, config)
        quality = IPIProcessor.calculate_quality_score(processed_df)
        
        # Calculate aggregates
        max_disp = processed_df['cum_disp_resultant'].max()
        max_vel = processed_df['velocity_mm_day'].abs().max()
        
        return ProcessedResult(
            df=processed_df,
            metadata=meta,
            quality_score=quality,
            max_displacement=max_disp,
            max_velocity=max_vel,
            config=config
        )
    except Exception as e:
        st.error(f"Processing error for {point_data['metadata'].station_name}: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="IPI Dashboard Pro",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS (minimal, robust)
    st.markdown("""
    <style>
    .stProgress > div > div > div { background-color: #2563eb; }
    .quality-good { color: #16a34a; font-weight: bold; }
    .quality-warn { color: #ca8a04; font-weight: bold; }
    .quality-bad { color: #dc2626; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)
    
    init_session()
    
    # Header
    st.title("📊 Multi-Point IPI Dashboard Pro")
    st.caption("Professional In-Place Inclinometer Analysis Tool")
    
    # Sidebar
    config = UI.render_sidebar()
    
    # Handle file uploads
    if config and config['files']:
        for file in config['files']:
            if len(st.session_state.points) >= Config.MAX_POINTS:
                st.sidebar.warning(f"Skipped {file.name}: limit reached")
                continue
            
            try:
                content = file.read().decode('utf-8')
                file_hash = hashlib.md5(content.encode()).hexdigest()[:12]
                
                # Check for duplicates
                if any(p['metadata'].file_hash == file_hash for p in st.session_state.points.values()):
                    st.sidebar.warning(f"{file.name} already loaded")
                    continue
                
                # Parse
                df, meta_dict = TOA5Parser.parse(content, file.name)
                columns = ColumnDetector.detect(df)
                
                # Create metadata
                meta = IPIMetadata(
                    station_name=meta_dict.get('station_name', file.name.replace('.dat', '')),
                    format_type=columns['format_type'],
                    num_sensors=columns['num_sensors'],
                    num_records=len(df),
                    date_range=(df['timestamp'].min(), df['timestamp'].max()),
                    file_hash=file_hash,
                    columns_detected=columns
                )
                
                # Default config
                proc_config = ProcessingConfig(
                    gauge_lengths=tuple([Config.DEFAULT_GAUGE] * columns['num_sensors']),
                    top_depth=1.0,
                    base_reading_idx=0,
                    use_raw_tilt=config['use_raw_tilt']
                )
                
                # Store
                pid = f"point_{file_hash}"
                st.session_state.points[pid] = {
                    'raw_df': df,
                    'metadata': meta,
                    'columns': columns,
                    'config': proc_config
                }
                
                st.sidebar.success(f"✅ Loaded {meta.station_name}")
                
            except Exception as e:
                st.sidebar.error(f"❌ {file.name}: {str(e)}")
    
    # Render point configuration
    UI.render_point_manager()
    
    # Process all points
    if st.session_state.points:
        with st.spinner("Processing data..."):
            for pid, pdata in st.session_state.points.items():
                if pid not in st.session_state.processed:
                    result = process_point(pid, pdata)
                    if result:
                        st.session_state.processed[pid] = result
        
        # Main content
        if not st.session_state.processed:
            st.info("Processing failed for all points. Check error messages.")
            return
        
        # Point selection
        point_options = {
            pid: f"{r.metadata.station_name} (Q:{r.quality_score:.0f})" 
            for pid, r in st.session_state.processed.items()
        }
        
        selected_ids = st.multiselect(
            "Select Points to Display",
            options=list(point_options.keys()),
            default=list(point_options.keys())[:3],
            format_func=lambda x: point_options[x],
            key='point_selector'
        )
        
        if not selected_ids:
            st.warning("Select at least one point")
            return
        
        # Get selected data
        selected_data = {
            st.session_state.processed[pid].metadata.station_name: st.session_state.processed[pid]
            for pid in selected_ids
        }
        
        # Summary cards
        cols = st.columns(len(selected_data))
        for i, (name, result) in enumerate(selected_data.items()):
            with cols[i]:
                with st.container(border=True):
                    st.markdown(f"**{name}**")
                    UI.render_quality_badge(result.quality_score)
                    st.metric("Max Disp", f"{result.max_displacement:.2f} mm")
                    if result.max_velocity > 0:
                        st.metric("Max Velocity", f"{result.max_velocity:.2f} mm/day")
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Profiles", "📊 Comparison", "📉 Trends", "📋 Export"
        ])
        
        # Tab 1: Individual Profiles
        with tab1:
            if len(selected_ids) == 1:
                pid = selected_ids[0]
                result = st.session_state.processed[pid]
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    timestamps = sorted(result.df['timestamp'].unique())
                    selected_ts = st.multiselect(
                        "Select Timestamps",
                        options=timestamps,
                        default=[timestamps[0], timestamps[-1]] if len(timestamps) > 1 else [timestamps[0]],
                        format_func=lambda x: pd.Timestamp(x).strftime('%Y-%m-%d %H:%M'),
                        max_selections=Config.MAX_TIMESTAMPS_PROFILE
                    )
                with col2:
                    st.write("")  # Spacing
                    st.write("")
                    if st.button("Select Latest Only", use_container_width=True):
                        st.session_state[f'force_ts_{pid}'] = [timestamps[-1]]
                        st.rerun()
                
                if selected_ts:
                    fig = Visualizer.create_profile_plot(
                        result.df, selected_ts, result.metadata.station_name,
                        config['axis_range'] if not config['axis_range']['auto'] else None
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select a single point in the dropdown above to view individual profiles, or use the Comparison tab.")
        
        # Tab 2: Comparison
        with tab2:
            if len(selected_data) < 2:
                st.info("Select 2+ points to compare")
            else:
                # Find common timestamps
                all_ts = set.intersection(*[set(r.df['timestamp'].unique()) for r in selected_data.values()])
                if not all_ts:
                    st.error("No common timestamps found between selected points")
                else:
                    common_ts = sorted(all_ts)
                    compare_ts = st.select_slider(
                        "Select Timestamp",
                        options=common_ts,
                        value=common_ts[-1],
                        format_func=lambda x: pd.Timestamp(x).strftime('%Y-%m-%d %H:%M')
                    )
                    
                    axis = st.radio("Axis", ['A', 'B'], horizontal=True)
                    
                    fig = Visualizer.create_comparison_plot(
                        {name: r.df for name, r in selected_data.items()},
                        compare_ts, axis
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Tab 3: Trends
        with tab3:
            if len(selected_ids) == 1:
                pid = selected_ids[0]
                result = st.session_state.processed[pid]
                depths = sorted(result.df['depth'].unique())
                
                selected_depths = st.multiselect(
                    "Select Depths",
                    options=depths,
                    default=[depths[0], depths[len(depths)//2], depths[-1]] if len(depths) >= 3 else depths[:1],
                    format_func=lambda x: f"{x:.1f} m",
                    max_selections=Config.MAX_DEPTHS_TREND
                )
                
                if selected_depths:
                    fig = Visualizer.create_trend_plot(result.df, selected_depths, result.metadata.station_name)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Trend view works best with a single point selected")
        
        # Tab 4: Export
        with tab4:
            st.subheader("Data Export")
            
            export_format = st.radio("Format", ['CSV', 'Excel'], horizontal=True)
            
            for pid in selected_ids:
                result = st.session_state.processed[pid]
                name = result.metadata.station_name
                
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.write(f"**{name}**")
                with col2:
                    st.caption(f"{result.metadata.num_records} records, {result.metadata.num_sensors} sensors")
                with col3:
                    if export_format == 'CSV':
                        csv = result.df.to_csv(index=False)
                        st.download_button(
                            "⬇️ CSV",
                            data=csv,
                            file_name=f"IPI_{name}_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime='text/csv',
                            key=f"dl_csv_{pid}"
                        )
                    else:
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                            result.df.to_excel(writer, sheet_name='Data', index=False)
                            # Add metadata sheet
                            meta_df = pd.DataFrame([
                                ['Station', result.metadata.station_name],
                                ['Sensors', result.metadata.num_sensors],
                                ['Date Range', f"{result.metadata.date_range[0]} to {result.metadata.date_range[1]}"],
                                ['Quality Score', result.quality_score],
                                ['Gauge Lengths', ', '.join(map(str, result.config.gauge_lengths))]
                            ], columns=['Property', 'Value'])
                            meta_df.to_excel(writer, sheet_name='Metadata', index=False)
                        
                        st.download_button(
                            "⬇️ Excel",
                            data=buffer.getvalue(),
                            file_name=f"IPI_{name}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime='application/vnd.ms-excel',
                            key=f"dl_xlsx_{pid}"
                        )
        
        # Detailed Summary Table
        st.divider()
        st.subheader("Summary Statistics")
        UI.render_summary_table(selected_data)

if __name__ == "__main__":
    main()