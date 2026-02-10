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
Version: 3.0 - Refactored for Performance & Robustness
"""

import csv
import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# =============================================================================
# CONSTANTS
# =============================================================================
MAX_IPIS_POINTS = 20
GAUGE_LENGTH_OPTIONS = [1.0, 2.0, 3.0]
DEFAULT_GAUGE_LENGTH = 3.0
DEFAULT_TOP_DEPTH = 1.0
DEFAULT_AXIS_RANGE = {"auto": True, "min": -50.0, "max": 50.0}
SENSORS_PER_ROW = 4

# High contrast colors for data series
CHART_COLORS = [
    "#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c",
    "#0891b2", "#c026d3", "#4f46e5", "#059669", "#d97706",
    "#7c3aed", "#db2777", "#0d9488", "#ca8a04", "#6366f1",
    "#e11d48", "#14b8a6", "#f59e0b", "#8b5cf6", "#f43f5e",
]

# Shared Plotly axis style
_AXIS_STYLE = dict(
    title_font=dict(size=11, color="#374151"),
    tickfont=dict(size=9, color="#4b5563"),
    gridcolor="#e5e7eb",
    linecolor="#d1d5db",
    linewidth=1,
    showline=True,
    mirror=True,
)

_LEGEND_STYLE = dict(
    orientation="h",
    yanchor="top",
    xanchor="center",
    x=0.5,
    bgcolor="#f8fafc",
    bordercolor="#cbd5e1",
    borderwidth=1,
)

_LAYOUT_COLORS = dict(plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Multi-Point IPI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS (condensed)
# =============================================================================
st.markdown(
    """
<style>
    .stApp { background-color: #f8fafc; color: #1e293b; }
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div { color: #1e293b !important; }

    .main-header {
        font-size: 2.2rem; font-weight: bold; color: #1e40af !important;
        text-align: center; margin-bottom: 0.5rem; padding: 1rem;
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-radius: 8px; border-bottom: 3px solid #2563eb;
    }
    .sub-header { font-size: 1.1rem; color: #475569 !important; text-align: center; margin-bottom: 1.5rem; }
    .point-card {
        background-color: #fff; padding: 1rem; border-radius: 8px;
        border-left: 4px solid #2563eb; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 0.5rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #1e293b !important; }
    section[data-testid="stSidebar"] * { color: #f1f5f9 !important; }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 { color: #60a5fa !important; }

    /* Main content */
    .main .block-container { color: #1e293b !important; }
    .main .block-container p, .main .block-container span,
    .main .block-container label, .main .block-container li { color: #374151 !important; }
    .main .block-container h1, .main .block-container h2,
    .main .block-container h3 { color: #1e40af !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #e2e8f0; padding: 0.5rem; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #fff; border-radius: 6px; color: #1e293b !important; font-weight: 500; }
    .stTabs [aria-selected="true"] { background-color: #2563eb !important; color: #fff !important; }

    /* Expander */
    .streamlit-expanderHeader { background-color: #e2e8f0 !important; color: #1e293b !important; border-radius: 6px; }
    .streamlit-expanderHeader p { color: #1e293b !important; font-weight: 600; }
    .streamlit-expanderContent { background-color: #fff; border: 1px solid #e2e8f0; color: #374151 !important; }

    /* Button */
    .stButton > button { background-color: #2563eb !important; color: #fff !important; border: none; border-radius: 6px; font-weight: 500; }
    .stButton > button:hover { background-color: #1d4ed8 !important; color: #fff !important; }

    /* Inputs */
    .stSelectbox > div > div, .stNumberInput > div > div > input,
    .stDateInput > div > div > input { background-color: #fff !important; color: #1e293b !important; border: 1px solid #cbd5e1; }
    .stMultiSelect > div > div { background-color: #fff !important; color: #1e293b !important; }

    /* Misc */
    .stAlert { background-color: #dbeafe !important; color: #1e40af !important; border: 1px solid #93c5fd; }
    [data-testid="stMetricValue"] { color: #1e40af !important; }
    [data-testid="stMetricLabel"] { color: #475569 !important; }
    .point-counter { background-color: #2563eb; color: white; padding: 0.25rem 0.75rem; border-radius: 20px; font-weight: bold; display: inline-block; }
    .point-counter-full { background-color: #dc2626; }
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class IPISPoint:
    """Represents a single IPIS monitoring point."""

    point_id: str
    name: str
    raw_df: pd.DataFrame
    metadata: Dict
    gauge_lengths: np.ndarray
    top_depth: float = DEFAULT_TOP_DEPTH
    base_reading_idx: int = 0
    num_sensors: int = 0
    detected_cols: Dict = field(default_factory=dict)
    color: str = "#2563eb"

    # Hash of configuration that affects processing; used to skip redundant work.
    _config_hash: str = field(default="", repr=False, compare=False)

    def __post_init__(self):
        if self.detected_cols:
            self.num_sensors = self.detected_cols.get("num_sensors", 0)

    # ------------------------------------------------------------------
    @property
    def config_hash(self) -> str:
        """Return a hash representing the current processing-relevant config."""
        h = hashlib.md5()
        h.update(self.gauge_lengths.tobytes())
        h.update(str(self.top_depth).encode())
        h.update(str(self.base_reading_idx).encode())
        return h.hexdigest()


# =============================================================================
# HELPER: Compute depth array from gauge lengths
# =============================================================================
def compute_depths(top_depth: float, gauge_lengths: np.ndarray) -> np.ndarray:
    """Return depth for each sensor given the top depth and per-sensor gauge lengths."""
    depths = np.empty(len(gauge_lengths))
    depths[0] = top_depth
    np.cumsum(gauge_lengths[:-1], out=depths[1:])
    depths[1:] += top_depth
    return depths


# =============================================================================
# DATA PARSING FUNCTIONS
# =============================================================================
def _split_concatenated_lines(file_content: str) -> List[str]:
    """Normalize line endings and split concatenated timestamp rows."""
    content = file_content.replace("\r\n", "\n").replace("\r", "\n")
    lines = content.split("\n")
    cleaned: List[str] = []
    ts_pat = re.compile(r'"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"')

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        timestamps = list(ts_pat.finditer(stripped))
        if len(timestamps) > 1:
            last_end = 0
            for i, m in enumerate(timestamps):
                if i == 0:
                    continue
                seg = stripped[last_end : m.start()].strip()
                if seg:
                    cleaned.append(seg)
                last_end = m.start()
            tail = stripped[last_end:].strip()
            if tail:
                cleaned.append(tail)
        else:
            cleaned.append(stripped)
    return cleaned


def _parse_csv_row(line: str) -> List[str]:
    """Parse a single CSV line using Python's csv module for correctness."""
    # csv.reader expects an iterable of lines
    reader = csv.reader([line])
    return next(reader)


def _parse_header_line(line: str) -> List[str]:
    """
    Parse a column header that may use the "wrapped outer-quote" format
    where inner quotes are doubled for escaping.
    """
    line = line.strip()
    if line.startswith('"') and line.endswith('"') and '""' in line:
        inner = line[1:-1].replace('""', '"')
        return _parse_csv_row(inner)
    return _parse_csv_row(line)


@st.cache_data(show_spinner="Parsing file…")
def parse_toa5_file(file_content: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Parse a Campbell Scientific TOA5 file.

    Supports column naming formats:
      - Format 1 (Old):  IPIS_Tilt_A(N)
      - Format 2 (New):  Tilt_A(1,N)  (standard CSV)
      - Format 3 (Newest): Tilt_A(1,N) with wrapped-quote header
    """
    lines = _split_concatenated_lines(file_content)
    if len(lines) < 5:
        raise ValueError("File appears too short or corrupted")

    header_info = _parse_csv_row(lines[0])
    metadata = {
        "format": header_info[0] if len(header_info) > 0 else "Unknown",
        "station_name": header_info[1] if len(header_info) > 1 else "Unknown",
        "logger_model": header_info[2] if len(header_info) > 2 else "Unknown",
        "serial_number": header_info[3] if len(header_info) > 3 else "Unknown",
        "program_name": header_info[5] if len(header_info) > 5 else "Unknown",
        "table_name": header_info[7] if len(header_info) > 7 else "Unknown",
    }

    columns = _parse_header_line(lines[1])
    expected = len(columns)

    data_lines = lines[4:]
    valid_rows: List[List[str]] = []
    skipped = 0

    for line in data_lines:
        try:
            fields = _parse_csv_row(line)
            n = len(fields)
            if n == expected:
                valid_rows.append(fields)
            elif n > expected:
                valid_rows.append(fields[:expected])
                skipped += 1
            else:
                fields.extend([""] * (expected - n))
                valid_rows.append(fields)
                skipped += 1
        except Exception:
            skipped += 1

    if not valid_rows:
        raise ValueError("No valid data rows found")

    df = pd.DataFrame(valid_rows, columns=columns)

    # Numeric conversion (vectorised)
    non_ts_cols = [c for c in df.columns if c != "TIMESTAMP"]
    df[non_ts_cols] = df[non_ts_cols].apply(pd.to_numeric, errors="coerce")

    if "TIMESTAMP" in df.columns:
        df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
        df.dropna(subset=["TIMESTAMP"], inplace=True)
        df.sort_values("TIMESTAMP", inplace=True)
        df.reset_index(drop=True, inplace=True)

    metadata["skipped_rows"] = skipped
    metadata["total_rows"] = len(valid_rows)
    return df, metadata


# =============================================================================
# COLUMN DETECTION
# =============================================================================
_SENSOR_NUM_RE_2D = re.compile(r"\(1,(\d+)\)")
_SENSOR_NUM_RE_1D = re.compile(r"\((\d+)\)")


def _extract_sensor_number(col_name: str) -> int:
    m = _SENSOR_NUM_RE_2D.search(col_name)
    if m:
        return int(m.group(1))
    m = _SENSOR_NUM_RE_1D.search(col_name)
    return int(m.group(1)) if m else 0


def detect_ipi_columns(df: pd.DataFrame) -> Dict:
    """Auto-detect IPI sensor columns."""
    columns = df.columns.tolist()
    has_2d = any("(1," in c for c in columns)

    detected: Dict = {
        "timestamp": None,
        "tilt_a": [],
        "tilt_b": [],
        "def_a": [],
        "def_b": [],
        "therm": [],
        "battery": None,
        "panel_temp": None,
        "int_temp": None,
        "num_sensors": 0,
        "format_type": "new_2d" if has_2d else "old_1d",
    }

    for col in columns:
        cl = col.lower()
        if "timestamp" in cl or cl == "ts":
            detected["timestamp"] = col
        elif ("batt" in cl and "volt" in cl) or cl == "battv":
            detected["battery"] = col
        elif "ptemp" in cl:
            detected["panel_temp"] = col
        elif cl == "int_temp":
            detected["int_temp"] = col
        elif "tilt_a" in cl:
            detected["tilt_a"].append(col)
        elif "tilt_b" in cl:
            detected["tilt_b"].append(col)
        elif "def_a" in cl or "ipi_def_a" in cl:
            detected["def_a"].append(col)
        elif "def_b" in cl or "ipi_def_b" in cl:
            detected["def_b"].append(col)
        elif ("therm" in cl and "ptemp" not in cl) or "ipi_temp" in cl:
            detected["therm"].append(col)

    for key in ("tilt_a", "tilt_b", "def_a", "def_b", "therm"):
        detected[key] = sorted(detected[key], key=_extract_sensor_number)

    detected["num_sensors"] = max(
        len(detected["tilt_a"]),
        len(detected["tilt_b"]),
        len(detected["def_a"]),
        len(detected["def_b"]),
    )
    return detected


# =============================================================================
# DISPLACEMENT CALCULATIONS (VECTORISED)
# =============================================================================
def _cumulative_from_bottom(arr: np.ndarray) -> np.ndarray:
    """Cumulative sum from the bottom sensor upward, handling NaN."""
    return np.flip(np.nancumsum(np.flip(arr, axis=-1), axis=-1), axis=-1)


def process_ipis_point(point: IPISPoint, use_raw_tilt: bool = True) -> pd.DataFrame:
    """
    Process a single IPIS point (vectorised).

    Returns a long-form DataFrame with columns:
        point_id, point_name, timestamp, sensor_num, depth,
        gauge_length, inc_disp_a, inc_disp_b, temperature,
        inc_disp_a_corr, inc_disp_b_corr,
        cum_disp_a, cum_disp_b, cum_disp_resultant
    """
    df = point.raw_df
    cols = point.detected_cols
    num_sensors = cols["num_sensors"]

    # --- Ensure gauge_lengths length matches ---
    gl = point.gauge_lengths
    if len(gl) < num_sensors:
        fill = gl[-1] if len(gl) > 0 else DEFAULT_GAUGE_LENGTH
        gl = np.concatenate([gl, np.full(num_sensors - len(gl), fill)])
    elif len(gl) > num_sensors:
        gl = gl[:num_sensors]

    depths = compute_depths(point.top_depth, gl)
    ts_col = cols["timestamp"]
    n_rows = len(df)

    # --- Build 2-D arrays (n_rows × num_sensors) for tilt / deflection ---
    if use_raw_tilt and cols["tilt_a"] and cols["tilt_b"]:
        tilt_a_2d = df[cols["tilt_a"][:num_sensors]].to_numpy(dtype=np.float64)
        tilt_b_2d = df[cols["tilt_b"][:num_sensors]].to_numpy(dtype=np.float64)
        inc_a_2d = tilt_a_2d * gl[np.newaxis, :] * 1000.0  # → mm
        inc_b_2d = tilt_b_2d * gl[np.newaxis, :] * 1000.0
    elif cols["def_a"] and cols["def_b"]:
        inc_a_2d = df[cols["def_a"][:num_sensors]].to_numpy(dtype=np.float64)
        inc_b_2d = df[cols["def_b"][:num_sensors]].to_numpy(dtype=np.float64)
    else:
        return pd.DataFrame()

    # --- Temperature (optional) ---
    if cols["therm"]:
        temp_2d = df[cols["therm"][:num_sensors]].to_numpy(dtype=np.float64)
    else:
        temp_2d = np.full((n_rows, num_sensors), np.nan)

    # --- Base reading correction ---
    base_idx = min(point.base_reading_idx, n_rows - 1)
    base_a = inc_a_2d[base_idx]  # shape (num_sensors,)
    base_b = inc_b_2d[base_idx]
    corr_a = inc_a_2d - base_a[np.newaxis, :]
    corr_b = inc_b_2d - base_b[np.newaxis, :]

    # --- Cumulative displacement (from bottom) ---
    cum_a = _cumulative_from_bottom(corr_a)
    cum_b = _cumulative_from_bottom(corr_b)
    cum_r = np.sqrt(cum_a ** 2 + cum_b ** 2)

    # --- Flatten to long-form DataFrame ---
    timestamps = np.repeat(df[ts_col].values, num_sensors)
    sensor_nums = np.tile(np.arange(1, num_sensors + 1), n_rows)
    depth_arr = np.tile(depths, n_rows)
    gl_arr = np.tile(gl, n_rows)

    result = pd.DataFrame(
        {
            "point_id": point.point_id,
            "point_name": point.name,
            "timestamp": timestamps,
            "sensor_num": sensor_nums,
            "depth": depth_arr,
            "gauge_length": gl_arr,
            "inc_disp_a": inc_a_2d.ravel(),
            "inc_disp_b": inc_b_2d.ravel(),
            "temperature": temp_2d.ravel(),
            "inc_disp_a_corr": corr_a.ravel(),
            "inc_disp_b_corr": corr_b.ravel(),
            "cum_disp_a": cum_a.ravel(),
            "cum_disp_b": cum_b.ravel(),
            "cum_disp_resultant": cum_r.ravel(),
        }
    )
    return result


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================
def _x_axis_config(axis_range: Optional[Dict], label: str = "Displacement (mm)") -> dict:
    cfg = dict(title_text=label, zeroline=True, zerolinecolor="#9ca3af", **_AXIS_STYLE)
    if axis_range and not axis_range.get("auto", True):
        cfg["range"] = [axis_range["min"], axis_range["max"]]
    return cfg


def _title_dict(text: str, size: int = 16) -> dict:
    return dict(text=text, font=dict(size=size, color="#1e293b"), x=0.5, xanchor="center", y=0.95)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def create_profile_plot_single(
    processed_df: pd.DataFrame,
    selected_timestamps: list,
    point_name: str,
    axis_range: Optional[Dict] = None,
) -> go.Figure:
    """Profile plot (A & B axes) for a single IPIS point."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("<b>A-Axis Displacement</b>", "<b>B-Axis Displacement</b>"),
        shared_yaxes=True,
        horizontal_spacing=0.10,
    )

    for i, ts in enumerate(selected_timestamps):
        data = processed_df.loc[processed_df["timestamp"] == ts].sort_values("depth")
        if data.empty:
            continue
        color = CHART_COLORS[i % len(CHART_COLORS)]
        ts_str = pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")
        common = dict(mode="lines+markers", line=dict(color=color, width=2.5), marker=dict(size=7), legendgroup=f"g{i}")

        fig.add_trace(
            go.Scatter(
                x=data["cum_disp_a"], y=data["depth"], name=ts_str, showlegend=True,
                hovertemplate="<b>Depth:</b> %{y:.2f} m<br><b>A-Axis:</b> %{x:.3f} mm<extra></extra>",
                **common,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data["cum_disp_b"], y=data["depth"], name=ts_str, showlegend=False,
                hovertemplate="<b>Depth:</b> %{y:.2f} m<br><b>B-Axis:</b> %{x:.3f} mm<extra></extra>",
                **common,
            ),
            row=1, col=2,
        )

    fig.add_vline(x=0, line_dash="dash", line_color="#64748b", line_width=1.5, row=1, col=1)
    fig.add_vline(x=0, line_dash="dash", line_color="#64748b", line_width=1.5, row=1, col=2)

    fig.update_layout(
        title=_title_dict(f"<b>{point_name} – Cumulative Displacement Profile</b>"),
        legend=dict(y=-0.12, title=dict(text="<b>Timestamp:</b> ", font=dict(size=10)), font=dict(size=9, color="#1e293b"), **_LEGEND_STYLE),
        height=600, margin=dict(t=60, b=80, l=70, r=50), **_LAYOUT_COLORS,
    )
    for ann in fig["layout"]["annotations"]:
        ann["y"] = 1.02
        ann["font"] = dict(size=12, color="#1e293b")

    xcfg = _x_axis_config(axis_range)
    fig.update_xaxes(**xcfg, row=1, col=1)
    fig.update_xaxes(**xcfg, row=1, col=2)
    fig.update_yaxes(title_text="Depth (m)", autorange="reversed", **_AXIS_STYLE, row=1, col=1)
    fig.update_yaxes(autorange="reversed", **_AXIS_STYLE, row=1, col=2)
    return fig


def create_profile_plot_comparison(
    points_data: Dict[str, pd.DataFrame],
    selected_timestamp,
    axis: str = "A",
    axis_range: Optional[Dict] = None,
) -> go.Figure:
    """Comparative profile plot across multiple IPIS points."""
    fig = go.Figure()
    disp_col = "cum_disp_a" if axis == "A" else "cum_disp_b"

    for i, (name, df) in enumerate(points_data.items()):
        data = df.loc[df["timestamp"] == selected_timestamp].sort_values("depth")
        if data.empty:
            continue
        color = CHART_COLORS[i % len(CHART_COLORS)]
        fig.add_trace(go.Scatter(
            x=data[disp_col], y=data["depth"], mode="lines+markers", name=name,
            line=dict(color=color, width=2.5), marker=dict(size=7),
            hovertemplate=f"<b>{name}</b><br>Depth: %{{y:.2f}} m<br>{axis}-Axis: %{{x:.3f}} mm<extra></extra>",
        ))

    fig.add_vline(x=0, line_dash="dash", line_color="#64748b", line_width=1.5)
    ts_str = pd.Timestamp(selected_timestamp).strftime("%Y-%m-%d %H:%M")

    xcfg: dict = dict(title="Cumulative Displacement (mm)", gridcolor="#e5e7eb", linecolor="#d1d5db", zeroline=True, zerolinecolor="#9ca3af")
    if axis_range and not axis_range.get("auto", True):
        xcfg["range"] = [axis_range["min"], axis_range["max"]]

    fig.update_layout(
        title=dict(text=f"<b>Multi-Point {axis}-Axis Comparison</b><br><sub>{ts_str}</sub>", font=dict(size=16, color="#1e293b"), x=0.5, xanchor="center"),
        xaxis=xcfg,
        yaxis=dict(title="Depth (m)", autorange="reversed", gridcolor="#e5e7eb", linecolor="#d1d5db"),
        legend=dict(y=-0.15, **_LEGEND_STYLE),
        height=600, margin=dict(t=80, b=100, l=70, r=50), **_LAYOUT_COLORS,
    )
    return fig


def create_trend_plot_single(processed_df: pd.DataFrame, selected_depths: list, point_name: str) -> go.Figure:
    """Trend plot for a single IPIS point."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("<b>A-Axis Time History</b>", "<b>B-Axis Time History</b>"),
        horizontal_spacing=0.10,
    )
    all_depths = sorted(processed_df["depth"].unique())

    for i, depth in enumerate(selected_depths):
        closest = min(all_depths, key=lambda d: abs(d - depth))
        data = processed_df.loc[processed_df["depth"] == closest].sort_values("timestamp")
        if data.empty:
            continue
        color = CHART_COLORS[i % len(CHART_COLORS)]
        common = dict(mode="lines+markers", line=dict(color=color, width=2), marker=dict(size=4), legendgroup=f"d{i}")

        fig.add_trace(
            go.Scatter(x=data["timestamp"], y=data["cum_disp_a"], name=f"{closest:.1f}m", showlegend=True,
                       hovertemplate="<b>Time:</b> %{x}<br><b>A-Axis:</b> %{y:.3f} mm<extra></extra>", **common),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=data["timestamp"], y=data["cum_disp_b"], name=f"{closest:.1f}m", showlegend=False,
                       hovertemplate="<b>Time:</b> %{x}<br><b>B-Axis:</b> %{y:.3f} mm<extra></extra>", **common),
            row=1, col=2,
        )

    fig.add_hline(y=0, line_dash="dash", line_color="#64748b", line_width=1.5, row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#64748b", line_width=1.5, row=1, col=2)

    fig.update_layout(
        title=_title_dict(f"<b>{point_name} – Displacement Time History</b>"),
        legend=dict(y=-0.15, title=dict(text="<b>Depth:</b> ", font=dict(size=10)), **_LEGEND_STYLE),
        height=450, margin=dict(t=60, b=80, l=70, r=50), hovermode="x unified", **_LAYOUT_COLORS,
    )
    for ann in fig["layout"]["annotations"]:
        ann["y"] = 1.02
        ann["font"] = dict(size=12, color="#1e293b")

    fig.update_xaxes(title_text="Date/Time", **_AXIS_STYLE, row=1, col=1)
    fig.update_xaxes(title_text="Date/Time", **_AXIS_STYLE, row=1, col=2)
    fig.update_yaxes(title_text="Displacement (mm)", zeroline=True, zerolinecolor="#9ca3af", **_AXIS_STYLE, row=1, col=1)
    fig.update_yaxes(title_text="Displacement (mm)", zeroline=True, zerolinecolor="#9ca3af", **_AXIS_STYLE, row=1, col=2)
    return fig


def create_trend_comparison(points_data: Dict[str, pd.DataFrame], selected_depth: float, axis: str = "A") -> go.Figure:
    """Comparative trend across multiple points at a given depth."""
    fig = go.Figure()
    disp_col = "cum_disp_a" if axis == "A" else "cum_disp_b"

    for i, (name, df) in enumerate(points_data.items()):
        all_depths = sorted(df["depth"].unique())
        closest = min(all_depths, key=lambda d: abs(d - selected_depth))
        data = df.loc[df["depth"] == closest].sort_values("timestamp")
        if data.empty:
            continue
        color = CHART_COLORS[i % len(CHART_COLORS)]
        fig.add_trace(go.Scatter(
            x=data["timestamp"], y=data[disp_col], mode="lines+markers",
            name=f"{name} ({closest:.1f}m)", line=dict(color=color, width=2), marker=dict(size=4),
            hovertemplate=f"<b>{name}</b><br>Time: %{{x}}<br>{axis}-Axis: %{{y:.3f}} mm<extra></extra>",
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="#64748b", line_width=1.5)
    fig.update_layout(
        title=_title_dict(f"<b>Multi-Point {axis}-Axis Comparison @ ~{selected_depth:.1f}m Depth</b>"),
        xaxis=dict(title="Date/Time", gridcolor="#e5e7eb"),
        yaxis=dict(title="Cumulative Displacement (mm)", gridcolor="#e5e7eb", zeroline=True, zerolinecolor="#9ca3af"),
        legend=dict(y=-0.15, **_LEGEND_STYLE),
        height=450, margin=dict(t=60, b=100, l=70, r=50), hovermode="x unified", **_LAYOUT_COLORS,
    )
    return fig


# =============================================================================
# SESSION STATE MANAGEMENT
# =============================================================================
def init_session_state():
    if "ipis_points" not in st.session_state:
        st.session_state.ipis_points: Dict[str, IPISPoint] = {}
    if "processed_data" not in st.session_state:
        st.session_state.processed_data: Dict[str, pd.DataFrame] = {}
    if "config_hashes" not in st.session_state:
        st.session_state.config_hashes: Dict[str, str] = {}
    if "axis_range" not in st.session_state:
        st.session_state.axis_range = dict(DEFAULT_AXIS_RANGE)


def generate_point_id(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()[:8]


def add_ipis_point(file_content: str, filename: str) -> Tuple[bool, str]:
    """Add a new IPIS point from file content."""
    if len(st.session_state.ipis_points) >= MAX_IPIS_POINTS:
        return False, f"Maximum limit of {MAX_IPIS_POINTS} points reached. Remove a point first."

    try:
        df, metadata = parse_toa5_file(file_content)
        detected_cols = detect_ipi_columns(df)

        if detected_cols["num_sensors"] == 0:
            return False, f"Could not detect IPI sensor columns in {filename}"

        point_id = generate_point_id(file_content)
        if point_id in st.session_state.ipis_points:
            return False, "This file appears to already be loaded (duplicate detected)"

        point_name = metadata.get("station_name", filename.replace(".dat", "").replace(".DAT", ""))
        color_idx = len(st.session_state.ipis_points) % len(CHART_COLORS)
        num_sensors = detected_cols["num_sensors"]

        point = IPISPoint(
            point_id=point_id,
            name=point_name,
            raw_df=df,
            metadata=metadata,
            gauge_lengths=np.full(num_sensors, DEFAULT_GAUGE_LENGTH),
            detected_cols=detected_cols,
            num_sensors=num_sensors,
            color=CHART_COLORS[color_idx],
        )
        st.session_state.ipis_points[point_id] = point

        fmt_desc = "2D array format" if detected_cols.get("format_type") == "new_2d" else "standard format"
        return True, f"Loaded: {point_name} ({num_sensors} sensors, {len(df)} records, {fmt_desc})"
    except Exception as e:
        return False, f"Error parsing {filename}: {e}"


def remove_ipis_point(point_id: str):
    st.session_state.ipis_points.pop(point_id, None)
    st.session_state.processed_data.pop(point_id, None)
    st.session_state.config_hashes.pop(point_id, None)


def process_all_points(use_raw_tilt: bool = True):
    """Process points only when their configuration has changed."""
    for pid, point in st.session_state.ipis_points.items():
        current_hash = point.config_hash
        if st.session_state.config_hashes.get(pid) == current_hash and pid in st.session_state.processed_data:
            continue  # skip — nothing changed
        result = process_ipis_point(point, use_raw_tilt)
        st.session_state.processed_data[pid] = result
        st.session_state.config_hashes[pid] = current_hash


# =============================================================================
# SIDEBAR: GAUGE LENGTH CONFIGURATION
# =============================================================================
def _render_gauge_config(point: IPISPoint, point_id: str):
    """Render per-sensor gauge length controls inside an expander."""
    depths = compute_depths(point.top_depth, point.gauge_lengths)

    unique_gauges = np.unique(point.gauge_lengths)
    if len(unique_gauges) == 1:
        summary = f"All sensors: `{unique_gauges[0]:.0f} m`"
    else:
        summary = f"Mixed: {', '.join(f'{g:.0f}m' for g in unique_gauges)}"
    st.caption(f"Current: {summary}")

    # Quick-set buttons
    st.markdown("**Set ALL Sensors:**")
    qc1, qc2, qc3 = st.columns(3)
    for col_widget, label, val in [(qc1, "All 1m", 1.0), (qc2, "All 2m", 2.0), (qc3, "All 3m", 3.0)]:
        with col_widget:
            if st.button(label, key=f"all{int(val)}_{point_id}", use_container_width=True):
                point.gauge_lengths = np.full(point.num_sensors, val)
                st.session_state.processed_data.pop(point_id, None)
                st.session_state.config_hashes.pop(point_id, None)
                st.rerun()

    # Per-sensor grid
    st.markdown("**Per-Sensor Gauge Length:**")
    num_rows = (point.num_sensors + SENSORS_PER_ROW - 1) // SENSORS_PER_ROW
    gauge_changed = False
    new_gl = point.gauge_lengths.copy()

    for row_i in range(num_rows):
        cols = st.columns(SENSORS_PER_ROW)
        for col_i, col_w in enumerate(cols):
            s_idx = row_i * SENSORS_PER_ROW + col_i
            if s_idx >= point.num_sensors:
                break
            with col_w:
                cur = point.gauge_lengths[s_idx]
                idx_default = GAUGE_LENGTH_OPTIONS.index(cur) if cur in GAUGE_LENGTH_OPTIONS else 2
                new_val = st.selectbox(
                    f"S{s_idx + 1} ({depths[s_idx]:.1f}m)",
                    options=GAUGE_LENGTH_OPTIONS,
                    index=idx_default,
                    format_func=lambda x: f"{int(x)}m",
                    key=f"gs_{point_id}_{s_idx}",
                )
                if new_val != cur:
                    new_gl[s_idx] = new_val
                    gauge_changed = True

    if gauge_changed:
        point.gauge_lengths = new_gl
        st.session_state.processed_data.pop(point_id, None)
        st.session_state.config_hashes.pop(point_id, None)


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    init_session_state()

    # Header
    st.markdown('<div class="main-header">📊 Multi-Point IPI Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">In-Place Inclinometer Monitoring – Multiple Points Analysis</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SIDEBAR
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Configuration")

        num_points = len(st.session_state.ipis_points)
        counter_cls = "point-counter-full" if num_points >= MAX_IPIS_POINTS else ""
        st.markdown(f'<span class="point-counter {counter_cls}">{num_points} / {MAX_IPIS_POINTS} Points</span>', unsafe_allow_html=True)
        st.divider()

        # 1. File upload
        st.subheader("1. Upload IPIS Data Files")
        if num_points >= MAX_IPIS_POINTS:
            st.error(f"⚠️ Maximum limit of {MAX_IPIS_POINTS} points reached!")
            st.info("Remove existing points to add new ones.")
        else:
            uploaded_files = st.file_uploader(
                "Upload .DAT Files", type=["dat", "csv"], accept_multiple_files=True,
                help=f"Upload Campbell Scientific TOA5 format files. Max {MAX_IPIS_POINTS} total points.",
                key="file_uploader",
            )
            if uploaded_files:
                for uf in uploaded_files:
                    if len(st.session_state.ipis_points) >= MAX_IPIS_POINTS:
                        st.warning(f"Skipped {uf.name}: Maximum points reached")
                        continue
                    content = uf.read().decode("utf-8")
                    ok, msg = add_ipis_point(content, uf.name)
                    (st.success if ok else st.error)(msg)

        st.divider()

        # 2. Processing options
        st.subheader("2. Processing Options")
        data_source = st.radio(
            "Data Source", options=["Raw Tilt (sin θ)", "Pre-calculated Deflection"], index=0,
            help="Select data source for displacement calculation",
        )
        use_raw_tilt = data_source == "Raw Tilt (sin θ)"
        st.divider()

        # 3. Axis range
        st.subheader("3. Plot Axis Settings")
        auto_range = st.checkbox("Auto X-Axis Range", value=st.session_state.axis_range.get("auto", True),
                                 help="Automatically scale X-axis based on data")
        st.session_state.axis_range["auto"] = auto_range

        if not auto_range:
            cmin, cmax = st.columns(2)
            with cmin:
                st.session_state.axis_range["min"] = st.number_input(
                    "X-Axis Min (mm)", value=float(st.session_state.axis_range["min"]), step=5.0, key="axis_min")
            with cmax:
                st.session_state.axis_range["max"] = st.number_input(
                    "X-Axis Max (mm)", value=float(st.session_state.axis_range["max"]), step=5.0, key="axis_max")

            st.caption("Quick Presets:")
            pcols = st.columns(4)
            for col_w, val in zip(pcols, [25, 50, 100, 200]):
                with col_w:
                    if st.button(f"±{val}", key=f"preset_{val}", use_container_width=True):
                        st.session_state.axis_range["min"] = -float(val)
                        st.session_state.axis_range["max"] = float(val)
                        st.rerun()

        st.divider()

        # 4. Loaded points
        st.subheader("4. Loaded Points")
        if st.session_state.ipis_points:
            for point_id, point in list(st.session_state.ipis_points.items()):
                with st.expander(f"📍 {point.name}", expanded=False):
                    fmt_type = point.detected_cols.get("format_type", "unknown")
                    badge = "🔷 2D Array" if fmt_type == "new_2d" else "🔶 Standard"
                    st.caption(f"{badge} | Sensors: {point.num_sensors} | Records: {len(point.raw_df)}")

                    st.markdown("**⚙️ Gauge Length Configuration**")
                    _render_gauge_config(point, point_id)
                    st.divider()

                    # Top depth
                    point.top_depth = st.number_input(
                        "Top Depth (m)", min_value=0.0, max_value=100.0,
                        value=float(point.top_depth), step=0.5, key=f"td_{point_id}")

                    # Base reading — use DataFrame integer index for reliable alignment
                    ts_series = point.raw_df[point.detected_cols["timestamp"]].sort_values()
                    base_options = [pd.Timestamp(t).strftime("%Y-%m-%d %H:%M") for t in ts_series.unique()]
                    st.markdown(f"**Base Reading** ({len(base_options)} timestamps available)")
                    selected_base = st.selectbox(
                        "Base Reading", options=base_options, index=0,
                        key=f"base_{point_id}", label_visibility="collapsed")
                    point.base_reading_idx = base_options.index(selected_base)

                    if st.button("🗑️ Remove", key=f"del_{point_id}", type="secondary"):
                        remove_ipis_point(point_id)
                        st.rerun()
        else:
            st.info("No points loaded. Upload .DAT files above.")

    # -------------------------------------------------------------------------
    # MAIN CONTENT
    # -------------------------------------------------------------------------
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

    # Process (skips unchanged points automatically)
    process_all_points(use_raw_tilt)

    if not st.session_state.processed_data:
        st.error("No data could be processed. Please check your files.")
        return

    # Point selector
    st.subheader("📍 Select Points to Display")
    all_names = {pid: p.name for pid, p in st.session_state.ipis_points.items()}
    selected_ids = st.multiselect(
        "Select IPIS Points", options=list(all_names), default=list(all_names)[:5],
        format_func=lambda x: all_names[x], help="Select which points to display")

    if not selected_ids:
        st.warning("Please select at least one IPIS point to display.")
        return

    selected_data = {
        st.session_state.ipis_points[pid].name: st.session_state.processed_data[pid]
        for pid in selected_ids if pid in st.session_state.processed_data
    }

    st.divider()

    # ---- Tabs ----
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Individual Profiles", "📊 Compare Profiles",
        "📉 Individual Trends", "📋 Compare Trends",
    ])

    # Tab 1: Individual Profile Plots
    with tab1:
        st.subheader("Individual Displacement Profiles")
        pf_id = st.selectbox("Select Point", options=selected_ids, format_func=lambda x: all_names[x], key="profile_point")
        df = st.session_state.processed_data[pf_id]
        ts_list = sorted(df["timestamp"].unique())

        c1, c2 = st.columns([3, 1])
        with c1:
            default_ts = [ts_list[0], ts_list[-1]] if len(ts_list) > 1 else ts_list[:1]
            sel_ts = st.multiselect(
                "Select Timestamps", options=ts_list, default=default_ts,
                format_func=lambda x: pd.Timestamp(x).strftime("%Y-%m-%d %H:%M"),
                max_selections=8, key="profile_timestamps")
        with c2:
            if st.button("Latest", key="latest_btn"):
                sel_ts = [ts_list[-1]]

        if sel_ts:
            fig = create_profile_plot_single(df, sel_ts, all_names[pf_id], axis_range=st.session_state.axis_range)
            st.plotly_chart(fig, use_container_width=True)

    # Tab 2: Compare Profiles
    with tab2:
        st.subheader("Compare Profiles Across Points")
        if len(selected_data) < 2:
            st.info("Select at least 2 points above to compare profiles.")
        else:
            all_ts = sorted({t for df in selected_data.values() for t in df["timestamp"].unique()})
            cmp_ts = st.select_slider(
                "Select Timestamp for Comparison", options=all_ts, value=all_ts[-1],
                format_func=lambda x: pd.Timestamp(x).strftime("%Y-%m-%d %H:%M"), key="compare_timestamp")
            cmp_axis = st.radio("Select Axis", options=["A", "B"], horizontal=True, key="compare_axis")
            fig = create_profile_plot_comparison(selected_data, cmp_ts, cmp_axis, axis_range=st.session_state.axis_range)
            st.plotly_chart(fig, use_container_width=True)

    # Tab 3: Individual Trends
    with tab3:
        st.subheader("Individual Displacement Trends")
        tr_id = st.selectbox("Select Point", options=selected_ids, format_func=lambda x: all_names[x], key="trend_point")
        df = st.session_state.processed_data[tr_id]
        avail_d = sorted(df["depth"].unique())
        default_depths = [avail_d[0], avail_d[len(avail_d) // 2], avail_d[-1]]
        sel_depths = st.multiselect(
            "Select Depths", options=avail_d, default=default_depths,
            format_func=lambda x: f"{x:.1f} m", max_selections=6, key="trend_depths")
        if sel_depths:
            fig = create_trend_plot_single(df, sel_depths, all_names[tr_id])
            st.plotly_chart(fig, use_container_width=True)

    # Tab 4: Compare Trends
    with tab4:
        st.subheader("Compare Trends Across Points")
        if len(selected_data) < 2:
            st.info("Select at least 2 points above to compare trends.")
        else:
            all_d = sorted({d for df in selected_data.values() for d in df["depth"].unique()})
            cmp_d = st.select_slider(
                "Select Depth for Comparison", options=all_d, value=all_d[len(all_d) // 2],
                format_func=lambda x: f"{x:.1f} m", key="compare_depth")
            cmp_ax_t = st.radio("Select Axis", options=["A", "B"], horizontal=True, key="compare_axis_trend")
            fig = create_trend_comparison(selected_data, cmp_d, cmp_ax_t)
            st.plotly_chart(fig, use_container_width=True)

    # ---- Summary ----
    st.divider()
    st.subheader("📊 Summary Statistics")

    rows = []
    for pid in selected_ids:
        point = st.session_state.ipis_points[pid]
        df = st.session_state.processed_data.get(pid)
        if df is None or df.empty:
            continue
        latest = df.loc[df["timestamp"] == df["timestamp"].max()]
        rows.append({
            "Point": point.name,
            "Sensors": point.num_sensors,
            "Records": len(point.raw_df),
            "Max A (mm)": f"{latest['cum_disp_a'].abs().max():.2f}",
            "Max B (mm)": f"{latest['cum_disp_b'].abs().max():.2f}",
            "Max Resultant (mm)": f"{latest['cum_disp_resultant'].max():.2f}",
            "Date Range": f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}",
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
