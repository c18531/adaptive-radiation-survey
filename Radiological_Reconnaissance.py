"""
Created on Sun MAR 12 13:04:48 2026

@author: wooseok
"""

from __future__ import annotations

import argparse
import math
import os
import copy
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import warnings
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import shift as ndi_shift, gaussian_filter, binary_fill_holes, binary_dilation
from scipy.signal import fftconvolve
from scipy.stats import norm
from shapely import affinity
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.prepared import prep
import geopandas as gpd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ---------------------------------------------------------------------
# Geometry / shapefile helpers
# ---------------------------------------------------------------------


def ensure_closed_polygon(poly: np.ndarray) -> np.ndarray:
    poly = np.asarray(poly, dtype=float)
    if poly.ndim != 2 or poly.shape[1] < 2:
        raise ValueError("polygon must have shape (n,2)")
    poly = poly[:, :2]
    if len(poly) == 0:
        return poly.reshape(0, 2)
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    return poly

def polygon_area_xy(poly: np.ndarray) -> float:
    poly = ensure_closed_polygon(poly)
    pts = poly[:-1]
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

def polygon_bbox(poly: np.ndarray) -> Dict[str, float]:
    poly = ensure_closed_polygon(poly)
    pts = poly[:-1]
    return {
        "xmin": float(np.min(pts[:, 0])),
        "xmax": float(np.max(pts[:, 0])),
        "ymin": float(np.min(pts[:, 1])),
        "ymax": float(np.max(pts[:, 1])),
    }

def read_dxf_polylines(path: str) -> List[Dict[str, object]]:
    lines = Path(path).read_text(errors="ignore").splitlines()
    if len(lines) % 2 != 0:
        lines = lines[:-1]
    pairs = list(zip(lines[0::2], lines[1::2]))

    in_entities = False
    entities: List[Dict[str, object]] = []
    i = 0
    while i < len(pairs):
        code, val = pairs[i]
        if code == "0" and val == "SECTION":
            if i + 1 < len(pairs) and pairs[i + 1][0] == "2":
                in_entities = pairs[i + 1][1] == "ENTITIES"
            i += 1
            continue
        if code == "0" and val == "ENDSEC":
            in_entities = False
            i += 1
            continue

        if in_entities and code == "0" and val == "POLYLINE":
            layer = None
            flags = 0
            verts: List[Tuple[float, float, float]] = []
            i += 1
            while i < len(pairs):
                c, v = pairs[i]
                if c == "8":
                    layer = v
                elif c == "70":
                    try:
                        flags = int(float(v))
                    except Exception:
                        flags = 0
                elif c == "0" and v == "VERTEX":
                    x = y = z = None
                    i += 1
                    while i < len(pairs):
                        c2, v2 = pairs[i]
                        if c2 == "10":
                            x = float(v2)
                        elif c2 == "20":
                            y = float(v2)
                        elif c2 == "30":
                            z = float(v2)
                        elif c2 == "0":
                            break
                        i += 1
                    if x is not None and y is not None:
                        verts.append((x, y, 0.0 if z is None else z))
                    continue
                elif c == "0" and v == "SEQEND":
                    break
                i += 1
            if verts:
                entities.append({
                    "layer": layer or f"polyline_{len(entities)+1}",
                    "closed": bool(flags & 1),
                    "verts": verts,
                })
        i += 1
    return entities

def infer_zone_polygons_from_dxf(path: str, zone_mode: str = "auto") -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, str]:
    entities = read_dxf_polylines(path)
    if len(entities) < 1:
        raise ValueError("DXF must contain at least one closed POLYLINE entity")
    rows = []
    polys = []
    for ent in entities:
        poly = ensure_closed_polygon(np.asarray(ent["verts"], dtype=float)[:, :2])
        if len(poly) < 4:
            continue
        area = polygon_area_xy(poly)
        rows.append({"layer": str(ent["layer"]), "area": area, **polygon_bbox(poly)})
        polys.append((area, poly))
    if not polys:
        raise ValueError("DXF does not contain any usable closed POLYLINE entity")
    polys.sort(key=lambda x: x[0], reverse=True)
    zone_mode = str(zone_mode).strip().lower()
    if zone_mode not in {"auto", "single", "dual"}:
        raise ValueError(f"Unsupported zone_mode: {zone_mode}")

    warm_poly = polys[0][1]
    if zone_mode == "single":
        hot_poly = np.zeros((0, 2), dtype=float)
        resolved_mode = "single"
    elif zone_mode == "dual":
        if len(polys) < 2:
            raise ValueError("zone_mode='dual' requires at least two closed POLYLINE entities in the DXF")
        hot_poly = polys[1][1]
        resolved_mode = "dual"
    else:
        if len(polys) >= 2:
            hot_poly = polys[1][1]
            resolved_mode = "dual"
        else:
            hot_poly = np.zeros((0, 2), dtype=float)
            resolved_mode = "single"

    summary = pd.DataFrame(rows).sort_values("area", ascending=False).reset_index(drop=True)
    return warm_poly, hot_poly, summary, resolved_mode

def _empty_geom() -> BaseGeometry:
    return GeometryCollection()

def is_geometry(obj: object) -> bool:
    return isinstance(obj, BaseGeometry)

def to_geometry(obj: object) -> BaseGeometry:
    if obj is None:
        return _empty_geom()
    if isinstance(obj, BaseGeometry):
        return obj
    arr = np.asarray(obj, dtype=float)
    if arr.size == 0:
        return _empty_geom()
    arr = ensure_closed_polygon(arr)
    if len(arr) < 4:
        return _empty_geom()
    return Polygon(arr)

def largest_polygon_component(geom: BaseGeometry) -> BaseGeometry:
    geom = to_geometry(geom)
    if geom.is_empty:
        return geom
    if isinstance(geom, Polygon):
        return geom
    polys = []
    if isinstance(geom, MultiPolygon):
        polys = [g for g in geom.geoms if not g.is_empty]
    elif hasattr(geom, "geoms"):
        polys = [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]
    return max(polys, key=lambda g: g.area) if polys else geom

def geometry_to_array(geom: object) -> np.ndarray:
    geom = to_geometry(geom)
    if geom.is_empty:
        return np.zeros((0, 2), dtype=float)
    geom = largest_polygon_component(geom)
    if isinstance(geom, Polygon):
        return np.asarray(geom.exterior.coords, dtype=float)
    return np.zeros((0, 2), dtype=float)

def points_in_polygon(points: np.ndarray, poly: object, include_boundary: bool = True) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    geom = to_geometry(poly)
    if geom.is_empty:
        return np.zeros(len(pts), dtype=bool)
    pg = prep(geom)
    return np.fromiter(
        (
            pg.covers(Point(float(x), float(y)))
            if include_boundary else
            pg.contains(Point(float(x), float(y)))
            for x, y in pts[:, :2]
        ),
        dtype=bool,
        count=len(pts),
    )

def scale_polygon_about_origin(poly: object, scale: float, origin: Tuple[float, float] = (0.0, 0.0)) -> object:
    if is_geometry(poly):
        if poly.is_empty:
            return poly
        return affinity.scale(poly, xfact=float(scale), yfact=float(scale), origin=origin)
    poly = ensure_closed_polygon(np.asarray(poly, dtype=float))
    ox, oy = origin
    out = poly.copy()
    out[:, 0] = ox + float(scale) * (out[:, 0] - ox)
    out[:, 1] = oy + float(scale) * (out[:, 1] - oy)
    return ensure_closed_polygon(out)

def _normalize_label_token(value: object) -> str:
    return str(value).strip().lower()

def validate_shapefile_triplet(shp_path: str, shx_path: Optional[str] = None, dbf_path: Optional[str] = None) -> Tuple[str, str, str]:
    shp = Path(shp_path)
    shx = Path(shx_path) if shx_path else shp.with_suffix('.shx')
    dbf = Path(dbf_path) if dbf_path else shp.with_suffix('.dbf')
    missing = [str(p) for p in (shp, shx, dbf) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing shapefile component(s): {', '.join(missing)}")
    return str(shp), str(shx), str(dbf)

def infer_shapefile_label_column(gdf: gpd.GeoDataFrame, requested: Optional[str] = None) -> str:
    if requested is not None:
        if requested not in gdf.columns:
            raise ValueError(f"Requested shapefile label column not found: {requested}")
        return requested
    preferred = ['LABEL', 'label', 'Area', 'AREA', 'Name', 'NAME', 'Zone', 'ZONE', 'Type', 'TYPE', 'ID', 'id']
    for col in preferred:
        if col in gdf.columns:
            return col
    for col in gdf.columns:
        if col == 'geometry':
            continue
        if pd.api.types.is_string_dtype(gdf[col]) or pd.api.types.is_object_dtype(gdf[col]):
            return str(col)
    for col in gdf.columns:
        if col != 'geometry':
            return str(col)
    raise ValueError('Could not infer a label column from the shapefile.')

def _parse_label_selector(selector: Optional[str]) -> List[str]:
    if selector is None:
        return []
    raw = str(selector).strip()
    if not raw:
        return []
    return [tok.strip() for tok in raw.split(',') if tok.strip()]

def _selector_is_all(tokens: Sequence[str]) -> bool:
    return any(_normalize_label_token(tok) in {'*', 'all'} for tok in tokens)

def select_shapefile_geometry(gdf: gpd.GeoDataFrame, label_col: str, selector: Optional[str], *, default_to_all: bool = False) -> BaseGeometry:
    tokens = _parse_label_selector(selector)
    if len(tokens) == 0 and default_to_all:
        subset = gdf
    elif _selector_is_all(tokens):
        subset = gdf
    else:
        normalized = gdf[label_col].astype(str).map(_normalize_label_token)
        keep = np.zeros(len(gdf), dtype=bool)
        for tok in tokens:
            keep |= normalized == _normalize_label_token(tok)
        subset = gdf.loc[keep]
    if len(subset) == 0:
        return _empty_geom()
    geom = unary_union([g for g in subset.geometry if g is not None and not g.is_empty])
    return to_geometry(geom)

def default_hot_label_selector(gdf: gpd.GeoDataFrame, label_col: str) -> Optional[str]:
    labels = gdf[label_col].astype(str)
    norm = labels.map(_normalize_label_token)
    for token in ['detonation site', 'detonation', 'hot', 'source', 'source area']:
        hit = labels[norm == token]
        if len(hit) > 0:
            return str(hit.iloc[0])
    areas = np.asarray([geom.area if geom is not None else np.nan for geom in gdf.geometry], dtype=float)
    valid = np.where(np.isfinite(areas) & (areas > 0))[0]
    if len(valid) == 0:
        return None
    idx = int(valid[np.argmin(areas[valid])])
    return str(labels.iloc[idx])

def read_zone_geometries_from_shapefile(
    shp_path: str,
    zone_mode: str = 'auto',
    *,
    shx_path: Optional[str] = None,
    dbf_path: Optional[str] = None,
    label_col: Optional[str] = None,
    survey_labels: Optional[str] = None,
    warm_labels: Optional[str] = None,
    hot_labels: Optional[str] = None,
) -> Tuple[BaseGeometry, BaseGeometry, BaseGeometry, pd.DataFrame, str, str]:
    shp_path, shx_path, dbf_path = validate_shapefile_triplet(shp_path, shx_path=shx_path, dbf_path=dbf_path)
    gdf = gpd.read_file(shp_path)
    if len(gdf) == 0:
        raise ValueError('Shapefile does not contain any geometries.')
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    if len(gdf) == 0:
        raise ValueError('Shapefile does not contain any non-empty geometries.')

    label_col = infer_shapefile_label_column(gdf, requested=label_col)
    zone_mode = str(zone_mode).strip().lower()
    if zone_mode not in {'auto', 'single', 'dual'}:
        raise ValueError(f'Unsupported zone_mode: {zone_mode}')

    if survey_labels is None or not str(survey_labels).strip():
        survey_labels = 'all'
    if warm_labels is None or not str(warm_labels).strip():
        warm_labels = survey_labels
    if hot_labels is None or not str(hot_labels).strip():
        hot_labels = default_hot_label_selector(gdf, label_col)

    survey_geom = select_shapefile_geometry(gdf, label_col, survey_labels, default_to_all=True)
    warm_geom = select_shapefile_geometry(gdf, label_col, warm_labels, default_to_all=True)
    hot_geom = select_shapefile_geometry(gdf, label_col, hot_labels, default_to_all=False) if hot_labels else _empty_geom()

    if survey_geom.is_empty:
        raise ValueError('Survey geometry selection from shapefile is empty.')
    if warm_geom.is_empty:
        raise ValueError('Warm prior geometry selection from shapefile is empty.')

    if zone_mode == 'single':
        hot_geom = _empty_geom()
        resolved_mode = 'single'
    elif zone_mode == 'dual':
        if hot_geom.is_empty:
            raise ValueError("zone_mode='dual' requires a non-empty hot area selection from the shapefile.")
        resolved_mode = 'dual'
    else:
        resolved_mode = 'dual' if not hot_geom.is_empty else 'single'

    survey_sel = select_shapefile_geometry(gdf, label_col, survey_labels, default_to_all=True)
    warm_sel = select_shapefile_geometry(gdf, label_col, warm_labels, default_to_all=True)
    hot_sel = select_shapefile_geometry(gdf, label_col, hot_labels, default_to_all=False) if hot_labels else _empty_geom()
    survey_prepped = prep(survey_sel) if not survey_sel.is_empty else None
    warm_prepped = prep(warm_sel) if not warm_sel.is_empty else None
    hot_prepped = prep(hot_sel) if not hot_sel.is_empty else None
    rows = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        xmin, ymin, xmax, ymax = geom.bounds
        rep = geom.representative_point()
        rows.append({
            'label': str(row[label_col]),
            'area': float(geom.area),
            'xmin': float(xmin),
            'xmax': float(xmax),
            'ymin': float(ymin),
            'ymax': float(ymax),
            'survey_selected': bool(survey_prepped.covers(rep)) if survey_prepped is not None else False,
            'warm_selected': bool(warm_prepped.covers(rep)) if warm_prepped is not None else False,
            'hot_selected': bool(hot_prepped.covers(rep)) if hot_prepped is not None else False,
        })
    summary = pd.DataFrame(rows).sort_values('area', ascending=False).reset_index(drop=True)
    return survey_geom, warm_geom, hot_geom, summary, resolved_mode, label_col

def read_vsp_csv(path: str, return_frame: bool = False) -> object:
    df = pd.read_csv(path)
    cols = [str(c).lower().strip() for c in df.columns]
    df = df.copy()
    df.columns = cols

    x_col = y_col = None
    if 'x coord' in cols and 'y coord' in cols:
        x_col, y_col = 'x coord', 'y coord'
    elif 'x' in cols and 'y' in cols:
        x_col, y_col = 'x', 'y'
    else:
        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        if len(num_cols) < 2:
            raise ValueError(f"Could not detect x/y columns in {path}")
        x_col, y_col = num_cols[:2]

    df = df.copy()
    df['__x__'] = pd.to_numeric(df[x_col], errors='coerce')
    df['__y__'] = pd.to_numeric(df[y_col], errors='coerce')
    df = df.dropna(subset=['__x__', '__y__']).reset_index(drop=True)

    if return_frame:
        return df
    return df[['__x__', '__y__']].to_numpy(dtype=float)

def baseline_points_from_frame(df: pd.DataFrame) -> np.ndarray:
    if df is None or len(df) == 0:
        return np.zeros((0, 2), dtype=float)
    if '__x__' not in df.columns or '__y__' not in df.columns:
        raise ValueError('Baseline dataframe must include __x__ and __y__ columns.')
    return df[['__x__', '__y__']].to_numpy(dtype=float)


# ---------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------


@dataclass
class FieldConfig:
    bounds: Tuple[float, float, float, float]
    grid_n: int = 201
    background_level: float = 0.02
    meas_sigma: float = 0.05
    measurement_diameter_m: float = 22.0
    measurement_model: str = "disk_avg"

@dataclass
class OperationalCriteria:
    thresholds: Tuple[float, ...] = (1.0, 5.0)
    confidence: float = 0.975
    labels: Tuple[str, ...] = ("Cold", "Warm", "Hot")
    colors: Tuple[str, ...] = ("#9fd39f", "#f4c06a", "#e96b63")

@dataclass
class TruthConfig:
    truth_scale: float = 0.5
    truth_scale_min: Optional[float] = None
    truth_scale_max: Optional[float] = None
    truth_scale_random_mode: str = "fixed"   
    origin_x: float = 0.0
    origin_y: float = 0.0
    n_sources_min: int = 1
    n_sources_max: int = 2
    plume_probability: float = 1.0
    prior_center_bias: float = 4.0
    amp_min: float = 3.6
    amp_max: float = 5.6
    value_scale: float = 1.0
    blur_sigma_cells: float = 2.0
    theta_jitter_deg: float = 8.0
    shape_follow_strength: float = 0.75
    shape_fill_fraction: float = 0.10
    prior_softness: float = 0.45
    meander_strength: float = 0.20
    roughness_strength: float = 0.20
    roughness_sigma_cells: float = 6.0
    lock_hot_anchor: bool = True
    hot_anchor_x: float = 0.0
    hot_anchor_y: float = 0.0
    meteo_tilt_max_deg: float = 5.0
    preserve_hot_core_during_warp: bool = True

@dataclass
class BaselineConfig:
    warm_spacing_m: float = 19.0
    hot_spacing_m: float = 13.0
    vsp_serpentine_axis: str = "col"
    vsp_line_tol_m: float = 22.0
    vsp_end_corner: str = "upper_right"
    max_samples: int = 0
    candidate_source: str = "auto"
    vsp_apply_valid_center: bool = False
    cross_phase_prune_enabled: bool = True
    cross_phase_prune_radius_m: float = 22.0

@dataclass
class AdaptiveConfig:
    n0: int = 6
    max_samples: int = 30
    candidate_spacing_m: float = 14.0
    min_spacing_m: float = 16.0
    epsilon: float = 0.05
    gamma: float = 0.01
    candidate_stride: int = 1
    hot_prior_weight: float = 0.10
    gp_restarts: int = 2
    max_step_m: float = 34.0
    travel_weight: float = 0.30
    frontier_weight: float = 0.20
    turn_weight: float = 0.08
    global_every: int = 6
    order_initial_points: bool = True
    candidate_source: str = "auto"        
    vsp_row_tol_m: float = 8.0             
    vsp_serpentine_axis: str = "row"      
    vsp_lookahead: int = 10
    vsp_skip_weight: float = 0.18
    stop_acq: float = -1.0                 
    stop_min_samples: int = 0
    acq_mode: str = "posterior"        
    SEED_START_XY = np.array([161.623, 189.213], dtype=float)
    ROLE_WARMHOT_DIST_M = 50.0
    ROLE_FINAL_TAIL_DIST_M = 50.0
    oracle_top_k: int = 25
    oracle_n_repl: int = 3

    metric_stop_enabled: bool = False
    metric_stop_zone_accuracy_target: float = 0.84
    metric_stop_hot_recall_target: float = 0.74
    metric_stop_action_recall_target: float = 0.80
    metric_stop_patience: int = 2
    
@dataclass
class SurveyGeometry:
    survey_poly: BaseGeometry
    survey_geom: BaseGeometry
    warm_poly: BaseGeometry
    hot_poly: BaseGeometry
    warm_geom: BaseGeometry
    hot_geom: BaseGeometry
    valid_center_geom: BaseGeometry
    valid_center_hot_geom: BaseGeometry

    cfg: FieldConfig
    boundary_summary: Optional[pd.DataFrame]
    has_hot_prior: bool
    zone_mode: str

@dataclass
class RunResult:
    name: str
    X_obs: np.ndarray
    y_obs: np.ndarray
    mu_grid: np.ndarray
    std_grid: np.ndarray
    pred_zone: np.ndarray
    truth_zone: np.ndarray
    zone_accuracy: float
    hot_recall: float
    action_recall: float
    action_precision: float
    n_used: int
    path_length_m: float
    travel_time_min: float
    station_time_total_min: float
    total_time_min: float
    diag: Optional[Dict[str, object]] = None
    
@dataclass
class DecisionWeightConfig:
    w_t: float = 1.0
    w_fr: float = 1.0
    w_fh: float = 0.30
    t_ref: float = 0.0

@dataclass
class MeteoStep:
    t0_h: float
    t1_h: float
    wind_dir_deg: float
    wind_speed_mps: float
    stability: str = "D"
    rain_mmph: float = 0.0

@dataclass
class SourceTerm:
    start_h: float = 0.0
    end_h: float = 1.0
    release_rate: float = 1.0
    dry_dep_vd: float = 0.002
    wet_scavenging: float = 0.03
    decay_lambda: float = 0.0
    source_sigma_m: float = 12.0

@dataclass
class ForecastConfig:
    enabled: bool = False
    horizon_h: float = 96.0
    dt_min: float = 30.0
    n_members: int = 24
    dose_dep_weight: float = 0.35
    source_jitter_m: float = 20.0
    random_walk_mps: float = 0.25
    min_std: float = 0.05

@dataclass
class TimingConfig:
    travel_speed_kmph: float = 15.0
    station_time_min: float = 1.0
    

# ---------------------------------------------------------------------
# Grids / measurement model
# ---------------------------------------------------------------------


def make_mesh(cfg: FieldConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax = cfg.bounds
    xs = np.linspace(xmin, xmax, cfg.grid_n)
    ys = np.linspace(ymin, ymax, cfg.grid_n)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    return xs, ys, X, Y

def build_interpolator(cfg: FieldConfig, Z: np.ndarray) -> RegularGridInterpolator:
    xs = np.linspace(cfg.bounds[0], cfg.bounds[1], cfg.grid_n)
    ys = np.linspace(cfg.bounds[2], cfg.bounds[3], cfg.grid_n)
    return RegularGridInterpolator((ys, xs), np.nan_to_num(Z, nan=cfg.background_level), bounds_error=False, fill_value=cfg.background_level)

def build_disk_kernel(cfg: FieldConfig, radius_m: float) -> np.ndarray:
    dx = (cfg.bounds[1] - cfg.bounds[0]) / max(cfg.grid_n - 1, 1)
    dy = (cfg.bounds[3] - cfg.bounds[2]) / max(cfg.grid_n - 1, 1)
    rx = int(np.ceil(radius_m / max(dx, 1e-9)))
    ry = int(np.ceil(radius_m / max(dy, 1e-9)))
    xx = (np.arange(-rx, rx + 1) * dx)[None, :]
    yy = (np.arange(-ry, ry + 1) * dy)[:, None]
    k = ((xx ** 2 + yy ** 2) <= radius_m ** 2).astype(float)
    k /= np.sum(k)
    return k

def build_measurement_field(cfg: FieldConfig, Z_true: np.ndarray) -> np.ndarray:
    if cfg.measurement_model == "point":
        return Z_true.copy()
    r = 0.5 * cfg.measurement_diameter_m
    kernel = build_disk_kernel(cfg, r)
    zfill = np.nan_to_num(Z_true, nan=cfg.background_level)
    zavg = fftconvolve(zfill, kernel, mode="same")
    zavg[np.isnan(Z_true)] = np.nan
    return zavg

def sample_field(cfg: FieldConfig, interp: RegularGridInterpolator, xy: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    vals = interp(np.column_stack([xy[:, 1], xy[:, 0]])).astype(float)
    vals += rng.normal(0.0, cfg.meas_sigma, size=len(vals))
    return vals


# ---------------------------------------------------------------------
# Point generation
# ---------------------------------------------------------------------


def deduplicate_points(points: np.ndarray, min_spacing: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return pts.reshape(0, 2)
    keep: List[np.ndarray] = []
    for p in pts:
        if not keep:
            keep.append(p)
            continue
        d = np.linalg.norm(np.vstack(keep) - p[None, :], axis=1)
        if float(np.min(d)) >= float(min_spacing):
            keep.append(p)
    return np.asarray(keep, dtype=float)

def hex_grid_in_geom(geom: Polygon, spacing_x: float, spacing_y: Optional[float] = None) -> np.ndarray:
    if geom.is_empty:
        return np.zeros((0, 2), dtype=float)
    if spacing_y is None:
        spacing_y = 0.8660254 * spacing_x
    xmin, ymin, xmax, ymax = geom.bounds
    pg = prep(geom)
    pts: List[Tuple[float, float]] = []
    y = ymin
    row = 0
    while y <= ymax + 1e-9:
        x = xmin + (0.5 * spacing_x if row % 2 else 0.0)
        while x <= xmax + 1e-9:
            if pg.covers(Point(float(x), float(y))):
                pts.append((float(x), float(y)))
            x += spacing_x
        y += spacing_y
        row += 1
    return np.asarray(pts, dtype=float)

def make_candidate_points_in_geom(bounds: Tuple[float, float, float, float], spacing: float, geom: Polygon) -> np.ndarray:
    xmin, xmax, ymin, ymax = bounds
    nx = max(5, int(np.ceil((xmax - xmin) / max(spacing, 1e-9))) + 1)
    ny = max(5, int(np.ceil((ymax - ymin) / max(spacing, 1e-9))) + 1)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel()])
    pg = prep(geom)
    keep = np.fromiter((pg.covers(Point(float(x), float(y))) for x, y in pts), dtype=bool, count=len(pts))
    return pts[keep]

def farthest_point_init(points: np.ndarray, k: int, seed_points: Optional[np.ndarray] = None) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if k >= len(pts):
        return pts.copy()
    if seed_points is None or len(seed_points) == 0:
        center = np.mean(pts, axis=0)
        selected = [int(np.argmax(np.linalg.norm(pts - center[None, :], axis=1)))]
    else:
        seed_points = np.asarray(seed_points, dtype=float)
        selected = []
        for sp in seed_points:
            j = int(np.argmin(np.linalg.norm(pts - sp[None, :], axis=1)))
            if j not in selected:
                selected.append(j)
        if not selected:
            center = np.mean(pts, axis=0)
            selected = [int(np.argmax(np.linalg.norm(pts - center[None, :], axis=1)))]
    while len(selected) < int(k):
        remaining = [i for i in range(len(pts)) if i not in selected]
        dmin = []
        for i in remaining:
            dmin.append(min(np.linalg.norm(pts[i] - pts[j]) for j in selected))
        selected.append(remaining[int(np.argmax(dmin))])
    return pts[np.asarray(selected, dtype=int)]

def nearest_neighbor_route(points: np.ndarray, start_xy: Optional[np.ndarray] = None) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 1:
        return pts.reshape(-1, 2).copy()
    remaining = list(range(len(pts)))
    if start_xy is None:
        start_idx = int(np.argmin(pts[:, 0]))
    else:
        start_xy = np.asarray(start_xy, dtype=float).reshape(1, 2)
        start_idx = int(np.argmin(np.linalg.norm(pts - start_xy, axis=1)))
    order = [start_idx]
    remaining.remove(start_idx)
    while remaining:
        last = pts[order[-1]]
        cand = pts[np.asarray(remaining, dtype=int)]
        j_local = int(np.argmin(np.linalg.norm(cand - last[None, :], axis=1)))
        order.append(remaining.pop(j_local))
    return pts[np.asarray(order, dtype=int)]

def path_length(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 1:
        return 0.0
    diffs = np.diff(pts, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))

def _cluster_lines(points: np.ndarray, line_tol_m: float, axis: str = "row") -> List[np.ndarray]:
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return []
    axis = str(axis).strip().lower()
    if axis not in {"row", "col"}:
        raise ValueError(f"Unsupported serpentine axis: {axis}")
    major_idx = 1 if axis == "row" else 0
    idx = np.argsort(pts[:, major_idx])
    lines: List[List[int]] = []
    cur: List[int] = [int(idx[0])]
    cur_mean = float(pts[idx[0], major_idx])
    for ii in idx[1:]:
        ii = int(ii)
        if abs(float(pts[ii, major_idx]) - cur_mean) <= float(max(line_tol_m, 1e-6)):
            cur.append(ii)
            cur_mean = float(np.mean(pts[np.asarray(cur, dtype=int), major_idx]))
        else:
            lines.append(cur)
            cur = [ii]
            cur_mean = float(pts[ii, major_idx])
    lines.append(cur)
    return [np.asarray(line, dtype=int) for line in lines]

def serpentine_order_indices(points: np.ndarray, row_tol_m: float = 8.0,
                             end_corner: str = "upper_right", axis: str = "row") -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 1:
        return np.arange(len(pts), dtype=int)

    axis = str(axis).strip().lower()
    lines = _cluster_lines(pts, row_tol_m, axis=axis)
    n_lines = len(lines)

    if axis == "row":
        if end_corner == "upper_right":
            first_forward = (n_lines % 2 == 1)
        elif end_corner == "upper_left":
            first_forward = (n_lines % 2 == 0)
        else:
            first_forward = True
        order: List[int] = []
        for line_i, line in enumerate(lines):
            line_pts = pts[line]
            if (line_i % 2 == 0 and first_forward) or (line_i % 2 == 1 and not first_forward):
                local = np.lexsort((line_pts[:, 1], line_pts[:, 0]))
            else:
                local = np.lexsort((line_pts[:, 1], -line_pts[:, 0]))
            order.extend(line[local].tolist())
        return np.asarray(order, dtype=int)

    if end_corner == "upper_right":
        first_forward = (n_lines % 2 == 1)   
    elif end_corner == "lower_right":
        first_forward = (n_lines % 2 == 0)   
    else:
        first_forward = True
    order = []
    for line_i, line in enumerate(lines):
        line_pts = pts[line]
        if (line_i % 2 == 0 and first_forward) or (line_i % 2 == 1 and not first_forward):
            local = np.lexsort((line_pts[:, 0], line_pts[:, 1]))
        else:
            local = np.lexsort((line_pts[:, 0], -line_pts[:, 1]))
        order.extend(line[local].tolist())
    return np.asarray(order, dtype=int)

def filter_points_in_geom(points: np.ndarray, geom: BaseGeometry) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return pts.reshape(0, 2)
    pg = prep(to_geometry(geom))
    keep = np.fromiter((pg.covers(Point(float(x), float(y))) for x, y in pts), dtype=bool, count=len(pts))
    return pts[keep]

def circular_order_indices(
    points: np.ndarray,
    end_corner: str = "upper_right",
    clockwise: bool = False,
    center: Optional[np.ndarray] = None,
) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 1:
        return np.arange(len(pts), dtype=int)

    if center is None:
        center = np.mean(pts, axis=0)
    center = np.asarray(center, dtype=float)

    ang = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order0 = np.argsort(-ang if clockwise else ang)

    corner_angles = {
        "upper_right": np.pi / 4.0,
        "upper_left": 3.0 * np.pi / 4.0,
        "lower_left": -3.0 * np.pi / 4.0,
        "lower_right": -np.pi / 4.0,
    }
    target = corner_angles.get(str(end_corner).lower(), np.pi / 4.0)

    ang0 = ang[order0]
    dtheta = np.angle(np.exp(1j * (ang0 - target)))
    pos = int(np.argmin(np.abs(dtheta)))
    return np.roll(order0, -pos)

def cluster_circular_rings(points: np.ndarray, radius_tol_m: float, center: Optional[np.ndarray] = None) -> List[np.ndarray]:
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return []
    if center is None:
        center = np.mean(pts, axis=0)
    center = np.asarray(center, dtype=float)

    r = np.linalg.norm(pts - center[None, :], axis=1)
    idx = np.argsort(-r)  

    rings: List[List[int]] = []
    cur: List[int] = [int(idx[0])]
    cur_mean = float(r[idx[0]])

    for ii in idx[1:]:
        ii = int(ii)
        if abs(float(r[ii]) - cur_mean) <= float(max(radius_tol_m, 1e-6)):
            cur.append(ii)
            cur_mean = float(np.mean(r[np.asarray(cur, dtype=int)]))
        else:
            rings.append(cur)
            cur = [ii]
            cur_mean = float(r[ii])
    rings.append(cur)
    return [np.asarray(ring, dtype=int) for ring in rings]

def concentric_circular_order_indices(
    points: np.ndarray,
    radius_tol_m: float,
    end_corner: str = "upper_right",
    clockwise: bool = False,
) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 1:
        return np.arange(len(pts), dtype=int)

    center = np.mean(pts, axis=0)
    rings = cluster_circular_rings(pts, radius_tol_m=radius_tol_m, center=center)

    order: List[int] = []
    for ring in rings:
        ring_pts = pts[ring]
        local = circular_order_indices(
            ring_pts,
            end_corner=end_corner,
            clockwise=clockwise,
            center=center,
        )
        order.extend(ring[local].tolist())
    return np.asarray(order, dtype=int)

def row_serpentine_top_first_indices(
    points: np.ndarray,
    row_tol_m: float = 8.0,
    start_corner: str = "upper_right",
) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 1:
        return np.arange(len(pts), dtype=int)

    lines = _cluster_lines(pts, row_tol_m, axis="row")
    lines = list(reversed(lines))  

    start_right = "right" in str(start_corner).strip().lower()
    order: List[int] = []
    for line_i, line in enumerate(lines):
        line_pts = pts[line]
        go_left_to_right = (not start_right) if (line_i % 2 == 0) else start_right
        if go_left_to_right:
            local = np.lexsort((line_pts[:, 1], line_pts[:, 0]))
        else:
            local = np.lexsort((line_pts[:, 1], -line_pts[:, 0]))
        order.extend(line[local].tolist())
    return np.asarray(order, dtype=int)

def order_baseline_csv_points(
    data: object,
    *,
    line_tol_m: float,
    end_corner: str,
    axis: str,
) -> np.ndarray:
    if data is None:
        return np.zeros((0, 2), dtype=float)

    if isinstance(data, pd.DataFrame):
        df = data.copy().reset_index(drop=True)
        pts = baseline_points_from_frame(df)
    else:
        df = None
        pts = np.asarray(data, dtype=float)

    if len(pts) <= 1:
        return pts.reshape(-1, 2).copy()

    axis_norm = str(axis).lower() if str(axis).lower() in {"row", "col"} else "col"
    if df is None:
        ord_all = serpentine_order_indices(
            pts,
            row_tol_m=float(max(line_tol_m, 1e-6)),
            end_corner=str(end_corner),
            axis=axis_norm,
        )
        return pts[ord_all]

    type_col = 'type' if 'type' in df.columns else None
    area_col = 'area' if 'area' in df.columns else None
    type_series = df[type_col].fillna('').astype(str).str.strip() if type_col else pd.Series([''] * len(df))
    area_series = df[area_col].fillna('').astype(str).str.strip() if area_col else pd.Series([''] * len(df))
    type_norm = type_series.str.lower()
    area_norm = area_series.str.lower()

    used = np.zeros(len(df), dtype=bool)
    ordered_idx: List[int] = []

    def _append_mask(mask: np.ndarray, mode: str = 'default') -> None:
        nonlocal used, ordered_idx
        mask = np.asarray(mask, dtype=bool) & (~used)
        if not np.any(mask):
            return
        idx_area = np.flatnonzero(mask)
        pts_area = pts[idx_area]
        if len(pts_area) <= 1:
            local = np.arange(len(pts_area), dtype=int)
        elif mode == 'circular':
            local = concentric_circular_order_indices(
                pts_area,
                radius_tol_m=float(max(line_tol_m, 1e-6)),              
                end_corner=str(end_corner),
                clockwise=False,
            )
        elif mode == 'near_top_rows':
            local = row_serpentine_top_first_indices(
                pts_area,
                row_tol_m=float(max(line_tol_m, 1e-6)),
                start_corner=str(end_corner),
            )
        else:
            local = serpentine_order_indices(
                pts_area,
                row_tol_m=float(max(line_tol_m, 1e-6)),
                end_corner=str(end_corner),
                axis=axis_norm,
            )
        ordered_idx.extend(idx_area[local].tolist())
        used[idx_area] = True

    perimeter_mask = type_norm.eq('perimeter').to_numpy()
    detonation_mask = area_norm.eq('detonation site').to_numpy()
    detonation_perimeter_mask = detonation_mask & perimeter_mask
    if np.any(detonation_perimeter_mask):
        _append_mask(detonation_perimeter_mask, mode='circular')
    elif np.any(perimeter_mask):
        _append_mask(perimeter_mask, mode='circular')

    phase2_mask = area_norm.eq('phase 2').to_numpy()
    _append_mask(phase2_mask, mode='default')

    near_mask = area_norm.eq('near').to_numpy()
    _append_mask(near_mask, mode='near_top_rows')

    remaining_area = area_series[~used]
    area_order = pd.unique(remaining_area) if len(remaining_area) > 0 else []
    for area_name in area_order:
        area_mask = (~used) & (area_series == area_name).to_numpy()
        if not np.any(area_mask):
            continue
        _append_mask(area_mask, mode='default')

    if not np.all(used):
        idx_rest = np.flatnonzero(~used)
        pts_rest = pts[idx_rest]
        local = serpentine_order_indices(
            pts_rest,
            row_tol_m=float(max(line_tol_m, 1e-6)),
            end_corner=str(end_corner),
            axis=axis_norm,
        ) if len(pts_rest) > 1 else np.arange(len(pts_rest), dtype=int)
        ordered_idx.extend(idx_rest[local].tolist())
    
    return pts[np.asarray(ordered_idx, dtype=int)]    

def _baseline_area_norm_series(df: pd.DataFrame) -> pd.Series:
    if df is None or len(df) == 0 or 'area' not in df.columns:
        return pd.Series([''] * (0 if df is None else len(df)), index=([] if df is None else df.index), dtype=object)
    return df['area'].fillna('').astype(str).str.strip().str.lower()

def baseline_csv_supports_phase_guided_search(df: Optional[pd.DataFrame]) -> bool:
    if df is None or len(df) == 0 or 'area' not in df.columns:
        return False
    area_tokens = set(_baseline_area_norm_series(df).tolist())
    return len(area_tokens & {'detonation site', 'phase 2', 'near'}) > 0

def ensure_baseline_xy_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    if '__x__' not in out.columns or '__y__' not in out.columns:
        if 'x coord' in out.columns and 'y coord' in out.columns:
            out['__x__'] = pd.to_numeric(out['x coord'], errors='coerce')
            out['__y__'] = pd.to_numeric(out['y coord'], errors='coerce')
        elif 'x' in out.columns and 'y' in out.columns:
            out['__x__'] = pd.to_numeric(out['x'], errors='coerce')
            out['__y__'] = pd.to_numeric(out['y'], errors='coerce')
        else:
            raise ValueError("Baseline dataframe must include x/y coordinates or __x__/__y__ columns.")
    out = out.dropna(subset=['__x__', '__y__']).reset_index(drop=True)
    return out

def estimate_point_spacing(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return 1.0
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, np.nan)
    nn = np.nanmin(d, axis=1)
    finite = nn[np.isfinite(nn) & (nn > 0)]
    return float(np.nanmedian(finite)) if finite.size > 0 else 1.0

def estimate_rowwise_x_step(df_phase: pd.DataFrame) -> float:
    phase = ensure_baseline_xy_frame(df_phase)
    if len(phase) < 2:
        return 1.0
    diffs: List[float] = []
    for _, sub in phase.groupby('__y__', sort=False):
        xs = np.sort(pd.to_numeric(sub['__x__'], errors='coerce').dropna().to_numpy(dtype=float))
        if len(xs) < 2:
            continue
        d = np.diff(xs)
        d = d[np.isfinite(d) & (d > 1e-6)]
        if d.size > 0:
            diffs.append(float(np.nanmedian(d)))
    if len(diffs) == 0:
        return float(max(estimate_point_spacing(phase[['__x__', '__y__']].to_numpy(dtype=float)), 1.0))
    return float(max(np.nanmedian(np.asarray(diffs, dtype=float)), 1.0))

def baseline_x_group_key(x_value: float, x_step: Optional[float] = None, decimals: int = 6) -> float:
    if x_step is not None and np.isfinite(float(x_step)) and float(x_step) > 1e-6:
        step = float(x_step)
        return float(np.round(float(x_value) / step) * step)
    return float(np.round(float(x_value), int(decimals)))

def polar_angles_deg(points: np.ndarray, origin: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    ang = np.degrees(np.arctan2(pts[:, 1] - float(origin[1]), pts[:, 0] - float(origin[0])))
    return np.mod(ang, 360.0)

def infer_phase_start_angle_deg(df_phase: pd.DataFrame, origin: Tuple[float, float] = (0.0, 0.0)) -> float:
    phase = ensure_baseline_xy_frame(df_phase)
    if len(phase) == 0:
        return 0.0
    p0 = phase[['__x__', '__y__']].to_numpy(dtype=float)[0]
    return float(polar_angles_deg(p0.reshape(1, 2), origin=origin)[0])

def sort_points_ccw_from_start(
    df_phase: pd.DataFrame,
    start_angle_deg: float,
    origin: Tuple[float, float] = (0.0, 0.0),
) -> pd.DataFrame:
    phase = ensure_baseline_xy_frame(df_phase)
    if len(phase) <= 1:
        return phase
    pts = phase[['__x__', '__y__']].to_numpy(dtype=float)
    phase = phase.copy()
    phase['__angle_ccw__'] = polar_angles_deg(pts, origin=origin)
    phase['__angle_rel__'] = np.mod(phase['__angle_ccw__'] - float(start_angle_deg), 360.0)
    phase = phase.sort_values(['__angle_rel__', '__angle_ccw__'], kind='mergesort').reset_index(drop=True)
    return phase.drop(columns=['__angle_ccw__', '__angle_rel__'], errors='ignore')

def order_phase1_detonation_points(
    df_phase: pd.DataFrame,
    origin: Tuple[float, float] = (0.0, 0.0),
) -> pd.DataFrame:
    phase = ensure_baseline_xy_frame(df_phase)
    if len(phase) <= 1:
        return phase
    pts = phase[['__x__', '__y__']].to_numpy(dtype=float)
    phase = phase.copy()
    phase['__radius__'] = np.linalg.norm(pts - np.asarray(origin, dtype=float)[None, :], axis=1)
    start_angle_deg = infer_phase_start_angle_deg(phase, origin=origin)

    ring_col = '__phase_ring__'
    if 'detonation_ring_order' in phase.columns and phase['detonation_ring_order'].notna().any():
        phase[ring_col] = pd.to_numeric(phase['detonation_ring_order'], errors='coerce').fillna(-1.0)
    else:
        tol = max(0.50 * estimate_point_spacing(pts), 1e-6)
        rings = cluster_circular_rings(pts, radius_tol_m=tol, center=np.asarray(origin, dtype=float))
        ring_ids = np.full(len(phase), -1, dtype=int)
        for ring_i, ring in enumerate(rings):
            ring_ids[np.asarray(ring, dtype=int)] = int(ring_i)
        phase[ring_col] = ring_ids

    ring_order = (
        phase.groupby(ring_col)['__radius__']
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    ordered_parts = []
    for ring_id in ring_order:
        sub = phase.loc[phase[ring_col] == ring_id].copy()
        sub = sort_points_ccw_from_start(sub, start_angle_deg=start_angle_deg, origin=origin)
        ordered_parts.append(sub)

    out = pd.concat(ordered_parts, ignore_index=True) if len(ordered_parts) > 0 else phase.iloc[0:0].copy()
    return out.drop(columns=['__radius__', ring_col], errors='ignore')

def order_phase2_radial_points(
    df_phase: pd.DataFrame,
    origin: Tuple[float, float] = (0.0, 0.0),
) -> pd.DataFrame:
    phase = ensure_baseline_xy_frame(df_phase)
    if len(phase) <= 1:
        return phase
    pts = phase[['__x__', '__y__']].to_numpy(dtype=float)
    start_angle_deg = infer_phase_start_angle_deg(phase, origin=origin)
    tol = max(0.50 * estimate_point_spacing(pts), 1e-6)
    rings = cluster_circular_rings(pts, radius_tol_m=tol, center=np.asarray(origin, dtype=float))

    ordered_parts = []
    for ring in reversed(rings):  
        sub = phase.iloc[np.asarray(ring, dtype=int)].copy()
        sub = sort_points_ccw_from_start(sub, start_angle_deg=start_angle_deg, origin=origin)
        ordered_parts.append(sub)

    return pd.concat(ordered_parts, ignore_index=True) if len(ordered_parts) > 0 else phase.iloc[0:0].copy()

def order_phase3_near_points(df_phase: pd.DataFrame) -> pd.DataFrame:
    phase = ensure_baseline_xy_frame(df_phase)
    return phase.reset_index(drop=True)

def classify_detection_level(value: float, thresholds: Sequence[float]) -> str:
    thr = np.sort(np.asarray(list(thresholds), dtype=float))
    if len(thr) == 0:
        return 'cold'
    if float(value) >= float(thr[-1]):
        return 'hot'
    if float(value) >= float(thr[0]):
        return 'warm'
    return 'cold'

def make_origin_segment_corridor(
    endpoint_xy: np.ndarray,
    half_width_m: float = 11.0,
    origin: Tuple[float, float] = (0.0, 0.0),
) -> BaseGeometry:
    origin_xy = np.asarray(origin, dtype=float)
    endpoint_xy = np.asarray(endpoint_xy, dtype=float)
    if np.linalg.norm(endpoint_xy - origin_xy) <= 1e-9:
        return Point(float(origin_xy[0]), float(origin_xy[1])).buffer(float(half_width_m))
    line = LineString([
        (float(origin_xy[0]), float(origin_xy[1])),
        (float(endpoint_xy[0]), float(endpoint_xy[1])),
    ])
    return line.buffer(float(half_width_m), cap_style=2, join_style=2)

def extend_point_to_radius(
    point_xy: np.ndarray,
    target_radius: float,
    origin: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    p = np.asarray(point_xy, dtype=float)
    o = np.asarray(origin, dtype=float)
    v = p - o
    n = float(np.linalg.norm(v))
    if n <= 1e-9:
        return p.copy()
    return o + (float(target_radius) / n) * v

def point_is_in_any_corridor(point_xy: np.ndarray, corridors: Sequence[BaseGeometry]) -> bool:
    if corridors is None or len(corridors) == 0:
        return False
    pt = Point(float(point_xy[0]), float(point_xy[1]))
    return any(corridor.covers(pt) for corridor in corridors if corridor is not None and not corridor.is_empty)

def prune_later_phase_against_kept(
    df_keep: pd.DataFrame,
    df_drop: pd.DataFrame,
    radius_m: float,
    drop_area_label: str,
    keep_label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(df_keep) == 0 or len(df_drop) == 0:
        empty = df_drop.iloc[0:0].copy()
        return df_drop.copy(), empty

    keep_xy = df_keep[['__x__', '__y__']].to_numpy(dtype=float)
    drop_xy = df_drop[['__x__', '__y__']].to_numpy(dtype=float)
    d = np.linalg.norm(drop_xy[:, None, :] - keep_xy[None, :, :], axis=2)
    nn_idx = np.argmin(d, axis=1)
    nn_dist = d[np.arange(len(drop_xy)), nn_idx]
    remove_mask = nn_dist <= float(radius_m) + 1e-9

    kept = df_drop.loc[~remove_mask].copy()
    removed = df_drop.loc[remove_mask].copy()
    if len(removed) > 0:
        removed['nearest_keep_x'] = keep_xy[nn_idx[remove_mask], 0]
        removed['nearest_keep_y'] = keep_xy[nn_idx[remove_mask], 1]
        removed['nearest_dist_m'] = nn_dist[remove_mask]
        removed['removed_from_area'] = str(drop_area_label)
        removed['kept_reference'] = str(keep_label)
        removed['removal_rule'] = f"{drop_area_label}_within_{radius_m:g}m_of_{keep_label}".replace(" ", "_")
    return kept, removed

def prune_baseline_all_phases(
    baseline_df: pd.DataFrame,
    radius_m: float = 22.0,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, float]]:
    df = ensure_baseline_xy_frame(baseline_df)
    if 'area' not in df.columns:
        return df, {'phase2_removed': df.iloc[0:0].copy(), 'near_removed': df.iloc[0:0].copy(), 'all_removed': df.iloc[0:0].copy()}, {}

    area_norm = df['area'].astype(str).str.strip().str.lower()
    det = df.loc[area_norm.eq('detonation site')].copy()
    ph2 = df.loc[area_norm.eq('phase 2')].copy()
    near = df.loc[area_norm.eq('near')].copy()
    other = df.loc[~area_norm.isin(['detonation site', 'phase 2', 'near'])].copy()

    ph2_kept, ph2_removed = prune_later_phase_against_kept(
        df_keep=det,
        df_drop=ph2,
        radius_m=radius_m,
        drop_area_label='phase 2',
        keep_label='detonation site',
    )

    earlier_keep = pd.concat([det, ph2_kept], ignore_index=False)
    near_kept, near_removed = prune_later_phase_against_kept(
        df_keep=earlier_keep,
        df_drop=near,
        radius_m=radius_m,
        drop_area_label='near',
        keep_label='detonation site_or_phase 2',
    )

    keep_indices = list(det.index) + list(ph2_kept.index) + list(near_kept.index) + list(other.index)
    keep_indices = sorted(set(int(i) for i in keep_indices))
    clean_df = df.loc[keep_indices].copy().reset_index(drop=True)

    if 'route_order_old' not in clean_df.columns and 'route_order' in clean_df.columns:
        clean_df['route_order_old'] = clean_df['route_order']
    if 'route_order' in clean_df.columns:
        clean_df['route_order'] = np.arange(1, len(clean_df) + 1, dtype=int)

    removed_all = pd.concat([ph2_removed, near_removed], ignore_index=True, sort=False)

    stats = {
        'original_total': float(len(df)),
        'original_detonation': float(len(det)),
        'original_phase2': float(len(ph2)),
        'original_near': float(len(near)),
        'removed_phase2': float(len(ph2_removed)),
        'removed_near': float(len(near_removed)),
        'clean_total': float(len(clean_df)),
    }

    clean_area = clean_df['area'].astype(str).str.strip().str.lower()
    det_xy = clean_df.loc[clean_area.eq('detonation site'), ['__x__', '__y__']].to_numpy(dtype=float)
    ph2_xy = clean_df.loc[clean_area.eq('phase 2'), ['__x__', '__y__']].to_numpy(dtype=float)
    near_xy = clean_df.loc[clean_area.eq('near'), ['__x__', '__y__']].to_numpy(dtype=float)

    if len(det_xy) > 0 and len(ph2_xy) > 0:
        d = np.linalg.norm(ph2_xy[:, None, :] - det_xy[None, :, :], axis=2)
        dmin = np.min(d, axis=1)
        stats['remaining_phase2_to_det_leq_radius'] = float(np.sum(dmin <= float(radius_m) + 1e-9))
        stats['remaining_phase2_to_det_min_dist_m'] = float(np.min(dmin))
    else:
        stats['remaining_phase2_to_det_leq_radius'] = 0.0
        stats['remaining_phase2_to_det_min_dist_m'] = float('nan')

    if len(near_xy) > 0 and (len(det_xy) + len(ph2_xy)) > 0:
        prev_xy = np.vstack([a for a in [det_xy, ph2_xy] if len(a) > 0])
        d = np.linalg.norm(near_xy[:, None, :] - prev_xy[None, :, :], axis=2)
        dmin = np.min(d, axis=1)
        stats['remaining_near_to_prev_leq_radius'] = float(np.sum(dmin <= float(radius_m) + 1e-9))
        stats['remaining_near_to_prev_min_dist_m'] = float(np.min(dmin))
    else:
        stats['remaining_near_to_prev_leq_radius'] = 0.0
        stats['remaining_near_to_prev_min_dist_m'] = float('nan')

    return clean_df, {
        'phase2_removed': ph2_removed.reset_index(drop=True),
        'near_removed': near_removed.reset_index(drop=True),
        'all_removed': removed_all.reset_index(drop=True),
    }, stats

def simulate_phase_guided_baseline_sampling(
    frame: pd.DataFrame,
    cfg: FieldConfig,
    Z_true: np.ndarray,
    rng: np.random.Generator,
    ops: OperationalCriteria,
    *,
    max_samples: int = 0,
    corridor_half_width_m: float = 11.0,
    use_observed_value_for_control: bool = False,
    origin: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    phase_frame = ensure_baseline_xy_frame(frame)
    area_norm = _baseline_area_norm_series(phase_frame)

    phase1_df = order_phase1_detonation_points(
        phase_frame.loc[area_norm == 'detonation site'].copy(),
        origin=origin,
    )
    phase2_df = order_phase2_radial_points(
        phase_frame.loc[area_norm == 'phase 2'].copy(),
        origin=origin,
    )
    phase3_df = order_phase3_near_points(
        phase_frame.loc[area_norm == 'near'].copy()
    )

    phase_specs = [
        ('phase1', phase1_df, 'hot'),
        ('phase2', phase2_df, 'cold'),
        ('phase3', phase3_df, None),
    ]

    phase2_max_radius = 0.0
    if len(phase2_df) > 0:
        pts_phase2 = phase2_df[['__x__', '__y__']].to_numpy(dtype=float)
        phase2_max_radius = float(np.max(np.linalg.norm(pts_phase2 - np.asarray(origin, dtype=float)[None, :], axis=1)))

    phase3_x_step = estimate_rowwise_x_step(phase3_df) if len(phase3_df) > 1 else None

    extra_mask = ~area_norm.isin(['detonation site', 'phase 2', 'near'])
    extra_df = phase_frame.loc[extra_mask].copy().reset_index(drop=True)
    if len(extra_df) > 0:
        phase_specs.append(('phase_extra', extra_df, None))

    Z_meas = build_measurement_field(cfg, Z_true)
    control_interp = build_interpolator(cfg, Z_meas)

    visited_xy: List[np.ndarray] = []
    observed_values: List[float] = []
    control_values: List[float] = []
    decision_values: List[float] = []
    point_phases: List[str] = []
    point_classes: List[str] = []
    point_markers: List[str] = []
    point_areas: List[str] = []
    phase1_hot_endpoints: List[np.ndarray] = []
    phase2_cold_endpoints: List[np.ndarray] = []
    phase3_exit_points: List[np.ndarray] = []
    phase3_skip_x_values: List[float] = []
    phase_skip_counts: Dict[str, int] = {}

    stop_all = False
    phase3_seen_ge1_by_x: Dict[float, bool] = {}
    phase3_skip_x_keys = set()

    for phase_name, phase_df, skip_trigger in phase_specs:
        phase_df = ensure_baseline_xy_frame(phase_df)
        corridors: List[BaseGeometry] = []
        skipped_here = 0

        for _, row in phase_df.iterrows():
            p = np.array([float(row['__x__']), float(row['__y__'])], dtype=float)
            phase3_x_key = None

            if phase_name == 'phase3':
                phase3_x_key = baseline_x_group_key(p[0], x_step=phase3_x_step)
                if phase3_x_key in phase3_skip_x_keys:
                    skipped_here += 1
                    continue

            if point_is_in_any_corridor(p, corridors):
                skipped_here += 1
                continue

            control_val = float(control_interp(np.array([[p[1], p[0]]], dtype=float))[0])
            observed_val = float(sample_field(cfg, control_interp, p[None, :], rng)[0])
            decision_val = observed_val if bool(use_observed_value_for_control) else control_val
            det_class = classify_detection_level(decision_val, ops.thresholds)

            visited_xy.append(p)
            observed_values.append(observed_val)
            control_values.append(control_val)
            decision_values.append(decision_val)
            point_phases.append(phase_name)
            point_classes.append(det_class)
            point_markers.append({'cold': 'o', 'warm': '^', 'hot': 'x'}[det_class])
            point_areas.append(str(row['area']) if 'area' in row else phase_name)

            if phase_name == 'phase1' and det_class == 'hot':
                corridors.append(make_origin_segment_corridor(p, half_width_m=float(corridor_half_width_m), origin=origin))
                phase1_hot_endpoints.append(p.copy())
            elif phase_name == 'phase2' and det_class == 'cold':
                p_outer = extend_point_to_radius(
                    p,
                    target_radius=max(float(phase2_max_radius), float(np.linalg.norm(p - np.asarray(origin, dtype=float)))),
                    origin=origin,
                )
                corridors.append(make_origin_segment_corridor(p_outer, half_width_m=float(corridor_half_width_m), origin=origin))
                phase2_cold_endpoints.append(p.copy())
            elif phase_name == 'phase3' and phase3_x_key is not None:
                if det_class in {'warm', 'hot'}:
                    phase3_seen_ge1_by_x[phase3_x_key] = True
                elif det_class == 'cold' and bool(phase3_seen_ge1_by_x.get(phase3_x_key, False)):
                    if phase3_x_key not in phase3_skip_x_keys:
                        phase3_skip_x_keys.add(phase3_x_key)
                        phase3_skip_x_values.append(float(p[0]))
                        phase3_exit_points.append(p.copy())

            if int(max_samples) > 0 and len(visited_xy) >= int(max_samples):
                stop_all = True
                break

        phase_skip_counts[phase_name] = int(skipped_here)
        if stop_all:
            break

    xy = np.asarray(visited_xy, dtype=float).reshape(-1, 2) if len(visited_xy) > 0 else np.zeros((0, 2), dtype=float)
    y = np.asarray(observed_values, dtype=float)
    diag = {
        'phase_guided': True,
        'corridor_half_width_m': float(corridor_half_width_m),
        'use_observed_value_for_control': bool(use_observed_value_for_control),
        'point_phases': point_phases,
        'point_classes': point_classes,
        'point_markers': point_markers,
        'point_areas': point_areas,
        'point_control_values': np.asarray(control_values, dtype=float),
        'point_decision_values': np.asarray(decision_values, dtype=float),
        'phase1_hot_endpoints': np.asarray(phase1_hot_endpoints, dtype=float).reshape(-1, 2) if len(phase1_hot_endpoints) > 0 else np.zeros((0, 2), dtype=float),
        'phase2_cold_endpoints': np.asarray(phase2_cold_endpoints, dtype=float).reshape(-1, 2) if len(phase2_cold_endpoints) > 0 else np.zeros((0, 2), dtype=float),
        'phase3_exit_points': np.asarray(phase3_exit_points, dtype=float).reshape(-1, 2) if len(phase3_exit_points) > 0 else np.zeros((0, 2), dtype=float),
        'phase3_x_group_step_m': float(phase3_x_step) if phase3_x_step is not None else np.nan,
        'phase3_skip_x_values': np.asarray(phase3_skip_x_values, dtype=float),
        'phase3_seen_ge1_x_values': np.asarray(sorted([x for x, seen in phase3_seen_ge1_by_x.items() if seen]), dtype=float),
        'phase_skip_counts': phase_skip_counts,
    }
    return xy, y, diag

def export_baseline_phaseguided_artifacts(
    out_dir: str,
    baseline_res: RunResult,
    baseline_df: Optional[pd.DataFrame] = None,
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    diag = baseline_res.diag if isinstance(baseline_res.diag, dict) else {}
    if not diag.get('phase_guided', False):
        return out

    os.makedirs(out_dir, exist_ok=True)
    n = len(baseline_res.X_obs)
    def _arr(name, default):
        val = diag.get(name, default)
        arr = np.asarray(val)
        if arr.ndim == 0:
            arr = np.repeat(arr, n)
        return arr

    trace_df = pd.DataFrame({
        'x': baseline_res.X_obs[:, 0],
        'y': baseline_res.X_obs[:, 1],
        'y_obs': baseline_res.y_obs,
        'field_control_value': _arr('point_control_values', np.full(n, np.nan)),
        'decision_value': _arr('point_decision_values', np.full(n, np.nan)),
        'phase': _arr('point_phases', np.array([''] * n, dtype=object)),
        'det_class': _arr('point_classes', np.array([''] * n, dtype=object)),
        'marker': _arr('point_markers', np.array([''] * n, dtype=object)),
        'area': _arr('point_areas', np.array([''] * n, dtype=object)),
    })
    trace_path = os.path.join(out_dir, 'baseline_phase_guided_trace.csv')
    trace_df.to_csv(trace_path, index=False)
    out['baseline_phase_guided_trace_csv'] = trace_path

    exit_pts = np.asarray(diag.get('phase3_exit_points', np.zeros((0, 2), dtype=float)), dtype=float).reshape(-1, 2)
    skip_x = np.asarray(diag.get('phase3_skip_x_values', np.zeros((0,), dtype=float)), dtype=float).reshape(-1)
    if len(exit_pts) > 0 or len(skip_x) > 0:
        k = min(len(skip_x), len(exit_pts)) if len(exit_pts) > 0 else len(skip_x)
        if k > 0:
            skip_df = pd.DataFrame({
                'skip_x_value': skip_x[:k],
                'exit_x': exit_pts[:k, 0] if len(exit_pts) >= k else np.full(k, np.nan),
                'exit_y': exit_pts[:k, 1] if len(exit_pts) >= k else np.full(k, np.nan),
            })
        else:
            skip_df = pd.DataFrame({'skip_x_value': skip_x})
        skip_path = os.path.join(out_dir, 'baseline_phase3_skip_summary.csv')
        skip_df.to_csv(skip_path, index=False)
        out['baseline_phase3_skip_summary_csv'] = skip_path

    if isinstance(baseline_df, pd.DataFrame):
        removed_tables = baseline_df.attrs.get('cross_phase_removed_tables', None)
        prune_stats = baseline_df.attrs.get('cross_phase_prune_stats', None)
        if isinstance(prune_stats, dict) and len(prune_stats) > 0:
            stats_path = os.path.join(out_dir, 'baseline_cross_phase_prune_stats.csv')
            pd.DataFrame([prune_stats]).to_csv(stats_path, index=False)
            out['baseline_cross_phase_prune_stats_csv'] = stats_path
        if isinstance(removed_tables, dict):
            for key, fname in [
                ('phase2_removed', 'phase2_removed_within_22m.csv'),
                ('near_removed', 'near_removed_within_22m.csv'),
                ('all_removed', 'all_removed_within_22m.csv'),
            ]:
                tbl = removed_tables.get(key, None)
                if isinstance(tbl, pd.DataFrame) and len(tbl) > 0:
                    path = os.path.join(out_dir, fname)
                    tbl.to_csv(path, index=False)
                    out[f'{key}_csv'] = path
    return out

def build_adaptive_candidate_domain(
    geom: SurveyGeometry,
    adapt: AdaptiveConfig,
    vsp_xy: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    use_vsp = (vsp_xy is not None and len(vsp_xy) > 0 and adapt.candidate_source == "vsp")

    if use_vsp:
        pts = filter_points_in_geom(vsp_xy, geom.valid_center_geom)   
        if len(pts) > 0:
            order = serpentine_order_indices(
                pts,
                row_tol_m=float(adapt.vsp_row_tol_m),
                end_corner="upper_right",
                axis=str(adapt.vsp_serpentine_axis),
            )
            domain_xy = np.asarray(pts, dtype=float)[order]
    
            hot_mask = (
                points_in_polygon(domain_xy, geom.hot_poly, include_boundary=True)
                if geom.has_hot_prior else np.zeros(len(domain_xy), dtype=bool)
            )
            route_rank = np.arange(len(domain_xy), dtype=int)
            return domain_xy, hot_mask, route_rank

    domain_xy = make_candidate_points_in_geom(
        geom.cfg.bounds,
        adapt.candidate_spacing_m,
        geom.valid_center_geom,   
    )
    hot_mask = (
        points_in_polygon(domain_xy, geom.hot_poly, include_boundary=True)
        if geom.has_hot_prior else np.zeros(len(domain_xy), dtype=bool)
    )
    return domain_xy, hot_mask, None

def build_baseline_points(geom: SurveyGeometry, cfg: BaselineConfig, vsp_xy: Optional[object] = None) -> np.ndarray:
    source = str(getattr(cfg, "candidate_source", "auto")).lower()
    has_csv_points = False
    if isinstance(vsp_xy, pd.DataFrame):
        has_csv_points = len(vsp_xy) > 0
    elif vsp_xy is not None:
        has_csv_points = len(np.asarray(vsp_xy, dtype=float)) > 0

    if source == "auto":
        source = "vsp" if has_csv_points else "hex"

    if source in {"vsp", "baseline_csv"}:
        if not has_csv_points:
            raise ValueError("baseline candidate_source='vsp' requires --baseline_csv (or --vsp_csv).")

        raw_data = vsp_xy
        if bool(getattr(cfg, "vsp_apply_valid_center", False)):
            if isinstance(raw_data, pd.DataFrame):
                pts0 = baseline_points_from_frame(raw_data)
                keep = points_in_polygon(pts0, geom.valid_center_geom, include_boundary=True)
                raw_data = raw_data.loc[keep].reset_index(drop=True)
            else:
                pts0 = np.asarray(raw_data, dtype=float)
                keep = points_in_polygon(pts0, geom.valid_center_geom, include_boundary=True)
                raw_data = pts0[keep]

        pts = order_baseline_csv_points(
            raw_data,
            line_tol_m=float(getattr(cfg, "vsp_line_tol_m", 22.0)),
            end_corner=str(getattr(cfg, "vsp_end_corner", "upper_right")),
            axis=str(getattr(cfg, "vsp_serpentine_axis", "col")).lower(),
        )
        if int(getattr(cfg, "max_samples", 0)) > 0:
            pts = pts[: int(cfg.max_samples)]
        return np.asarray(pts, dtype=float)

    if source == "grid":
        pts = make_candidate_points_in_geom(geom.cfg.bounds, cfg.warm_spacing_m, geom.valid_center_geom)
        axis = str(getattr(cfg, "vsp_serpentine_axis", "row")).lower()
        if axis == "none":
            axis = "row"
        if len(pts) > 1:
            row_tol = max(0.45 * float(cfg.warm_spacing_m), 1e-6)
            order = serpentine_order_indices(
                pts,
                row_tol_m=row_tol,
                end_corner=str(getattr(cfg, "vsp_end_corner", "upper_right")),
                axis=axis,
            )
            pts = pts[order]
        if int(getattr(cfg, "max_samples", 0)) > 0:
            pts = pts[: int(cfg.max_samples)]
        return pts

    if source != "hex":
        raise ValueError(f"Unsupported baseline candidate source: {source}")

    r = 0.5 * geom.cfg.measurement_diameter_m
    warm = hex_grid_in_geom(geom.valid_center_geom, cfg.warm_spacing_m)
    hot = hex_grid_in_geom(geom.valid_center_hot_geom, cfg.hot_spacing_m)
    if len(hot) == 0:
        pts = warm
    else:
        pts = deduplicate_points(np.vstack([warm, hot]), min_spacing=max(r * 0.9, cfg.hot_spacing_m * 0.55))

    if int(getattr(cfg, "max_samples", 0)) > 0:
        pts = pts[: int(cfg.max_samples)]
    return pts

def compute_time_metrics(path_length_m: float, n_used: int, timing: TimingConfig) -> Dict[str, float]:
    speed_m_per_min = max(float(timing.travel_speed_kmph) * 1000.0 / 60.0, 1e-12)
    travel_time_min = float(path_length_m) / speed_m_per_min
    station_time_total_min = float(n_used) * float(timing.station_time_min)
    total_time_min = travel_time_min + station_time_total_min
    return {
        "travel_time_min": travel_time_min,
        "station_time_total_min": station_time_total_min,
        "total_time_min": total_time_min,
    }

# ---------------------------------------------------------------------
# Truth generation
# ---------------------------------------------------------------------


def _principal_axis_stats(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]), 1.0, 1.0
    center = np.mean(pts[:, :2], axis=0)
    if len(pts) < 3:
        return center, np.array([1.0, 0.0]), np.array([0.0, 1.0]), 1.0, 1.0
    centered = pts[:, :2] - center
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    idx = int(np.argmax(vals))
    v_major = vecs[:, idx]
    v_major = v_major / max(np.linalg.norm(v_major), 1e-12)
    v_minor = np.array([-v_major[1], v_major[0]])
    proj_major = centered @ v_major
    proj_minor = centered @ v_minor
    major_span = float(np.percentile(proj_major, 95) - np.percentile(proj_major, 5))
    minor_span = float(np.percentile(proj_minor, 95) - np.percentile(proj_minor, 5))
    return center, v_major, v_minor, max(major_span, 1.0), max(minor_span, 1.0)

def _blurred_mask(mask: np.ndarray, sigma: float) -> np.ndarray:
    arr = gaussian_filter(mask.astype(float), sigma=float(sigma), mode="nearest")
    m = float(np.nanmax(arr))
    if m > 0:
        arr = arr / m
    return np.clip(arr, 0.0, 1.0)

def _smooth_random_field(shape: Tuple[int, int], sigma: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, 1.0, size=shape)
    arr = gaussian_filter(noise, sigma=float(max(sigma, 1e-3)), mode="reflect")
    arr = arr - float(np.mean(arr))
    s = float(np.std(arr))
    if s > 1e-9:
        arr = arr / s
    return arr

def _guided_gaussian_plume(
    X: np.ndarray,
    Y: np.ndarray,
    cx: float,
    cy: float,
    v_major: np.ndarray,
    major_span: float,
    minor_span: float,
    amp: float,
    rng: np.random.Generator,
    theta_jitter_deg: float,
    fill_fraction: float,
    meander_strength: float,
    roughness_strength: float,
    roughness_field: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    ang = float(rng.normal(0.0, np.deg2rad(theta_jitter_deg)))
    ca, sa = math.cos(ang), math.sin(ang)
    v_major = np.asarray(v_major, dtype=float)
    v_major = np.array([ca * v_major[0] - sa * v_major[1], sa * v_major[0] + ca * v_major[1]])
    v_major = v_major / max(np.linalg.norm(v_major), 1e-12)
    v_minor = np.array([-v_major[1], v_major[0]])

    dx = X - float(cx)
    dy = Y - float(cy)
    along = v_major[0] * dx + v_major[1] * dy
    cross = v_minor[0] * dx + v_minor[1] * dy

    aspect = max(float(major_span) / max(float(minor_span), 1e-6), 1.0)
    elong = float(np.clip((aspect - 2.0) / 4.0, 0.0, 1.0))

    L_down = float(rng.uniform(0.80 + 0.15 * elong, 1.20 + 0.25 * elong) * max(major_span, 1.0))
    L_up = float(rng.uniform(0.10, 0.22) * max(major_span, 1.0))
    sigma_cross0 = float(rng.uniform(0.22, 0.40 + 0.08 * elong) * max(minor_span, 1.0) + (0.03 + 0.02 * elong) * max(major_span, 1.0))
    growth = float(rng.uniform(0.45, 1.05))
    sigma_cross = sigma_cross0 * (1.0 + growth * np.clip(along, 0.0, None) / max(L_down, 1.0))

    down = np.clip(along, 0.0, None)
    up = np.clip(-along, 0.0, None)

    meander_amp = float(meander_strength) * (0.65 + 0.35 * elong) * max(minor_span, 1.0)
    lam1 = float(rng.uniform(0.28, 0.60) * max(major_span, 1.0))
    lam2 = float(rng.uniform(0.12, 0.26) * max(major_span, 1.0))
    phase1 = float(rng.uniform(0.0, 2.0 * np.pi))
    phase2 = float(rng.uniform(0.0, 2.0 * np.pi))
    taper = 1.0 - np.exp(-down / max(0.08 * max(major_span, 1.0), 1e-6))
    centerline_shift = taper * (
        meander_amp * np.sin(2.0 * np.pi * along / max(lam1, 1e-6) + phase1)
        + 0.45 * meander_amp * np.sin(2.0 * np.pi * along / max(lam2, 1e-6) + phase2)
    )
    cross_eff = cross - centerline_shift

    core = amp * np.exp(-down / max(L_down, 1e-6))
    core *= np.exp(-0.5 * (up / max(L_up, 1e-6)) ** 2)
    core *= np.exp(-0.5 * (cross_eff / np.maximum(sigma_cross, 1e-6)) ** 2)

    broad = float(fill_fraction) * amp * np.exp(-0.5 * (down / max((1.10 + 0.25 * elong) * major_span, 1e-6)) ** 2)
    broad *= np.exp(-0.5 * (cross_eff / max((0.60 + 0.15 * elong) * minor_span + (0.06 + 0.02 * elong) * major_span, 1e-6)) ** 2)

    tail_amp = (0.15 + 0.30 * elong) * float(fill_fraction) * amp
    tail = tail_amp * np.exp(-down / max((1.55 + 0.35 * elong) * major_span, 1e-6))
    tail *= np.exp(-0.5 * (cross_eff / max((0.80 + 0.10 * elong) * minor_span + 0.08 * major_span, 1e-6)) ** 2)

    plume = core + broad + tail
    if roughness_field is not None and roughness_strength > 0:
        mod = np.clip(1.0 + float(roughness_strength) * roughness_field, 0.55, 1.65)
        plume = plume * mod

    meta = {
        "theta_deg": float(np.degrees(math.atan2(v_major[1], v_major[0]))),
        "L_down": L_down,
        "L_up": L_up,
        "sigma_cross0": sigma_cross0,
        "meander_amp": meander_amp,
    }
    return plume, meta

def _anisotropic_bump(
    X: np.ndarray,
    Y: np.ndarray,
    cx: float,
    cy: float,
    v_major: np.ndarray,
    major_span: float,
    minor_span: float,
    amp: float,
    rng: np.random.Generator,
    theta_jitter_deg: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    ang = float(rng.normal(0.0, np.deg2rad(theta_jitter_deg)))
    ca, sa = math.cos(ang), math.sin(ang)
    v_major = np.asarray(v_major, dtype=float)
    v_major = np.array([ca * v_major[0] - sa * v_major[1], sa * v_major[0] + ca * v_major[1]])
    v_major = v_major / max(np.linalg.norm(v_major), 1e-12)
    v_minor = np.array([-v_major[1], v_major[0]])
    dx = X - float(cx)
    dy = Y - float(cy)
    xr = v_major[0] * dx + v_major[1] * dy
    yr = v_minor[0] * dx + v_minor[1] * dy
    sigma_long = float(rng.uniform(0.10, 0.22) * max(major_span, 1.0))
    sigma_short = float(rng.uniform(0.14, 0.28) * max(minor_span, 1.0) + 0.02 * max(major_span, 1.0))
    bump = amp * np.exp(-0.5 * ((xr / max(sigma_long, 1e-6)) ** 2 + (yr / max(sigma_short, 1e-6)) ** 2))
    meta = {
        "theta_deg": float(np.degrees(math.atan2(v_major[1], v_major[0]))),
        "sigma_long": sigma_long,
        "sigma_short": sigma_short,
    }
    return bump, meta

def generate_truth_field(geom: SurveyGeometry, truth_cfg: TruthConfig, ops: OperationalCriteria, rng: np.random.Generator, meteo_steps: Optional[Sequence[MeteoStep]] = None,) -> Tuple[np.ndarray, Dict[str, object]]:
    cfg = geom.cfg
    xs, ys, X, Y = make_mesh(cfg)
    pts = np.column_stack([X.ravel(), Y.ravel()])

    survey_mask = points_in_polygon(pts, geom.survey_poly, include_boundary=True).reshape(X.shape)
    prior_warm_poly = scale_polygon_about_origin(geom.warm_poly, truth_cfg.truth_scale, (truth_cfg.origin_x, truth_cfg.origin_y))
    prior_warm_mask = points_in_polygon(pts, prior_warm_poly, include_boundary=True).reshape(X.shape)
    if geom.has_hot_prior:
        prior_hot_poly = scale_polygon_about_origin(geom.hot_poly, truth_cfg.truth_scale, (truth_cfg.origin_x, truth_cfg.origin_y))
        prior_hot_mask = points_in_polygon(pts, prior_hot_poly, include_boundary=True).reshape(X.shape)
    else:
        prior_hot_poly = np.zeros((0, 2), dtype=float)
        prior_hot_mask = np.zeros_like(X, dtype=bool)

    Z = np.full_like(X, np.nan, dtype=float)
    Z[survey_mask] = cfg.background_level

    survey_pts = pts[survey_mask.ravel()]
    if len(survey_pts) == 0:
        raise ValueError("Survey mask is empty")

    prior_warm_pts = pts[prior_warm_mask.ravel()]
    axis_pts = prior_warm_pts if len(prior_warm_pts) > 0 else survey_pts

    warm_center, v_major, v_minor, major_span, minor_span = _principal_axis_stats(axis_pts)
    if geom.has_hot_prior:
        hot_support_pts = pts[prior_hot_mask.ravel()]
        if len(hot_support_pts) == 0:
            hot_support_pts = pts[points_in_polygon(pts, geom.hot_poly, include_boundary=True)]
        hot_center, _, _, _, _ = _principal_axis_stats(hot_support_pts if len(hot_support_pts) else survey_pts)
        toward_warm = np.asarray(warm_center) - np.asarray(hot_center)
        if np.dot(v_major, toward_warm) < 0:
            v_major = -v_major
            v_minor = -v_minor
    else:
        centered = axis_pts - warm_center[None, :]
        proj_major = centered @ v_major
        proj_minor = centered @ v_minor
        shift_major = float(rng.uniform(-0.28, 0.28) * major_span)
        shift_minor = float(rng.uniform(-0.24, 0.24) * minor_span)
        target = warm_center + shift_major * v_major + shift_minor * v_minor
        d_target = np.linalg.norm(axis_pts - target[None, :], axis=1)
        bias_axis = np.exp(-0.5 * (proj_minor / max(0.80 * minor_span + 0.05 * major_span, 1.0)) ** 2)
        bias_target = np.exp(-0.5 * (d_target / max(0.22 * max(major_span, minor_span), 1.0)) ** 2)
        weights_center = 0.20 + 0.45 * bias_axis + 0.35 * bias_target
        weights_center = np.clip(weights_center, 1e-12, None)
        hot_center = axis_pts[int(rng.choice(len(axis_pts), p=weights_center / np.sum(weights_center)))]
        if float(rng.random()) < 0.5:
            v_major = -v_major
            v_minor = -v_minor

    # ------------------------------------------------------------
    # 1) lock hot core around an anchor (default: 0,0)
    # ------------------------------------------------------------
    if geom.has_hot_prior and truth_cfg.lock_hot_anchor:
        hot_anchor = np.array([truth_cfg.hot_anchor_x, truth_cfg.hot_anchor_y], dtype=float)

        # Prefer to snap to the nearest point inside the hot support if possible
        hot_support_pts = pts[prior_hot_mask.ravel()]
        if len(hot_support_pts) == 0:
            hot_support_pts = pts[points_in_polygon(pts, geom.hot_poly, include_boundary=True)]

        ref_pts = hot_support_pts if len(hot_support_pts) > 0 else survey_pts
        idx0 = int(np.argmin(np.linalg.norm(ref_pts - hot_anchor[None, :], axis=1)))
        hot_center = np.asarray(ref_pts[idx0], dtype=float)

    # ------------------------------------------------------------
    # 2) small weather-driven tilt for the warm contour direction
    # ------------------------------------------------------------
    tilt_deg = compute_small_meteo_tilt_deg(
        meteo_steps,
        max_abs_deg=float(truth_cfg.meteo_tilt_max_deg),
    )

    if abs(tilt_deg) > 1e-12:
        ang = np.deg2rad(tilt_deg)
        ca, sa = np.cos(ang), np.sin(ang)

        v_major = np.array([
            ca * v_major[0] - sa * v_major[1],
            sa * v_major[0] + ca * v_major[1],
        ], dtype=float)
        v_major = v_major / max(np.linalg.norm(v_major), 1e-12)
        v_minor = np.array([-v_major[1], v_major[0]], dtype=float)

    env_sigma = max(2.0, 0.024 * cfg.grid_n)
    survey_env = _blurred_mask(survey_mask, env_sigma)
    prior_warm_env = _blurred_mask(prior_warm_mask, env_sigma)
    prior_hot_env = _blurred_mask(prior_hot_mask, max(1.5, 0.018 * cfg.grid_n)) if np.any(prior_hot_mask) else np.zeros_like(X, dtype=float)

    blend = float(np.clip(truth_cfg.prior_softness, 0.0, 1.0))
    shape_env = np.clip(
        (1.0 - blend) * survey_env + blend * prior_warm_env,
        0.0, 1.0
    )
    shape_gate = np.clip(
        (1.0 - float(truth_cfg.shape_follow_strength))
        + float(truth_cfg.shape_follow_strength) * shape_env,
        0.0, 1.0
    )
    
    along_all = (pts - hot_center) @ v_major
    cross_all = np.abs((pts - hot_center) @ v_minor)
    survey_w = np.where(survey_mask.ravel(), 1.0, 0.0)
    if geom.has_hot_prior:
        prior_w = 0.30 + 0.70 * prior_warm_env.ravel()
        hot_boost = 1.0 + float(truth_cfg.prior_center_bias - 1.0) * prior_hot_env.ravel()
        down_bias = 0.35 + np.exp(-0.5 * ((np.clip(along_all, 0.0, None) - 0.15 * major_span) / max(0.40 * major_span, 1.0)) ** 2)
    else:
        prior_w = 0.55 + 0.45 * prior_warm_env.ravel()
        hot_boost = np.ones_like(prior_w)
        down_bias = 0.55 + np.exp(-0.5 * ((np.clip(along_all, 0.0, None) - 0.05 * major_span) / max(0.55 * major_span, 1.0)) ** 2)
    line_bias = np.exp(-0.5 * (cross_all / max(0.85 * minor_span + 0.08 * major_span, 1.0)) ** 2)
    weights = survey_w * prior_w * hot_boost * line_bias * down_bias
    valid_pts = pts[weights > 0]
    valid_w = weights[weights > 0]
    if len(valid_pts) == 0:
        raise ValueError("No valid truth points after applying soft prior")

    n_sources = int(rng.integers(truth_cfg.n_sources_min, truth_cfg.n_sources_max + 1))
    centers: List[np.ndarray] = [np.asarray(hot_center, dtype=float)]
    select_w = valid_w.copy()
    while len(centers) < min(n_sources, len(valid_pts)):
        p = select_w / np.sum(select_w)
        j = int(rng.choice(len(valid_pts), p=p))
        pick = valid_pts[j]
        if any(np.linalg.norm(pick - c) < 0.14 * max(major_span, minor_span) for c in centers):
            select_w[j] *= 0.2
            if np.sum(select_w) <= 0:
                break
            continue
        centers.append(np.asarray(pick, dtype=float))
        select_w[j] = 0.0
        if np.sum(select_w) <= 0:
            break

    rough_field = _smooth_random_field(X.shape, sigma=float(max(truth_cfg.roughness_sigma_cells, 1.0)), rng=rng)
    rough_field *= survey_env

    sources = []
    mean_amp = 0.0
    for k, center in enumerate(centers):
        cx, cy = float(center[0]), float(center[1])
        amp = float(rng.uniform(truth_cfg.amp_min, truth_cfg.amp_max))
        mean_amp += amp
        use_plume = (k == 0) or (float(rng.random()) < float(truth_cfg.plume_probability))
        if use_plume:
            comp, comp_meta = _guided_gaussian_plume(
                X, Y, cx, cy, v_major, major_span, minor_span, amp, rng,
                truth_cfg.theta_jitter_deg, truth_cfg.shape_fill_fraction,
                truth_cfg.meander_strength, truth_cfg.roughness_strength, rough_field,
            )
            comp *= shape_gate
            if k == 0 and np.any(prior_hot_mask):
                comp *= (1.0 + 0.30 * prior_hot_env)
            src_kind = "guided_plume_irregular"
        else:
            comp, comp_meta = _anisotropic_bump(
                X, Y, cx, cy, v_major, major_span, minor_span, amp * 0.85, rng, truth_cfg.theta_jitter_deg
            )
            comp *= shape_gate
            if truth_cfg.roughness_strength > 0:
                mod = np.clip(1.0 + 0.20 * truth_cfg.roughness_strength * rough_field, 0.70, 1.45)
                comp *= mod
            src_kind = "shape_bump"

        Z[survey_mask] += comp[survey_mask]
        sources.append({"kind": src_kind, "center": (cx, cy), "amp": amp, **comp_meta})

    if len(centers) > 0 and truth_cfg.shape_fill_fraction > 0:
        fill_amp = float(truth_cfg.shape_fill_fraction) * (mean_amp / max(len(centers), 1))
        dx = X - float(hot_center[0])
        dy = Y - float(hot_center[1])
        along = v_major[0] * dx + v_major[1] * dy
        cross = v_minor[0] * dx + v_minor[1] * dy
        down = np.clip(along, 0.0, None)
        spill_frac = 0.40
        diffuse_env = (1.0 - spill_frac) * prior_warm_env + spill_frac * survey_env

        diffuse = 1.70 * fill_amp * diffuse_env
        diffuse *= np.exp(-down / max(1.85 * major_span, 1.0))
        diffuse *= np.exp(-0.5 * (cross / max(1.10 * minor_span + 0.15 * major_span, 1.0)) ** 2)

        if truth_cfg.roughness_strength > 0:
            diffuse *= np.clip(1.0 + 0.12 * truth_cfg.roughness_strength * rough_field, 0.85, 1.20)

        Z[survey_mask] += diffuse[survey_mask]

        if np.any(prior_hot_mask):
           hot_bridge = gaussian_filter(
               prior_hot_mask.astype(float),
               sigma=max(1.5, 0.012 * cfg.grid_n),
               mode="nearest",
           )
           hot_bridge = hot_bridge / max(float(np.nanmax(hot_bridge)), 1e-12)
           hot_bridge = np.clip(hot_bridge, 0.0, 1.0)

           warm_thr = float(ops.thresholds[0]) if len(ops.thresholds) > 0 else float(cfg.background_level)
           bridge_target = cfg.background_level + 1.08 * warm_thr * hot_bridge
           
           gain = 0.42 * hot_bridge
           Z[survey_mask] = (
                (1.0 - gain[survey_mask]) * Z[survey_mask]
                + gain[survey_mask] * np.maximum(Z[survey_mask], bridge_target[survey_mask])     
           )

    if truth_cfg.blur_sigma_cells > 0:
        zfill = np.nan_to_num(Z, nan=cfg.background_level)
        zblur = gaussian_filter(zfill, sigma=float(truth_cfg.blur_sigma_cells), mode="nearest")
        Z[survey_mask] = zblur[survey_mask]
    if (not geom.has_hot_prior) and float(np.nanmax(Z[survey_mask])) < float(max(ops.thresholds)):
        target_thr = float(max(ops.thresholds))
        core_peak = target_thr + 0.35 * max(target_thr, 1.0)

        dx = X - float(hot_center[0])
        dy = Y - float(hot_center[1])
        xr = v_major[0] * dx + v_major[1] * dy
        yr = v_minor[0] * dx + v_minor[1] * dy

        sigma_long = max(0.06 * major_span, 0.45 * cfg.measurement_diameter_m)
        sigma_short = max(0.05 * minor_span + 0.12 * cfg.measurement_diameter_m, 0.35 * cfg.measurement_diameter_m)

        hot_core = core_peak * np.exp(-0.5 * ((xr / sigma_long) ** 2 + (yr / sigma_short) ** 2))
        hot_core *= np.clip(shape_gate, 0.6, 1.0)

        Z[survey_mask] = np.maximum(Z[survey_mask], cfg.background_level + hot_core[survey_mask])

    if float(getattr(truth_cfg, "value_scale", 1.0)) != 1.0:
        vscale = max(float(getattr(truth_cfg, "value_scale", 1.0)), 0.0)
        Z[survey_mask] = cfg.background_level + vscale * (Z[survey_mask] - cfg.background_level)

        z_before = np.array(Z, copy=True)

        warm_support = np.logical_or(
            prior_warm_mask,
            np.nan_to_num(Z, nan=cfg.background_level) >= float(ops.thresholds[0]),
        )
        warm_support = binary_dilation(warm_support, iterations=2)

        sigma_warm = 3.0
        z_num = gaussian_filter(
            np.nan_to_num(Z, nan=cfg.background_level) * warm_support.astype(float),
            sigma=sigma_warm,
            mode="nearest",
        )
        z_den = gaussian_filter(
            warm_support.astype(float),
            sigma=sigma_warm,
            mode="nearest",
        )
        z_warm = np.divide(z_num, np.maximum(z_den, 1e-12))

        warm_floor = cfg.background_level + 1.05 * float(ops.thresholds[0])
        target_warm = np.maximum(z_warm, warm_floor)

        lift_mask = prior_warm_mask & (target_warm > Z)
        Z[lift_mask] = 0.30 * Z[lift_mask] + 0.70 * target_warm[lift_mask]

        hot_keep = np.nan_to_num(z_before, nan=cfg.background_level) >= float(max(ops.thresholds))
        Z[hot_keep] = np.maximum(Z[hot_keep], z_before[hot_keep])
        
    if len(ops.thresholds) > 0:
        hot_mask_now = np.nan_to_num(Z, nan=cfg.background_level) >= float(max(ops.thresholds))
        Z = enforce_warm_continuity(
            Z,
            survey_mask=survey_mask,
            background_level=cfg.background_level,
            warm_threshold=float(ops.thresholds[0]),
            hot_mask=hot_mask_now,
            smooth_sigma=2.2,
            blend=0.95,
        )

    meta = {
        "survey_mask": survey_mask,
        "pred_survey_poly": geom.survey_poly,
        "pred_warm_poly": geom.warm_poly,
        "pred_hot_poly": geom.hot_poly,
        "has_hot_prior": geom.has_hot_prior,
        "truth_prior_warm_poly": prior_warm_poly,
        "truth_prior_hot_poly": prior_hot_poly,
        "truth_prior_hot_mask": prior_hot_mask,
        "truth_prior_warm_mask": prior_warm_mask,
        "shape_env": shape_env,
        "prior_hot_env": prior_hot_env,
        "major_axis_theta_deg": float(np.degrees(math.atan2(v_major[1], v_major[0]))),
        "major_span": float(major_span),
        "minor_span": float(minor_span),
        "sources": sources,
        "hot_anchor_x": float(truth_cfg.hot_anchor_x),
        "hot_anchor_y": float(truth_cfg.hot_anchor_y),
        "meteo_tilt_deg": float(tilt_deg),
    }
    return Z, meta

def enforce_warm_continuity(
    Z: np.ndarray,
    survey_mask: np.ndarray,
    background_level: float,
    warm_threshold: float,
    hot_mask: Optional[np.ndarray] = None,
    smooth_sigma: float = 1.0,
    blend: float = 0.75,
) -> np.ndarray:
    Z0 = np.array(Z, dtype=float, copy=True)
    zplot = np.nan_to_num(Z0, nan=background_level)

    warm_mask = zplot >= float(warm_threshold)
    warm_mask_filled = binary_fill_holes(warm_mask) & survey_mask
    holes = warm_mask_filled & (~warm_mask)

    if not np.any(holes):
        Z0[~survey_mask] = np.nan
        return Z0

    # normalized smoothing
    warm_support = warm_mask.astype(float)
    z_num = gaussian_filter(zplot * warm_support, sigma=float(smooth_sigma), mode="nearest")
    z_den = gaussian_filter(warm_support, sigma=float(smooth_sigma), mode="nearest")
    z_smooth = np.full_like(z_num, background_level, dtype=float)
    valid = np.isfinite(z_num) & np.isfinite(z_den) & (z_den > 1e-12)
    np.divide(z_num, z_den, out=z_smooth, where=valid)

    # floor
    floor = background_level + 1.15 * float(warm_threshold)

    Z1 = zplot.copy()
    Z1[holes] = np.maximum(z_smooth[holes], floor)

    feather = gaussian_filter(holes.astype(float), sigma=1.6 * float(smooth_sigma), mode="nearest")
    feather = np.clip(feather / max(float(np.max(feather)), 1e-12), 0.0, 1.0)
    feather *= float(blend)

    Z2 = (1.0 - feather) * zplot + feather * gaussian_filter(Z1, sigma=float(smooth_sigma), mode="nearest")

    if hot_mask is not None and np.any(hot_mask):
        Z2[hot_mask] = np.maximum(Z2[hot_mask], zplot[hot_mask])

    Z2[~survey_mask] = np.nan
    return Z2

def apply_meteo_warp_to_truth(
    Z_true: np.ndarray,
    geom: SurveyGeometry,
    meteo_steps: Sequence[MeteoStep],
    advect_scale: float = 2e-4,
    blur_scale: float = 0.03,
    hot_threshold: Optional[float] = None,
    warm_threshold: Optional[float] = None,
    preserve_hot_core: bool = True,
    post_fill_sigma: float = 0.8,
) -> np.ndarray:
    cfg = geom.cfg
    xs, ys, X, Y = make_mesh(cfg)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    survey_mask = points_in_polygon(pts, geom.survey_poly, include_boundary=True).reshape(X.shape)

    dx = (cfg.bounds[1] - cfg.bounds[0]) / max(cfg.grid_n - 1, 1)
    dy = (cfg.bounds[3] - cfg.bounds[2]) / max(cfg.grid_n - 1, 1)

    bg = float(cfg.background_level)
    z_full = np.nan_to_num(Z_true, nan=bg)
    excess = np.clip(z_full - bg, 0.0, None)

    if hot_threshold is None:
        hot_threshold = float(np.nanmax(z_full) * 0.80)

    if preserve_hot_core:
        hot_core = np.where(z_full >= hot_threshold, excess, 0.0)
        warm_tail = np.clip(excess - hot_core, 0.0, None)
    else:
        hot_core = np.zeros_like(excess)
        warm_tail = excess.copy()

    z_state = warm_tail.copy()
    z_state[~survey_mask] = 0.0

    for step in meteo_steps:
        dt_h = max(float(step.t1_h) - float(step.t0_h), 0.0)
        if dt_h <= 0:
            continue

        u, v = _wind_to_uv(step.wind_dir_deg, step.wind_speed_mps)

        shift_x_cells = advect_scale * (u * 3600.0 * dt_h) / max(dx, 1e-9)
        shift_y_cells = advect_scale * (v * 3600.0 * dt_h) / max(dy, 1e-9)

        z_state = ndi_shift(z_state, shift=(shift_y_cells, shift_x_cells), order=1, mode="nearest")

        _, grow_cross, _ = _stability_growth_rates(step.stability)
        sigma_cells = blur_scale * (grow_cross * dt_h) / max(max(dx, dy), 1e-9)
        if sigma_cells > 1e-6:
            z_state = gaussian_filter(z_state, sigma=sigma_cells, mode="nearest")

        if float(step.rain_mmph) > 0:
            z_state *= np.clip(1.0 - 0.08 * float(step.rain_mmph), 0.65, 1.0)

        z_state[~survey_mask] = 0.0

    Z_out = bg + hot_core + z_state
    Z_out[~survey_mask] = np.nan

    if warm_threshold is not None:
       hot_mask_now = None
       if hot_threshold is not None:
          hot_mask_now = np.nan_to_num(Z_out, nan=bg) >= float(hot_threshold)

       Z_out = enforce_warm_continuity(
            Z_out,
            survey_mask=survey_mask,
            background_level=bg,
            warm_threshold=float(warm_threshold),
            hot_mask=hot_mask_now,
            smooth_sigma=float(post_fill_sigma),
            blend=0.80,
        ) 
        
    Z_out[~survey_mask] = np.nan
    return Z_out

def _wind_to_uv(wind_dir_deg: float, wind_speed_mps: float) -> Tuple[float, float]:
    theta = np.deg2rad(270.0 - float(wind_dir_deg))
    u = float(wind_speed_mps * np.cos(theta))
    v = float(wind_speed_mps * np.sin(theta))
    return u, v

def compute_small_meteo_tilt_deg(
    meteo_steps: Optional[Sequence[MeteoStep]],
    max_abs_deg: float = 5.0,
) -> float:
    
    if not meteo_steps:
        return 0.0

    vec = np.zeros(2, dtype=float)
    total_w = 0.0

    for step in meteo_steps:
        dt_h = max(float(step.t1_h) - float(step.t0_h), 0.0)
        if dt_h <= 0:
            continue
        u, v = _wind_to_uv(step.wind_dir_deg, step.wind_speed_mps)
        vec += dt_h * np.array([u, v], dtype=float)
        total_w += dt_h

    if total_w <= 0 or np.linalg.norm(vec) < 1e-12:
        return 0.0

    theta_deg = float(np.degrees(np.arctan2(vec[1], vec[0])))
    return float(np.clip(theta_deg, -max_abs_deg, max_abs_deg))

def _stability_growth_rates(stability: str) -> Tuple[float, float, float]:
    tab = {
        "A": (180.0, 120.0, 0.45),
        "B": (140.0, 100.0, 0.38),
        "C": (100.0, 75.0, 0.30),
        "D": (70.0, 55.0, 0.24),
        "E": (50.0, 38.0, 0.18),
        "F": (35.0, 28.0, 0.14),
    }
    return tab.get(str(stability).upper(), tab["D"])

def _oriented_gaussian_puff_field(
    X: np.ndarray,
    Y: np.ndarray,
    cx: float,
    cy: float,
    theta_rad: float,
    sigma_along: float,
    sigma_cross: float,
    mass: float,
) -> np.ndarray:
    ca = math.cos(theta_rad)
    sa = math.sin(theta_rad)
    dx = X - float(cx)
    dy = Y - float(cy)
    along = ca * dx + sa * dy
    cross = -sa * dx + ca * dy

    sigma_along = max(float(sigma_along), 1e-6)
    sigma_cross = max(float(sigma_cross), 1e-6)

    z = float(mass) / (2.0 * np.pi * sigma_along * sigma_cross)
    z *= np.exp(-0.5 * ((along / sigma_along) ** 2 + (cross / sigma_cross) ** 2))
    return z

def _build_puff_precomp(geom: SurveyGeometry, truth_cfg: TruthConfig) -> Dict[str, object]:
    cfg = geom.cfg
    xs, ys, X, Y = make_mesh(cfg)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    survey_mask = points_in_polygon(pts, geom.survey_poly, include_boundary=True).reshape(X.shape)

    prior_warm_poly = scale_polygon_about_origin(
        geom.warm_poly, truth_cfg.truth_scale, (truth_cfg.origin_x, truth_cfg.origin_y)
    )
    prior_warm_mask = points_in_polygon(pts, prior_warm_poly, include_boundary=True).reshape(X.shape)

    if geom.has_hot_prior:
        prior_hot_poly = scale_polygon_about_origin(
            geom.hot_poly, truth_cfg.truth_scale, (truth_cfg.origin_x, truth_cfg.origin_y)
        )
        prior_hot_mask = points_in_polygon(pts, prior_hot_poly, include_boundary=True).reshape(X.shape)
    else:
        prior_hot_poly = np.zeros((0, 2), dtype=float)
        prior_hot_mask = np.zeros_like(X, dtype=bool)

    survey_env = _blurred_mask(survey_mask, max(2.0, 0.020 * cfg.grid_n))
    warm_env = _blurred_mask(prior_warm_mask, max(1.5, 0.018 * cfg.grid_n))
    hot_env = _blurred_mask(prior_hot_mask, max(1.2, 0.014 * cfg.grid_n)) if np.any(prior_hot_mask) else np.zeros_like(X)

    prior = 0.25 * survey_mask.astype(float) + 0.45 * survey_env + 0.30 * warm_env
    if geom.has_hot_prior:
        prior += float(truth_cfg.prior_center_bias) * hot_env

    prior[~survey_mask] = 0.0
    s = float(np.sum(prior))
    if s > 0:
        prior /= s
    else:
        prior[survey_mask] = 1.0 / max(int(np.sum(survey_mask)), 1)

    return {
        "X": X,
        "Y": Y,
        "survey_mask": survey_mask,
        "prior": prior,
        "prior_warm_poly": prior_warm_poly,
        "prior_hot_poly": prior_hot_poly,
        "prior_hot_mask": prior_hot_mask,
        "prior_warm_mask": prior_warm_mask,
    }

def _sample_source_xy_from_prior(
    pre: Dict[str, object],
    geom: SurveyGeometry,
    fcst_cfg: ForecastConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    X = np.asarray(pre["X"], dtype=float)
    Y = np.asarray(pre["Y"], dtype=float)
    prior = np.asarray(pre["prior"], dtype=float)
    survey_mask = np.asarray(pre["survey_mask"], dtype=bool)

    flat_p = prior.ravel()
    idx = int(rng.choice(len(flat_p), p=flat_p))
    x0 = float(X.ravel()[idx] + rng.normal(0.0, fcst_cfg.source_jitter_m))
    y0 = float(Y.ravel()[idx] + rng.normal(0.0, fcst_cfg.source_jitter_m))

    if not geom.warm_geom.covers(Point(x0, y0)):
        survey_pts = np.column_stack([X[survey_mask], Y[survey_mask]])
        j = int(np.argmin(np.linalg.norm(survey_pts - np.array([[x0, y0]]), axis=1)))
        x0, y0 = map(float, survey_pts[j])

    return np.array([x0, y0], dtype=float)

def simulate_puff_member(
    geom: SurveyGeometry,
    truth_cfg: TruthConfig,
    source_term: SourceTerm,
    meteo_steps: Sequence[MeteoStep],
    fcst_cfg: ForecastConfig,
    rng: np.random.Generator,
    pre: Optional[Dict[str, object]] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    cfg = geom.cfg
    pre = _build_puff_precomp(geom, truth_cfg) if pre is None else pre

    X = np.asarray(pre["X"], dtype=float)
    Y = np.asarray(pre["Y"], dtype=float)
    survey_mask = np.asarray(pre["survey_mask"], dtype=bool)
    prior = np.asarray(pre["prior"], dtype=float)

    rough = _smooth_random_field(X.shape, max(truth_cfg.roughness_sigma_cells, 1.0), rng)
    source_xy = _sample_source_xy_from_prior(pre, geom, fcst_cfg, rng)

    dose_air = np.zeros_like(X, dtype=float)
    dose_dep = np.zeros_like(X, dtype=float)
    active: List[Dict[str, float]] = []

    dt_h_nom = max(float(fcst_cfg.dt_min) / 60.0, 1e-6)
    horizon_h = float(fcst_cfg.horizon_h)

    if meteo_steps is None or len(meteo_steps) == 0:
        meteo_steps = [MeteoStep(0.0, horizon_h, 250.0, 5.0, "D", 0.0)]

    for step in meteo_steps:
        t = float(step.t0_h)
        step_end = min(float(step.t1_h), horizon_h)
        while t < step_end - 1e-12:
            dt_h = min(dt_h_nom, step_end - t, horizon_h - t)
            dt_s = 3600.0 * dt_h
            t_mid = t + 0.5 * dt_h

            if source_term.start_h <= t_mid <= source_term.end_h:
                rel_mass = float(source_term.release_rate * dt_h * rng.lognormal(mean=0.0, sigma=0.25))
                active.append({
                    "x": float(source_xy[0] + rng.normal(0.0, 0.35 * fcst_cfg.source_jitter_m)),
                    "y": float(source_xy[1] + rng.normal(0.0, 0.35 * fcst_cfg.source_jitter_m)),
                    "m": rel_mass,
                    "sig_a": float(max(source_term.source_sigma_m * rng.uniform(0.85, 1.20), 2.0)),
                    "sig_c": float(max(0.75 * source_term.source_sigma_m * rng.uniform(0.85, 1.20), 2.0)),
                })

            u, v = _wind_to_uv(step.wind_dir_deg, step.wind_speed_mps)
            grow_a, grow_c, rw_scale = _stability_growth_rates(step.stability)
            theta = math.atan2(v, u) if (abs(u) + abs(v)) > 1e-12 else 0.0

            kept: List[Dict[str, float]] = []
            for puff in active:
                puff["x"] += u * dt_s + rng.normal(0.0, fcst_cfg.random_walk_mps * rw_scale * math.sqrt(dt_s))
                puff["y"] += v * dt_s + rng.normal(0.0, fcst_cfg.random_walk_mps * rw_scale * math.sqrt(dt_s))

                puff["sig_a"] = math.sqrt(puff["sig_a"] ** 2 + (grow_a * dt_h) ** 2)
                puff["sig_c"] = math.sqrt(puff["sig_c"] ** 2 + (grow_c * dt_h) ** 2)

                field = _oriented_gaussian_puff_field(
                    X, Y,
                    puff["x"], puff["y"],
                    theta,
                    puff["sig_a"], puff["sig_c"],
                    puff["m"],
                )

                field *= np.clip(0.35 + 0.65 * prior, 0.10, 1.30)

                if truth_cfg.roughness_strength > 0:
                    field *= np.clip(1.0 + 0.10 * truth_cfg.roughness_strength * rough, 0.70, 1.40)

                field[~survey_mask] = 0.0
                dose_air += field * dt_h

                dep_step = (
                    source_term.dry_dep_vd * field * dt_s
                    + source_term.wet_scavenging * float(step.rain_mmph) * field * dt_h
                )
                dose_dep += dep_step

                loss_h = (
                    float(source_term.decay_lambda)
                    + 0.08 * float(source_term.dry_dep_vd)
                    + float(source_term.wet_scavenging) * float(step.rain_mmph)
                )
                puff["m"] *= math.exp(-loss_h * dt_h)

                if puff["m"] > 1e-8 and puff["sig_a"] < 3.0 * max(cfg.bounds[1] - cfg.bounds[0], cfg.bounds[3] - cfg.bounds[2]):
                    kept.append(puff)

            active = kept
            t += dt_h

    Z = cfg.background_level + dose_air + fcst_cfg.dose_dep_weight * dose_dep

    if truth_cfg.blur_sigma_cells > 0:
        Z = gaussian_filter(np.nan_to_num(Z, nan=cfg.background_level),
                            sigma=max(0.4 * truth_cfg.blur_sigma_cells, 0.0),
                            mode="nearest")
        
    if not np.any(np.isfinite(Z[survey_mask])):
        Z[survey_mask] = cfg.background_level
        
    Z[~survey_mask] = np.nan

    meta = {
        "survey_mask": survey_mask,
        "pred_warm_poly": geom.warm_poly,
        "pred_hot_poly": geom.hot_poly,
        "has_hot_prior": geom.has_hot_prior,
        "truth_prior_warm_poly": pre["prior_warm_poly"],
        "truth_prior_hot_poly": pre["prior_hot_poly"],
        "truth_prior_hot_mask": pre["prior_hot_mask"],
        "truth_prior_warm_mask": pre["prior_warm_mask"],
        "source_xy": (float(source_xy[0]), float(source_xy[1])),
    }
    return Z, meta

def simulate_puff_ensemble(
    geom: SurveyGeometry,
    truth_cfg: TruthConfig,
    source_term: SourceTerm,
    meteo_steps: Sequence[MeteoStep],
    fcst_cfg: ForecastConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    pre = _build_puff_precomp(geom, truth_cfg)
    survey_mask = np.asarray(pre["survey_mask"], dtype=bool)

    members = []
    valid_member_mask = []
    last_meta: Dict[str, object] = {}

    for _ in range(int(max(fcst_cfg.n_members, 1))):
        sub_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        Zm, meta = simulate_puff_member(
            geom, truth_cfg, source_term, meteo_steps, fcst_cfg, sub_rng, pre=pre
        )

        inside_vals = np.asarray(Zm, dtype=float)[survey_mask]
        is_valid = bool(np.any(np.isfinite(inside_vals)))
        members.append(np.asarray(Zm, dtype=float))
        valid_member_mask.append(is_valid)
        last_meta = meta

    arr = np.stack(members, axis=0)  
    valid_member_mask = np.asarray(valid_member_mask, dtype=bool)

    if not np.any(valid_member_mask):
        fallback = np.full_like(arr[0], geom.cfg.background_level, dtype=float)
        fallback[~survey_mask] = np.nan

        meta_out = dict(last_meta)
        meta_out["forecast_members"] = int(arr.shape[0])
        meta_out["forecast_valid_members"] = 0
        meta_out["forecast_fallback"] = True

        arr[:] = fallback[None, :, :]
        mean_grid = fallback.copy()
        std_grid = np.full_like(fallback, np.nan, dtype=float)
        std_grid[survey_mask] = float(fcst_cfg.min_std)
        return mean_grid, std_grid, arr, meta_out

    arr_valid = arr[valid_member_mask]
    valid_mean = np.mean(arr_valid[:, survey_mask], axis=0)

    arr_filled = arr.copy()
    for i in range(arr.shape[0]):
        if not valid_member_mask[i]:
            arr_filled[i, survey_mask] = valid_mean
            arr_filled[i, ~survey_mask] = np.nan

    mean_grid = np.full_like(arr_filled[0], np.nan, dtype=float)
    std_grid = np.full_like(arr_filled[0], np.nan, dtype=float)

    mean_grid[survey_mask] = np.mean(arr_filled[:, survey_mask], axis=0)
    std_grid[survey_mask] = np.std(arr_filled[:, survey_mask], axis=0, ddof=0)
    std_grid[survey_mask] = np.maximum(std_grid[survey_mask], float(fcst_cfg.min_std))

    meta_out = dict(last_meta)
    meta_out["forecast_members"] = int(arr.shape[0])
    meta_out["forecast_valid_members"] = int(np.sum(valid_member_mask))
    meta_out["forecast_fallback"] = False

    return mean_grid, std_grid, arr_filled, meta_out

def generate_truth_and_forecast(
    geom,
    truth_cfg,
    ops,
    source_term,
    meteo_steps,
    fcst_cfg,
    rng,
    truth_model="legacy",
    truth_weather_mode="none",
    truth_meteo_advect_scale=0.05,
    truth_meteo_blur_scale=0.35,
):
    truth_model = str(truth_model).lower()

    # ----------------------------
    # 1) truth generation
    # ----------------------------
    if truth_model == "legacy":
        Z_true, meta = generate_truth_field(
            geom, truth_cfg, ops, rng, meteo_steps=None
        )

        if truth_weather_mode == "warp" and meteo_steps:
            Z_true = apply_meteo_warp_to_truth(
                Z_true,
                geom,
                meteo_steps,
                advect_scale=truth_meteo_advect_scale,
                blur_scale=truth_meteo_blur_scale,
                hot_threshold=max(ops.thresholds) if len(ops.thresholds) > 0 else None,
                warm_threshold=ops.thresholds[0] if len(ops.thresholds) > 0 else None,
                preserve_hot_core=bool(truth_cfg.preserve_hot_core_during_warp),
                post_fill_sigma=0.8,
            )
            meta = dict(meta)
            meta["truth_weather_mode"] = "warp"    

    elif truth_model == "puff":
        source_term = SourceTerm() if source_term is None else source_term
        meteo_steps = [MeteoStep(0.0, fcst_cfg.horizon_h if fcst_cfg else 96.0, 250.0, 5.0, "D", 0.0)] \
            if not meteo_steps else list(meteo_steps)

        dummy_fcst = ForecastConfig(enabled=True) if fcst_cfg is None else copy.deepcopy(fcst_cfg)
        dummy_fcst.enabled = True

        forecast_mean_grid, forecast_std_grid, members, meta_fc = simulate_puff_ensemble(
            geom, truth_cfg, source_term, meteo_steps, dummy_fcst, rng
        )

        survey_mask = np.isfinite(forecast_mean_grid)
        valid_idx = []
        for i in range(members.shape[0]):
            inside_vals = np.asarray(members[i], dtype=float)[survey_mask]
            if np.any(np.isfinite(inside_vals)):
                valid_idx.append(i)

        if len(valid_idx) == 0:
            Z_true = np.asarray(forecast_mean_grid, dtype=float).copy()
            truth_idx = -1
        else:
            truth_idx = int(rng.choice(valid_idx))
            Z_true = np.asarray(members[truth_idx], dtype=float).copy()

        meta = dict(meta_fc)
        meta["truth_member_idx"] = truth_idx
        meta["truth_weather_mode"] = "none"

    else:
        raise ValueError(f"Unsupported truth_model: {truth_model}")

    # ----------------------------
    # 2) forecast generation
    # ----------------------------
    if fcst_cfg is None or not fcst_cfg.enabled:
        return Z_true, None, None, meta

    source_term = SourceTerm() if source_term is None else source_term
    meteo_steps = [MeteoStep(0.0, fcst_cfg.horizon_h, 250.0, 5.0, "D", 0.0)] \
        if not meteo_steps else list(meteo_steps)

    forecast_mean_grid, forecast_std_grid, members, meta_fc = simulate_puff_ensemble(
        geom, truth_cfg, source_term, meteo_steps, fcst_cfg, rng
    )

    meta = dict(meta)
    meta["forecast_mean_grid"] = forecast_mean_grid
    meta["forecast_std_grid"] = forecast_std_grid
    meta["forecast_members"] = meta_fc.get("forecast_members", None)
    meta["forecast_valid_members"] = meta_fc.get("forecast_valid_members", None)
    return Z_true, forecast_mean_grid, forecast_std_grid, meta

# ---------------------------------------------------------------------
# GP + MILE acquisition
# ---------------------------------------------------------------------


def fit_gp(X: np.ndarray, y: np.ndarray, cfg: FieldConfig, n_restarts: int) -> GaussianProcessRegressor:
    span = max(cfg.bounds[1] - cfg.bounds[0], cfg.bounds[3] - cfg.bounds[2], 10.0)
    length0 = max(0.14 * span, cfg.measurement_diameter_m * 0.55)
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(
        length_scale=length0,
        length_scale_bounds=(max(0.04 * span, cfg.measurement_diameter_m * 0.4), max(1.2 * span, cfg.measurement_diameter_m * 3.0)),
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=max(cfg.meas_sigma ** 2, 1e-6),
        normalize_y=True,
        n_restarts_optimizer=int(max(n_restarts, 0)),
        random_state=0,
    )
    gp.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float).ravel())
    return gp

def posterior_covariance_matrix(gp: GaussianProcessRegressor, X_train: np.ndarray, X_domain: np.ndarray, noise_var: float) -> np.ndarray:
    K_ss = gp.kernel_(X_domain)
    K_s = gp.kernel_(X_train, X_domain)
    K_xx = gp.kernel_(X_train) + noise_var * np.eye(len(X_train))
    L = np.linalg.cholesky(K_xx + 1e-9 * np.eye(len(X_train)))
    v = np.linalg.solve(L, K_s)
    K_post = K_ss - v.T @ v
    K_post = 0.5 * (K_post + K_post.T)
    diag = np.clip(np.diag(K_post), 0.0, None)
    np.fill_diagonal(K_post, diag)
    return K_post

def posterior_grid_from_current_gp(
    gp: GaussianProcessRegressor,
    geom: SurveyGeometry,
    forecast_mean_grid: Optional[np.ndarray] = None,
    forecast_std_grid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg = geom.cfg
    _, _, X, Y = make_mesh(cfg)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    survey_mask_flat = points_in_polygon(pts, geom.survey_poly, include_boundary=True)

    mu_flat = np.full(len(pts), np.nan, dtype=float)
    std_flat = np.full(len(pts), np.nan, dtype=float)
    qpts = pts[survey_mask_flat]

    if forecast_mean_grid is None:
        mu_pred, std_pred = gp.predict(qpts, return_std=True)
    else:
        forecast_mean_interp = build_interpolator(cfg, forecast_mean_grid)
        forecast_std_interp = build_interpolator(cfg, np.nan_to_num(forecast_std_grid, nan=0.0))

        f_mu = forecast_mean_interp(np.column_stack([qpts[:, 1], qpts[:, 0]])).astype(float)
        f_std = forecast_std_interp(np.column_stack([qpts[:, 1], qpts[:, 0]])).astype(float)

        res_mu, res_std = gp.predict(qpts, return_std=True)
        mu_pred = f_mu + res_mu
        std_pred = np.sqrt(np.maximum(f_std ** 2 + res_std ** 2, 1e-12))

    mu_flat[survey_mask_flat] = np.asarray(mu_pred, dtype=float)
    std_flat[survey_mask_flat] = np.asarray(std_pred, dtype=float)

    survey_mask = survey_mask_flat.reshape(X.shape)
    return mu_flat.reshape(X.shape), std_flat.reshape(X.shape), X, Y, survey_mask

def boundary_stop_status(
    mu_grid: np.ndarray,
    std_grid: np.ndarray,
    X_grid: np.ndarray,
    Y_grid: np.ndarray,
    survey_mask: np.ndarray,
    X_obs: np.ndarray,
    ops: OperationalCriteria,
    adapt: AdaptiveConfig,
    cfg: FieldConfig,
) -> Dict[str, object]:
    std_safe = np.maximum(np.asarray(std_grid, dtype=float), 1e-12)
    frontier = boundary_frontier_score(mu_grid, std_safe, ops.thresholds)
    frontier_mask = survey_mask & np.isfinite(frontier) & (
        frontier >= float(adapt.boundary_stop_frontier_min)
    )

    thr_list = list(ops.thresholds) if bool(adapt.boundary_stop_use_all_thresholds) else [float(ops.thresholds[0])]

    separated = True
    for t in thr_list:
        p = 1.0 - norm.cdf((float(t) - mu_grid) / std_safe)
        hi = survey_mask & (p >= float(adapt.boundary_stop_conf_high))
        lo = survey_mask & (p <= float(adapt.boundary_stop_conf_low))
        if int(np.sum(hi)) < 3 or int(np.sum(lo)) < 3:
            separated = False
            break

    if not np.any(frontier_mask):
        return {
            "ready": False,
            "frontier_mask": frontier_mask,
            "cover_frac": 0.0,
            "frontier_cells": 0,
            "separated": separated,
        }

    r_cover = float(adapt.boundary_stop_cover_radius_m)
    if r_cover <= 0.0:
        r_cover = float(cfg.measurement_diameter_m)

    bpts = np.column_stack([X_grid[frontier_mask], Y_grid[frontier_mask]])
    d = np.linalg.norm(bpts[:, None, :] - np.asarray(X_obs, dtype=float)[None, :, :], axis=2)
    dmin = np.min(d, axis=1)
    covered = dmin <= r_cover

    w = frontier[frontier_mask]
    cover_frac = float(np.sum(w * covered) / max(np.sum(w), 1e-12))

    return {
        "ready": bool(separated and cover_frac >= float(adapt.boundary_stop_cover_frac)),
        "frontier_mask": frontier_mask,
        "cover_frac": cover_frac,
        "frontier_cells": int(np.sum(frontier_mask)),
        "separated": separated,
    }

def min_distance_mask(candidates: np.ndarray, chosen: np.ndarray, min_spacing: float) -> np.ndarray:
    if len(chosen) == 0:
        return np.ones(len(candidates), dtype=bool)
    d = np.linalg.norm(candidates[:, None, :] - chosen[None, :, :], axis=2)
    return np.min(d, axis=1) >= float(min_spacing)


def beta_from_delta(delta: float) -> float:
    return float(norm.ppf(delta))

def exceedance_probability(mu: np.ndarray, std: np.ndarray, threshold: float) -> np.ndarray:
    """
    Posterior exceedance probability:
        P(f(x) > threshold)
    """
    mu = np.asarray(mu, dtype=float)
    std = np.maximum(np.asarray(std, dtype=float), 1e-12)
    thr = float(threshold)
    return 1.0 - norm.cdf((thr - mu) / std)

def threshold_weights(thresholds: Sequence[float]) -> np.ndarray:
    t = np.asarray(thresholds, dtype=float)
    order = np.argsort(t)
    w = np.ones(len(t), dtype=float)
    w[order] = np.arange(1, len(t) + 1, dtype=float)
    return w

def boundary_frontier_score(mu: np.ndarray, std: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    std_eff = np.maximum(np.asarray(std, dtype=float), 1e-12)
    score = np.zeros_like(std_eff, dtype=float)
    for t in thresholds:
        p = 1.0 - norm.cdf((float(t) - np.asarray(mu, dtype=float)) / std_eff)
        score = np.maximum(score, 4.0 * p * (1.0 - p))
    return np.clip(score, 0.0, 1.0)

def robust_scale_from_scores(scores: np.ndarray) -> float:
    finite = np.asarray(scores, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    q25, q75 = np.nanpercentile(finite, [25, 75])
    return float(max(q75 - q25, abs(q75), np.nanstd(finite), 1e-6))

def expected_superlevel_size_after_sample(mu: np.ndarray, sigma2: np.ndarray, K_post: np.ndarray, candidate_idx: int,
                                          thresholds: Sequence[float], weights: np.ndarray, beta: float,
                                          noise_var: float, epsilon: float) -> float:
    cov = K_post[:, candidate_idx]
    denom = float(sigma2[candidate_idx] + noise_var)
    sigma_plus2 = np.clip(sigma2 - (cov ** 2) / max(denom, 1e-12), 0.0, None)
    sigma_plus = np.sqrt(sigma_plus2)
    influence_sd = np.abs(cov) / math.sqrt(max(denom, 1e-12))
    sigma = np.sqrt(np.clip(sigma2, 0.0, None))
    total = 0.0
    for t, w in zip(thresholds, weights):
        current = np.sum((mu - beta * sigma) > (float(t) - float(epsilon)))
        p_after = np.zeros_like(mu, dtype=float)
        nz = influence_sd > 1e-12
        p_after[nz] = norm.cdf((mu[nz] - beta * sigma_plus[nz] - float(t)) / influence_sd[nz])
        p_after[~nz] = ((mu[~nz] - beta * sigma_plus[~nz] - float(t)) > 0.0).astype(float)
        total += float(w) * (float(np.sum(p_after)) - float(current))
    return total

def _hot_threshold(ops: OperationalCriteria) -> float:
    if len(ops.thresholds) == 0:
        raise ValueError("At least one operational threshold is required.")
    return float(np.max(np.asarray(ops.thresholds, dtype=float)))

def choose_next_point_mile_pag(gp: GaussianProcessRegressor, X_obs: np.ndarray, domain_xy: np.ndarray,
                               hot_prior_mask: np.ndarray, ops: OperationalCriteria,
                               cfg: FieldConfig, adapt: AdaptiveConfig,
                               route_rank: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    mu, std = gp.predict(domain_xy, return_std=True)
    sigma2 = np.clip(std ** 2, 0.0, None)
    K_post = posterior_covariance_matrix(gp, X_obs, domain_xy, noise_var=cfg.meas_sigma ** 2)
    beta = beta_from_delta(ops.confidence)
    weights = threshold_weights(ops.thresholds)
    hot_thr = _hot_threshold(ops)

    exploit_term = np.full(len(domain_xy), -np.inf, dtype=float)
    explore_term = np.full(len(domain_xy), -np.inf, dtype=float)
    spacing_mask = min_distance_mask(domain_xy, X_obs, adapt.min_spacing_m)
    for j in range(0, len(domain_xy), max(int(adapt.candidate_stride), 1)):
        if not spacing_mask[j]:
            continue
        imp = expected_superlevel_size_after_sample(mu, sigma2, K_post, int(j), ops.thresholds, weights,
                                                    beta, cfg.meas_sigma ** 2, adapt.epsilon)
        p_exc_j = exceedance_probability(
            np.array([mu[j]]),
            np.array([np.sqrt(sigma2[j])]),
            hot_thr,
        )[0]
        exploit_term[j] = float(imp) * float(np.clip((p_exc_j - 0.20) / 0.80, 0.0, 1.0))
        explore_term[j] = float(adapt.gamma * std[j])
    base_acq = np.maximum(exploit_term, explore_term)
    
    finite = base_acq[np.isfinite(base_acq)]
    if finite.size == 0:
        base_acq = std.copy()
        finite = base_acq[np.isfinite(base_acq)]

    score = base_acq.copy()
    score_scale = robust_scale_from_scores(score)

    if hot_prior_mask is not None and np.any(np.isfinite(score)):
        score = np.where(hot_prior_mask, score + float(adapt.hot_prior_weight) * score_scale, score)

    frontier = boundary_frontier_score(mu, std, ops.thresholds)
    score = np.where(np.isfinite(score), score + float(adapt.frontier_weight) * score_scale * frontier, score)

    if len(X_obs) > 0:
        last = np.asarray(X_obs[-1], dtype=float)
        dist_last = np.linalg.norm(domain_xy - last[None, :], axis=1)

        n_added_so_far = max(len(X_obs) - int(adapt.n0), 0)
        use_local_mode = (int(adapt.global_every) <= 0) or (n_added_so_far % max(int(adapt.global_every), 1) != 0)

        if float(adapt.max_step_m) > 0:
            local_mask = dist_last <= float(adapt.max_step_m)
            feasible_local = local_mask & np.isfinite(score)
            if use_local_mode and np.any(feasible_local):
                score = np.where(local_mask, score, -np.inf)

        ref_dist = max(float(adapt.max_step_m), float(adapt.candidate_spacing_m), 1e-6)
        travel_penalty = (dist_last / ref_dist) ** 1.25
        score = np.where(np.isfinite(score), score - float(adapt.travel_weight) * score_scale * travel_penalty, score)

        if len(X_obs) >= 2 and float(adapt.turn_weight) > 0:
            prev_vec = np.asarray(X_obs[-1], dtype=float) - np.asarray(X_obs[-2], dtype=float)
            prev_norm = float(np.linalg.norm(prev_vec))
            if prev_norm > 1e-9:
                cand_vec = domain_xy - last[None, :]
                cand_norm = np.linalg.norm(cand_vec, axis=1)
                nz = cand_norm > 1e-9
                cosang = np.ones(len(domain_xy), dtype=float)
                cosang[nz] = (cand_vec[nz] @ (prev_vec / prev_norm)) / np.maximum(cand_norm[nz], 1e-9)
                cosang = np.clip(cosang, -1.0, 1.0)
                turn_penalty = 0.5 * (1.0 - cosang)
                score = np.where(np.isfinite(score), score - float(adapt.turn_weight) * score_scale * turn_penalty, score)

    if route_rank is not None and len(domain_xy) > 0 and len(X_obs) > 0:
        route_rank = np.asarray(route_rank, dtype=int)
        obs_idx = []
        for p in np.asarray(X_obs, dtype=float):
            d = np.linalg.norm(domain_xy - p[None, :], axis=1)
            obs_idx.append(int(np.argmin(d)))
        visited = np.zeros(len(domain_xy), dtype=bool)
        visited[np.asarray(obs_idx, dtype=int)] = True
        last = np.asarray(X_obs[-1], dtype=float)
        last_idx = int(np.argmin(np.linalg.norm(domain_xy - last[None, :], axis=1)))
        current_rank = int(route_rank[last_idx])
        gaps = route_rank - current_rank
        ahead = (gaps > 0) & (~visited)
        score = np.where(ahead, score, -np.inf)
        if int(adapt.vsp_lookahead) > 0:
            near_ahead = ahead & (gaps <= int(adapt.vsp_lookahead))
            if np.any(near_ahead & np.isfinite(score)):
                score = np.where(near_ahead, score, -np.inf)
        skip_penalty = np.clip(gaps - 1, 0.0, None)
        score = np.where(np.isfinite(score), score - float(adapt.vsp_skip_weight) * score_scale * skip_penalty, score)

    idx = int(np.nanargmax(score))
    return domain_xy[idx], score

def choose_next_point_mile_pag_debug(
    gp: GaussianProcessRegressor,
    X_obs: np.ndarray,
    domain_xy: np.ndarray,
    hot_prior_mask: np.ndarray,
    ops: OperationalCriteria,
    cfg: FieldConfig,
    adapt: AdaptiveConfig,
    route_rank: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    mu, std = gp.predict(domain_xy, return_std=True)
    sigma2 = np.clip(std ** 2, 0.0, None)
    K_post = posterior_covariance_matrix(gp, X_obs, domain_xy, noise_var=cfg.meas_sigma ** 2)
    beta = beta_from_delta(ops.confidence)
    weights = threshold_weights(ops.thresholds)
    hot_thr = _hot_threshold(ops)

    exploit_term = np.full(len(domain_xy), -np.inf, dtype=float)
    explore_term = np.full(len(domain_xy), -np.inf, dtype=float)
    spacing_mask = min_distance_mask(domain_xy, X_obs, adapt.min_spacing_m)

    for j in range(0, len(domain_xy), max(int(adapt.candidate_stride), 1)):
        if not spacing_mask[j]:
            continue
        imp = expected_superlevel_size_after_sample(
            mu, sigma2, K_post, int(j),
            ops.thresholds, weights,
            beta, cfg.meas_sigma ** 2, adapt.epsilon
        )
        p_exc_j = exceedance_probability(
            np.array([mu[j]]),
            np.array([np.sqrt(sigma2[j])]),
            hot_thr,
        )[0]
        exploit_term[j] = float(imp) * float(np.clip((p_exc_j - 0.20) / 0.80, 0.0, 1.0))
        explore_term[j] = float(adapt.gamma * std[j])

    base_acq = np.maximum(exploit_term, explore_term)
    finite = base_acq[np.isfinite(base_acq)]
    if finite.size == 0:
        base_acq = std.copy()
        explore_term = std.copy()
        exploit_term = np.zeros_like(std)

    score = base_acq.copy()
    score_scale = robust_scale_from_scores(score)

    if hot_prior_mask is not None and np.any(np.isfinite(score)):
        score = np.where(hot_prior_mask, score + float(adapt.hot_prior_weight) * score_scale, score)

    frontier = boundary_frontier_score(mu, std, ops.thresholds)
    score = np.where(np.isfinite(score), score + float(adapt.frontier_weight) * score_scale * frontier, score)

    if len(X_obs) > 0:
        last = np.asarray(X_obs[-1], dtype=float)
        dist_last = np.linalg.norm(domain_xy - last[None, :], axis=1)

        n_added_so_far = max(len(X_obs) - int(adapt.n0), 0)
        use_local_mode = (int(adapt.global_every) <= 0) or (n_added_so_far % max(int(adapt.global_every), 1) != 0)

        if float(adapt.max_step_m) > 0:
            local_mask = dist_last <= float(adapt.max_step_m)
            feasible_local = local_mask & np.isfinite(score)
            if use_local_mode and np.any(feasible_local):
                score = np.where(local_mask, score, -np.inf)

        ref_dist = max(float(adapt.max_step_m), float(adapt.candidate_spacing_m), 1e-6)
        travel_penalty = (dist_last / ref_dist) ** 1.25
        score = np.where(np.isfinite(score), score - float(adapt.travel_weight) * score_scale * travel_penalty, score)

        if len(X_obs) >= 2 and float(adapt.turn_weight) > 0:
            prev_vec = np.asarray(X_obs[-1], dtype=float) - np.asarray(X_obs[-2], dtype=float)
            prev_norm = float(np.linalg.norm(prev_vec))
            if prev_norm > 1e-9:
                cand_vec = domain_xy - last[None, :]
                cand_norm = np.linalg.norm(cand_vec, axis=1)
                nz = cand_norm > 1e-9
                cosang = np.ones(len(domain_xy), dtype=float)
                cosang[nz] = (cand_vec[nz] @ (prev_vec / prev_norm)) / np.maximum(cand_norm[nz], 1e-9)
                cosang = np.clip(cosang, -1.0, 1.0)
                turn_penalty = 0.5 * (1.0 - cosang)
                score = np.where(np.isfinite(score), score - float(adapt.turn_weight) * score_scale * turn_penalty, score)

    if route_rank is not None and len(domain_xy) > 0 and len(X_obs) > 0:
        route_rank = np.asarray(route_rank, dtype=int)
        obs_idx = []
        for p in np.asarray(X_obs, dtype=float):
            d = np.linalg.norm(domain_xy - p[None, :], axis=1)
            obs_idx.append(int(np.argmin(d)))
        visited = np.zeros(len(domain_xy), dtype=bool)
        visited[np.asarray(obs_idx, dtype=int)] = True
        last = np.asarray(X_obs[-1], dtype=float)
        last_idx = int(np.argmin(np.linalg.norm(domain_xy - last[None, :], axis=1)))
        current_rank = int(route_rank[last_idx])
        gaps = route_rank - current_rank
        ahead = (gaps > 0) & (~visited)
        score = np.where(ahead, score, -np.inf)
        if int(adapt.vsp_lookahead) > 0:
            near_ahead = ahead & (gaps <= int(adapt.vsp_lookahead))
            if np.any(near_ahead & np.isfinite(score)):
                score = np.where(near_ahead, score, -np.inf)
        skip_penalty = np.clip(gaps - 1, 0.0, None)
        score = np.where(np.isfinite(score), score - float(adapt.vsp_skip_weight) * score_scale * skip_penalty, score)

    idx = int(np.nanargmax(score))
    chosen_role = "explore" if float(explore_term[idx]) >= float(exploit_term[idx]) else "exploit"
    p_exc = exceedance_probability(mu, np.sqrt(sigma2), hot_thr)
    chosen_role = _role_from_terms(explore_term, exploit_term, idx)
    if chosen_role == "exploit" and float(p_exc[idx]) < 0.35:
        chosen_role = "explore"

    dbg = {
        "mu_pred": mu,
        "std_pred": np.sqrt(sigma2), 
        "explore_term": explore_term,
        "exploit_term": exploit_term,
        "frontier": frontier,
        "score": score,
        "chosen_role": chosen_role,
    }
    return domain_xy[idx], score, dbg

def classify_pag_truth(Z: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    zone = np.zeros_like(Z, dtype=int)
    for i, t in enumerate(sorted(thresholds), start=1):
        zone = np.where(Z >= float(t), i, zone)
    zone[np.isnan(Z)] = -1
    return zone

def expected_superlevel_size_after_sample_total(
    mu: np.ndarray,
    sigma2_total: np.ndarray,
    sigma2_assim: np.ndarray,
    K_post_assim: np.ndarray,
    candidate_idx: int,
    thresholds: Sequence[float],
    weights: np.ndarray,
    beta: float,
    noise_var: float,
    epsilon: float,
) -> float:
    cov = K_post_assim[:, candidate_idx]
    denom = float(sigma2_assim[candidate_idx] + noise_var)

    sigma_plus2 = np.clip(sigma2_total - (cov ** 2) / max(denom, 1e-12), 0.0, None)
    sigma_plus = np.sqrt(sigma_plus2)
    sigma_now = np.sqrt(np.clip(sigma2_total, 0.0, None))

    influence_sd = np.abs(cov) / math.sqrt(max(denom, 1e-12))
    total = 0.0

    for t, w in zip(thresholds, weights):
        current = np.sum((mu - beta * sigma_now) > (float(t) - float(epsilon)))

        p_after = np.zeros_like(mu, dtype=float)
        nz = influence_sd > 1e-12
        p_after[nz] = norm.cdf((mu[nz] - beta * sigma_plus[nz] - float(t)) / influence_sd[nz])
        p_after[~nz] = ((mu[~nz] - beta * sigma_plus[~nz] - float(t)) > 0.0).astype(float)

        total += float(w) * (float(np.sum(p_after)) - float(current))

    return total

def choose_next_point_mile_pag_residual(
    gp: GaussianProcessRegressor,
    X_obs: np.ndarray,
    domain_xy: np.ndarray,
    hot_prior_mask: np.ndarray,
    ops: OperationalCriteria,
    cfg: FieldConfig,
    adapt: AdaptiveConfig,
    forecast_mean_interp: RegularGridInterpolator,
    forecast_std_interp: RegularGridInterpolator,
    route_rank: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    mu_res, std_res = gp.predict(domain_xy, return_std=True)
    sigma2_res = np.clip(std_res ** 2, 0.0, None)

    f_mean = forecast_mean_interp(np.column_stack([domain_xy[:, 1], domain_xy[:, 0]])).astype(float)
    f_std = forecast_std_interp(np.column_stack([domain_xy[:, 1], domain_xy[:, 0]])).astype(float)
    f_std = np.clip(f_std, 0.0, None)

    mu = f_mean + mu_res
    sigma2_total = np.clip(f_std ** 2 + sigma2_res, 0.0, None)
    std_total = np.sqrt(sigma2_total)

    K_post = posterior_covariance_matrix(gp, X_obs, domain_xy, noise_var=cfg.meas_sigma ** 2)
    beta = beta_from_delta(ops.confidence)
    weights = threshold_weights(ops.thresholds)
    hot_thr = _hot_threshold(ops)

    exploit_term = np.full(len(domain_xy), -np.inf, dtype=float)
    explore_term = np.full(len(domain_xy), -np.inf, dtype=float)
    spacing_mask = min_distance_mask(domain_xy, X_obs, adapt.min_spacing_m)

    for j in range(0, len(domain_xy), max(int(adapt.candidate_stride), 1)):
        if not spacing_mask[j]:
            continue
        imp = expected_superlevel_size_after_sample_total(
            mu=mu,
            sigma2_total=sigma2_total,
            sigma2_assim=sigma2_res,
            K_post_assim=K_post,
            candidate_idx=int(j),
            thresholds=ops.thresholds,
            weights=weights,
            beta=beta,
            noise_var=cfg.meas_sigma ** 2,
            epsilon=adapt.epsilon,
        )
        p_exc_j = exceedance_probability(
            np.array([mu[j]]),
            np.array([std_total[j]]),
            hot_thr,
        )[0]
        exploit_term[j] = float(imp) * float(np.clip((p_exc_j - 0.20) / 0.80, 0.0, 1.0))
        explore_term[j] = float(adapt.gamma * std_total[j])

    base_acq = np.maximum(exploit_term, explore_term)

    finite = base_acq[np.isfinite(base_acq)]
    if finite.size == 0:
        base_acq = std_total.copy()

    score = base_acq.copy()
    score_scale = robust_scale_from_scores(score)

    if hot_prior_mask is not None and np.any(np.isfinite(score)):
        score = np.where(hot_prior_mask, score + float(adapt.hot_prior_weight) * score_scale, score)

    frontier = boundary_frontier_score(mu, std_total, ops.thresholds)
    score = np.where(np.isfinite(score), score + float(adapt.frontier_weight) * score_scale * frontier, score)

    if len(X_obs) > 0:
        last = np.asarray(X_obs[-1], dtype=float)
        dist_last = np.linalg.norm(domain_xy - last[None, :], axis=1)

        n_added_so_far = max(len(X_obs) - int(adapt.n0), 0)
        use_local_mode = (int(adapt.global_every) <= 0) or (n_added_so_far % max(int(adapt.global_every), 1) != 0)

        if float(adapt.max_step_m) > 0:
            local_mask = dist_last <= float(adapt.max_step_m)
            feasible_local = local_mask & np.isfinite(score)
            if use_local_mode and np.any(feasible_local):
                score = np.where(local_mask, score, -np.inf)

        ref_dist = max(float(adapt.max_step_m), float(adapt.candidate_spacing_m), 1e-6)
        travel_penalty = (dist_last / ref_dist) ** 1.25
        score = np.where(np.isfinite(score), score - float(adapt.travel_weight) * score_scale * travel_penalty, score)

        if len(X_obs) >= 2 and float(adapt.turn_weight) > 0:
            prev_vec = np.asarray(X_obs[-1], dtype=float) - np.asarray(X_obs[-2], dtype=float)
            prev_norm = float(np.linalg.norm(prev_vec))
            if prev_norm > 1e-9:
                cand_vec = domain_xy - last[None, :]
                cand_norm = np.linalg.norm(cand_vec, axis=1)
                nz = cand_norm > 1e-9
                cosang = np.ones(len(domain_xy), dtype=float)
                cosang[nz] = (cand_vec[nz] @ (prev_vec / prev_norm)) / np.maximum(cand_norm[nz], 1e-9)
                cosang = np.clip(cosang, -1.0, 1.0)
                turn_penalty = 0.5 * (1.0 - cosang)
                score = np.where(np.isfinite(score), score - float(adapt.turn_weight) * score_scale * turn_penalty, score)

    if route_rank is not None and len(domain_xy) > 0 and len(X_obs) > 0:
        route_rank = np.asarray(route_rank, dtype=int)
        obs_idx = []
        for p in np.asarray(X_obs, dtype=float):
            d = np.linalg.norm(domain_xy - p[None, :], axis=1)
            obs_idx.append(int(np.argmin(d)))
        visited = np.zeros(len(domain_xy), dtype=bool)
        visited[np.asarray(obs_idx, dtype=int)] = True
        last = np.asarray(X_obs[-1], dtype=float)
        last_idx = int(np.argmin(np.linalg.norm(domain_xy - last[None, :], axis=1)))
        current_rank = int(route_rank[last_idx])
        gaps = route_rank - current_rank
        ahead = (gaps > 0) & (~visited)
        score = np.where(ahead, score, -np.inf)
        if int(adapt.vsp_lookahead) > 0:
            near_ahead = ahead & (gaps <= int(adapt.vsp_lookahead))
            if np.any(near_ahead & np.isfinite(score)):
                score = np.where(near_ahead, score, -np.inf)
        skip_penalty = np.clip(gaps - 1, 0.0, None)
        score = np.where(np.isfinite(score), score - float(adapt.vsp_skip_weight) * score_scale * skip_penalty, score)

    idx = int(np.nanargmax(score))
    return domain_xy[idx], score

def _role_from_terms(explore_term: np.ndarray, exploit_term: np.ndarray, idx: int) -> str:
    ex = float(explore_term[idx]) if np.isfinite(explore_term[idx]) else -np.inf
    ep = float(exploit_term[idx]) if np.isfinite(exploit_term[idx]) else -np.inf
    if (not np.isfinite(ex)) and (not np.isfinite(ep)):
        return "seed"
    return "explore" if ex >= ep else "exploit"

def _role_for_selected_point(
    x_sel: np.ndarray,
    domain_xy: np.ndarray,
    explore_term: np.ndarray,
    exploit_term: np.ndarray,
) -> str:
    d = np.linalg.norm(domain_xy - np.asarray(x_sel, dtype=float)[None, :], axis=1)
    idx = int(np.argmin(d))
    return _role_from_terms(explore_term, exploit_term, idx)

def _domain_index_for_point(x_sel: np.ndarray, domain_xy: np.ndarray) -> int:
    d = np.linalg.norm(domain_xy - np.asarray(x_sel, dtype=float)[None, :], axis=1)
    return int(np.argmin(d))

def _role_for_oracle_onestep(
    x_oracle: np.ndarray,
    x_post: np.ndarray,
    domain_xy: np.ndarray,
    dbg: Dict[str, object],
    ops: OperationalCriteria,
    dist_gate_m: float = 50.0,
) -> str:
    oracle_idx = _domain_index_for_point(x_oracle, domain_xy)
    post_idx = _domain_index_for_point(x_post, domain_xy)
    post_role = str(dbg.get("chosen_role", "exploit"))

    if oracle_idx == post_idx and post_role == "explore":
        return "explore"

    mu_pred = np.asarray(dbg.get("mu_pred", []), dtype=float)
    std_pred = np.asarray(dbg.get("std_pred", []), dtype=float)

    if mu_pred.size == len(domain_xy) and std_pred.size == len(domain_xy):
        zone_pred = classify_pag_from_posterior(
            mu_pred,
            std_pred,
            ops.thresholds,
            ops.confidence,
        )
        warmhot_mask = zone_pred >= 1 

        if np.any(warmhot_mask):
            dmin = float(np.min(np.linalg.norm(
                domain_xy[warmhot_mask] - np.asarray(x_oracle, dtype=float)[None, :],
                axis=1,
            )))
            if dmin > float(dist_gate_m):
                return "explore"

    return "exploit"

def choose_next_point_mile_pag_residual_debug(
    gp: GaussianProcessRegressor,
    X_obs: np.ndarray,
    domain_xy: np.ndarray,
    hot_prior_mask: np.ndarray,
    ops: OperationalCriteria,
    cfg: FieldConfig,
    adapt: AdaptiveConfig,
    forecast_mean_interp: RegularGridInterpolator,
    forecast_std_interp: RegularGridInterpolator,
    route_rank: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    mu_res, std_res = gp.predict(domain_xy, return_std=True)
    sigma2_res = np.clip(std_res ** 2, 0.0, None)

    f_mean = forecast_mean_interp(np.column_stack([domain_xy[:, 1], domain_xy[:, 0]])).astype(float)
    f_std = forecast_std_interp(np.column_stack([domain_xy[:, 1], domain_xy[:, 0]])).astype(float)
    f_std = np.clip(f_std, 0.0, None)

    mu = f_mean + mu_res
    sigma2_total = np.clip(f_std ** 2 + sigma2_res, 0.0, None)
    std_total = np.sqrt(sigma2_total)

    K_post = posterior_covariance_matrix(gp, X_obs, domain_xy, noise_var=cfg.meas_sigma ** 2)
    beta = beta_from_delta(ops.confidence)
    weights = threshold_weights(ops.thresholds)
    hot_thr = _hot_threshold(ops)

    exploit_term = np.full(len(domain_xy), -np.inf, dtype=float)
    explore_term = np.full(len(domain_xy), -np.inf, dtype=float)
    spacing_mask = min_distance_mask(domain_xy, X_obs, adapt.min_spacing_m)

    for j in range(0, len(domain_xy), max(int(adapt.candidate_stride), 1)):
        if not spacing_mask[j]:
            continue
        imp = expected_superlevel_size_after_sample_total(
            mu=mu,
            sigma2_total=sigma2_total,
            sigma2_assim=sigma2_res,
            K_post_assim=K_post,
            candidate_idx=int(j),
            thresholds=ops.thresholds,
            weights=weights,
            beta=beta,
            noise_var=cfg.meas_sigma ** 2,
            epsilon=adapt.epsilon,
        )
        p_exc_j = exceedance_probability(
            np.array([mu[j]]),
            np.array([std_total[j]]),
            hot_thr,
        )[0]
        exploit_term[j] = float(imp) * float(np.clip((p_exc_j - 0.20) / 0.80, 0.0, 1.0))
        explore_term[j] = float(adapt.gamma * std_total[j])

    base_acq = np.maximum(exploit_term, explore_term)
    finite = base_acq[np.isfinite(base_acq)]
    if finite.size == 0:
        base_acq = std_total.copy()
        explore_term = std_total.copy()
        exploit_term = np.zeros_like(std_total)

    score = base_acq.copy()
    score_scale = robust_scale_from_scores(score)

    if hot_prior_mask is not None and np.any(np.isfinite(score)):
        score = np.where(hot_prior_mask, score + float(adapt.hot_prior_weight) * score_scale, score)

    frontier = boundary_frontier_score(mu, std_total, ops.thresholds)
    score = np.where(np.isfinite(score), score + float(adapt.frontier_weight) * score_scale * frontier, score)

    if len(X_obs) > 0:
        last = np.asarray(X_obs[-1], dtype=float)
        dist_last = np.linalg.norm(domain_xy - last[None, :], axis=1)

        n_added_so_far = max(len(X_obs) - int(adapt.n0), 0)
        use_local_mode = (int(adapt.global_every) <= 0) or (n_added_so_far % max(int(adapt.global_every), 1) != 0)

        if float(adapt.max_step_m) > 0:
            local_mask = dist_last <= float(adapt.max_step_m)
            feasible_local = local_mask & np.isfinite(score)
            if use_local_mode and np.any(feasible_local):
                score = np.where(local_mask, score, -np.inf)

        ref_dist = max(float(adapt.max_step_m), float(adapt.candidate_spacing_m), 1e-6)
        travel_penalty = (dist_last / ref_dist) ** 1.25
        score = np.where(np.isfinite(score), score - float(adapt.travel_weight) * score_scale * travel_penalty, score)

        if len(X_obs) >= 2 and float(adapt.turn_weight) > 0:
            prev_vec = np.asarray(X_obs[-1], dtype=float) - np.asarray(X_obs[-2], dtype=float)
            prev_norm = float(np.linalg.norm(prev_vec))
            if prev_norm > 1e-9:
                cand_vec = domain_xy - last[None, :]
                cand_norm = np.linalg.norm(cand_vec, axis=1)
                nz = cand_norm > 1e-9
                cosang = np.ones(len(domain_xy), dtype=float)
                cosang[nz] = (cand_vec[nz] @ (prev_vec / prev_norm)) / np.maximum(cand_norm[nz], 1e-9)
                cosang = np.clip(cosang, -1.0, 1.0)
                turn_penalty = 0.5 * (1.0 - cosang)
                score = np.where(np.isfinite(score), score - float(adapt.turn_weight) * score_scale * turn_penalty, score)

    if route_rank is not None and len(domain_xy) > 0 and len(X_obs) > 0:
        route_rank = np.asarray(route_rank, dtype=int)
        obs_idx = []
        for p in np.asarray(X_obs, dtype=float):
            d = np.linalg.norm(domain_xy - p[None, :], axis=1)
            obs_idx.append(int(np.argmin(d)))
        visited = np.zeros(len(domain_xy), dtype=bool)
        visited[np.asarray(obs_idx, dtype=int)] = True
        last = np.asarray(X_obs[-1], dtype=float)
        last_idx = int(np.argmin(np.linalg.norm(domain_xy - last[None, :], axis=1)))
        current_rank = int(route_rank[last_idx])
        gaps = route_rank - current_rank
        ahead = (gaps > 0) & (~visited)
        score = np.where(ahead, score, -np.inf)
        if int(adapt.vsp_lookahead) > 0:
            near_ahead = ahead & (gaps <= int(adapt.vsp_lookahead))
            if np.any(near_ahead & np.isfinite(score)):
                score = np.where(near_ahead, score, -np.inf)
        skip_penalty = np.clip(gaps - 1, 0.0, None)
        score = np.where(np.isfinite(score), score - float(adapt.vsp_skip_weight) * score_scale * skip_penalty, score)

    idx = int(np.nanargmax(score))
    p_exc = exceedance_probability(mu, std_total, hot_thr)
    chosen_role = _role_from_terms(explore_term, exploit_term, idx)
    if chosen_role == "exploit" and float(p_exc[idx]) < 0.35:
        chosen_role = "explore"

    dbg = {
        "mu_pred": mu,
        "std_pred": std_total,
        "explore_term": explore_term,
        "exploit_term": exploit_term,
        "frontier": frontier,
        "score": score,
        "chosen_role": chosen_role,
    }
    return domain_xy[idx], score, dbg

def choose_next_point_oracle_onestep(
    geom: SurveyGeometry,
    ops: OperationalCriteria,
    adapt: AdaptiveConfig,
    Z_true: np.ndarray,
    rng: np.random.Generator,
    X_obs: np.ndarray,
    y_obs: np.ndarray,
    domain_xy: np.ndarray,
    hot_mask: np.ndarray,
    route_rank: Optional[np.ndarray],
    timing_cfg: TimingConfig,
    weight_cfg: DecisionWeightConfig,
    forecast_mean_grid: Optional[np.ndarray] = None,
    forecast_std_grid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    cfg = geom.cfg
    truth_interp = build_interpolator(cfg, build_measurement_field(cfg, Z_true))

    if forecast_mean_grid is None:
        gp = fit_gp(X_obs, y_obs, cfg, adapt.gp_restarts)
        _, score = choose_next_point_mile_pag(
            gp, X_obs, domain_xy, hot_mask, ops, cfg, adapt, route_rank=route_rank
        )
    else:
        forecast_mean_interp = build_interpolator(cfg, forecast_mean_grid)
        forecast_std_interp = build_interpolator(cfg, np.nan_to_num(forecast_std_grid, nan=0.0))
        forecast_meas_grid = build_measurement_field(cfg, forecast_mean_grid)
        forecast_meas_interp = build_interpolator(cfg, forecast_meas_grid)

        f_obs = forecast_meas_interp(np.column_stack([X_obs[:, 1], X_obs[:, 0]])).astype(float)
        residual_obs = y_obs - f_obs
        gp = fit_gp(X_obs, residual_obs, cfg, adapt.gp_restarts)

        _, score = choose_next_point_mile_pag_residual(
            gp=gp,
            X_obs=X_obs,
            domain_xy=domain_xy,
            hot_prior_mask=hot_mask,
            ops=ops,
            cfg=cfg,
            adapt=adapt,
            forecast_mean_interp=forecast_mean_interp,
            forecast_std_interp=forecast_std_interp,
            route_rank=route_rank,
        )

    finite = np.where(np.isfinite(score))[0]
    if len(finite) == 0:
        raise ValueError("No finite candidates available for oracle one-step selection.")

    top_k = int(min(max(adapt.oracle_top_k, 1), len(finite)))
    top_idx = finite[np.argsort(score[finite])[-top_k:]]
    cand_top = domain_xy[top_idx]

    best_x = None
    best_q = np.inf

    for x in cand_top:
        q_vals = []

        for _ in range(int(max(adapt.oracle_n_repl, 1))):
            z_new = sample_field(cfg, truth_interp, x[None, :], rng)[0]

            X2 = np.vstack([X_obs, x[None, :]])
            y2 = np.append(y_obs, z_new)

            res2 = evaluate_design(
                name="oracle_tmp",
                geom=geom,
                ops=ops,
                Z_true=Z_true,
                X_obs=X2,
                y_obs=y2,
                n_restarts=adapt.gp_restarts,
                forecast_mean_grid=forecast_mean_grid,
                forecast_std_grid=forecast_std_grid,
                timing_cfg=timing_cfg,
            )

            loss2 = true_weighted_loss_from_result(res2, ops, weight_cfg)["J_true"]
            q_vals.append(loss2)

        q_mean = float(np.nanmean(q_vals))

        if q_mean < best_q:
            best_q = q_mean
            best_x = np.asarray(x, dtype=float)

    if best_x is None:
        raise ValueError("Oracle one-step failed to select a candidate.")

    return best_x, {
        "best_q": float(best_q),
        "top_k": float(top_k),
    }

def classify_pag_from_posterior(mu: np.ndarray, std: np.ndarray, thresholds: Sequence[float], delta: float) -> np.ndarray:
    zone = np.zeros_like(mu, dtype=int)
    for i, t in enumerate(sorted(thresholds), start=1):
        p = 1.0 - norm.cdf((float(t) - mu) / np.maximum(std, 1e-12))
        zone = np.where(p > float(delta), i, zone)
    zone[np.isnan(mu)] = -1
    return zone

def metrics_from_zone_maps(pred_zone: np.ndarray, truth_zone: np.ndarray, hot_idx: int) -> Dict[str, float]:
    valid = truth_zone >= 0
    zone_accuracy = float(np.mean(pred_zone[valid] == truth_zone[valid])) if np.any(valid) else float("nan")

    hot_truth = truth_zone == int(hot_idx)
    hot_pred = pred_zone == int(hot_idx)
    n_hot_truth = int(np.sum(hot_truth))

    if n_hot_truth == 0:
        hot_recall = float("nan")
    else:
        hot_recall = float(np.sum(hot_truth & hot_pred) / n_hot_truth)

    action_truth = truth_zone >= 1
    action_pred = pred_zone >= 1
    n_action_truth = int(np.sum(action_truth))
    n_action_pred = int(np.sum(action_pred))

    action_recall = float(np.sum(action_truth & action_pred) / max(n_action_truth, 1))
    action_precision = float(np.sum(action_truth & action_pred) / max(n_action_pred, 1))

    return {
        "zone_accuracy": zone_accuracy,
        "hot_recall": hot_recall,
        "action_recall": action_recall,
        "action_precision": action_precision,
        "truth_has_hot": bool(n_hot_truth > 0),
        "pred_has_hot": bool(np.any(hot_pred)),
    }

def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    b = np.zeros_like(m, dtype=bool)

    b[:-1, :] |= (m[:-1, :] != m[1:, :])
    b[1:,  :] |= (m[1:,  :] != m[:-1, :])
    b[:, :-1] |= (m[:, :-1] != m[:, 1:])
    b[:, 1:]  |= (m[:, 1:]  != m[:, :-1])

    return b

def boundary_metrics_from_result(
    res: RunResult,
    ops: OperationalCriteria,
    target: str = "warm",
    dilate_cells: int = 1,
) -> Dict[str, float]:
    valid = res.truth_zone >= 0
    target = str(target).strip().lower()

    if target == "hot":
        hot_idx = top_zone_index(ops)
        truth_region = (res.truth_zone == hot_idx) & valid
        pred_region  = (res.pred_zone  == hot_idx) & valid
    else:
        # outer action/warm boundary
        truth_region = (res.truth_zone >= 1) & valid
        pred_region  = (res.pred_zone  >= 1) & valid

    b_true = _binary_boundary(truth_region)
    b_pred = _binary_boundary(pred_region)

    n_true = int(np.sum(b_true))
    n_pred = int(np.sum(b_pred))

    if n_true == 0:
        return {
            "boundary_recall": float("nan"),
            "boundary_precision": float("nan"),
            "boundary_f1": float("nan"),
            "truth_boundary_cells": 0.0,
            "pred_boundary_cells": float(n_pred),
        }

    if int(dilate_cells) > 0:
        b_true_d = binary_dilation(b_true, iterations=int(dilate_cells))
        b_pred_d = binary_dilation(b_pred, iterations=int(dilate_cells))
    else:
        b_true_d = b_true
        b_pred_d = b_pred

    recall = float(np.sum(b_pred_d & b_true) / max(n_true, 1))
    precision = float(np.sum(b_true_d & b_pred) / max(n_pred, 1)) if n_pred > 0 else 0.0

    if (recall + precision) > 0:
        f1 = float(2.0 * recall * precision / (recall + precision))
    else:
        f1 = 0.0

    return {
        "boundary_recall": recall,
        "boundary_precision": precision,
        "boundary_f1": f1,
        "truth_boundary_cells": float(n_true),
        "pred_boundary_cells": float(n_pred),
    }

def evaluate_design(
    name: str,
    geom: SurveyGeometry,
    ops: OperationalCriteria,
    Z_true: np.ndarray,
    X_obs: np.ndarray,
    y_obs: np.ndarray,
    n_restarts: int,
    forecast_mean_grid: Optional[np.ndarray] = None,
    forecast_std_grid: Optional[np.ndarray] = None,
    timing_cfg: Optional[TimingConfig] = None,
) -> RunResult:
    cfg = geom.cfg
    _, _, X, Y = make_mesh(cfg)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    survey_mask = points_in_polygon(pts, geom.survey_poly, include_boundary=True)

    mu_flat = np.full(len(pts), np.nan)
    std_flat = np.full(len(pts), np.nan)

    if forecast_mean_grid is None:
        gp = fit_gp(X_obs, y_obs, cfg, n_restarts)
        mu_pred, std_pred = gp.predict(pts[survey_mask], return_std=True)
    else:
        forecast_mean_interp = build_interpolator(cfg, forecast_mean_grid)
        forecast_std_interp = build_interpolator(cfg, np.nan_to_num(forecast_std_grid, nan=0.0))

        forecast_meas_grid = build_measurement_field(cfg, forecast_mean_grid)
        forecast_meas_interp = build_interpolator(cfg, forecast_meas_grid)

        f_obs = forecast_meas_interp(np.column_stack([X_obs[:, 1], X_obs[:, 0]])).astype(float)
        residual_obs = np.asarray(y_obs, dtype=float) - f_obs

        gp = fit_gp(X_obs, residual_obs, cfg, n_restarts)

        res_mu, res_std = gp.predict(pts[survey_mask], return_std=True)
        f_mu = forecast_mean_interp(np.column_stack([pts[survey_mask, 1], pts[survey_mask, 0]])).astype(float)
        f_std = forecast_std_interp(np.column_stack([pts[survey_mask, 1], pts[survey_mask, 0]])).astype(float)

        mu_pred = f_mu + res_mu
        std_pred = np.sqrt(np.maximum(f_std ** 2 + res_std ** 2, 1e-12))

    mu_flat[survey_mask] = mu_pred
    std_flat[survey_mask] = std_pred

    mu_grid = mu_flat.reshape(X.shape)
    std_grid = std_flat.reshape(X.shape)

    pred_zone = classify_pag_from_posterior(mu_grid, std_grid, ops.thresholds, ops.confidence)
    truth_zone = classify_pag_truth(Z_true, ops.thresholds)
    m = metrics_from_zone_maps(pred_zone, truth_zone, hot_idx=top_zone_index(ops))

    path_len = path_length(X_obs)
    timing_cfg = TimingConfig() if timing_cfg is None else timing_cfg
    tm = compute_time_metrics(path_len, len(X_obs), timing_cfg)

    return RunResult(
        name=name,
        X_obs=np.asarray(X_obs, float),
        y_obs=np.asarray(y_obs, float),
        mu_grid=mu_grid,
        std_grid=std_grid,
        pred_zone=pred_zone,
        truth_zone=truth_zone,
        zone_accuracy=m["zone_accuracy"],
        hot_recall=m["hot_recall"],
        action_recall=m["action_recall"],
        action_precision=m["action_precision"],
        n_used=len(X_obs),
        path_length_m=path_len,
        travel_time_min=tm["travel_time_min"],
        station_time_total_min=tm["station_time_total_min"],
        total_time_min=tm["total_time_min"],
    )

def run_baseline(
    geom: SurveyGeometry,
    ops: OperationalCriteria,
    base_cfg: BaselineConfig,
    Z_true: np.ndarray,
    rng: np.random.Generator,
    vsp_xy: Optional[np.ndarray],
    forecast_mean_grid: Optional[np.ndarray] = None,
    forecast_std_grid: Optional[np.ndarray] = None,
    timing_cfg: Optional[TimingConfig] = None,
) -> RunResult:
    cfg = geom.cfg

    if isinstance(vsp_xy, pd.DataFrame) and baseline_csv_supports_phase_guided_search(vsp_xy):
        xy, y, diag = simulate_phase_guided_baseline_sampling(
            frame=vsp_xy,
            cfg=cfg,
            Z_true=Z_true,
            rng=rng,
            ops=ops,
            max_samples=int(getattr(base_cfg, "max_samples", 0)),
            corridor_half_width_m=0.5 * float(cfg.measurement_diameter_m),
            use_observed_value_for_control=False,
            origin=(0.0, 0.0),
        )
        res = evaluate_design(
            "baseline", geom, ops, Z_true, xy, y,
            n_restarts=1,
            forecast_mean_grid=forecast_mean_grid,
            forecast_std_grid=forecast_std_grid,
            timing_cfg=timing_cfg,
        )
        res.diag = {} if res.diag is None else dict(res.diag)
        res.diag.update(diag)
        return res

    xy = build_baseline_points(geom, base_cfg, vsp_xy)
    Z_meas = build_measurement_field(cfg, Z_true)
    interp = build_interpolator(cfg, Z_meas)
    y = sample_field(cfg, interp, xy, rng)

    return evaluate_design(
        "baseline", geom, ops, Z_true, xy, y,
        n_restarts=1,
        forecast_mean_grid=forecast_mean_grid,
        forecast_std_grid=forecast_std_grid,
        timing_cfg=timing_cfg,
    )

def initial_adaptive_points(
    domain_xy: np.ndarray,
    hot_mask: np.ndarray,
    n0: int,
    order_points: bool = True,
    route_rank: Optional[np.ndarray] = None,
    seed_anchor_xy: Optional[np.ndarray] = None,
) -> np.ndarray:
    n0 = int(min(max(n0, 1), len(domain_xy)))
    pts_domain = np.asarray(domain_xy, dtype=float)

    anchor_xy = None
    anchor_idx = None
    if seed_anchor_xy is not None and len(pts_domain) > 0:
        anchor_xy = np.asarray(seed_anchor_xy, dtype=float).reshape(1, 2)
        anchor_idx = int(np.argmin(np.linalg.norm(pts_domain - anchor_xy, axis=1)))
        anchor_xy = pts_domain[anchor_idx].copy()

    if route_rank is not None:
        if anchor_idx is None:
            pts = np.asarray(pts_domain[:n0], dtype=float)
        else:
            idx = np.arange(anchor_idx, min(anchor_idx + n0, len(pts_domain)), dtype=int)
            if len(idx) < n0:
                rem = n0 - len(idx)
                idx = np.concatenate([idx, np.arange(0, rem, dtype=int)])
            pts = np.asarray(pts_domain[idx], dtype=float)
        return pts

    seed_stack = None
    if anchor_idx is not None:
        seed_stack = pts_domain[anchor_idx:anchor_idx + 1]

    if hot_mask is None or not np.any(hot_mask):
        pts = farthest_point_init(pts_domain, n0, seed_stack)
    else:
        n_hot = max(1, min(int(round(0.4 * n0)), int(np.sum(hot_mask))))
        hot_seed = farthest_point_init(pts_domain[hot_mask], n_hot)
        if seed_stack is not None:
            pts = farthest_point_init(pts_domain, n0, np.vstack([seed_stack, hot_seed]))
        else:
            pts = farthest_point_init(pts_domain, n0, hot_seed)

    if order_points:
        return nearest_neighbor_route(
            pts,
            start_xy=anchor_xy if anchor_xy is not None else None,
        )
    return pts

def relabel_point_roles_by_distance(
    X_obs: np.ndarray,
    pred_zone: np.ndarray,
    cfg: FieldConfig,
    n_seed: int,
    warmhot_dist_m: float = 50.0,
    final_tail_dist_m: float = 50.0,
) -> List[str]:
    X_obs = np.asarray(X_obs, dtype=float)
    n = len(X_obs)
    n_seed = int(max(0, min(int(n_seed), n)))

    roles: List[str] = ["seed"] * n_seed + ["exploit"] * (n - n_seed)
    if n <= n_seed:
        return roles

    _, _, Xg, Yg = make_mesh(cfg)
    warmhot_mask = np.asarray(pred_zone) >= 1
    warmhot_pts = np.column_stack([Xg[warmhot_mask], Yg[warmhot_mask]])

    if len(warmhot_pts) == 0:
        for i in range(n_seed, n):
            roles[i] = "explore"
        return roles

    d_wh = np.linalg.norm(
        X_obs[n_seed:, None, :] - warmhot_pts[None, :, :],
        axis=2,
    ).min(axis=1)

    enter_rel = np.where(d_wh <= float(warmhot_dist_m))[0]
    if len(enter_rel) == 0:
        for i in range(n_seed, n):
            roles[i] = "explore"
        return roles

    enter_idx = n_seed + int(enter_rel[0])

    for i in range(n_seed, enter_idx):
        roles[i] = "explore"

    for i in range(enter_idx, n):
        roles[i] = "exploit"

    d_final = np.linalg.norm(
        X_obs - X_obs[-1][None, :],
        axis=1,
    )

    tail_start = None
    for i in range(n - 2, enter_idx - 1, -1):
        if d_final[i] > float(final_tail_dist_m):
            tail_start = i
            break

    if tail_start is not None:
        for i in range(tail_start, n):
            roles[i] = "explore"

    return roles

def run_adaptive(
    geom: SurveyGeometry,
    ops: OperationalCriteria,
    adapt: AdaptiveConfig,
    Z_true: np.ndarray,
    rng: np.random.Generator,
    init_xy: Optional[np.ndarray],
    forecast_mean_grid: Optional[np.ndarray] = None,
    forecast_std_grid: Optional[np.ndarray] = None,
    timing_cfg: Optional[TimingConfig] = None,
    weight_cfg: Optional[DecisionWeightConfig] = None,
) -> RunResult:
    cfg = geom.cfg
    domain_xy, hot_mask, route_rank = build_adaptive_candidate_domain(geom, adapt, init_xy)

    if init_xy is None or len(init_xy) == 0 or route_rank is not None:
        X_obs = initial_adaptive_points(
            domain_xy,
            hot_mask,
            adapt.n0,
            order_points=adapt.order_initial_points,
            route_rank=route_rank,
            seed_anchor_xy=np.asarray(adapt.SEED_START_XY, dtype=float),
        )
    else:
        X_obs = np.asarray(init_xy, dtype=float)
        if adapt.order_initial_points and len(X_obs) > 1:
            X_obs = nearest_neighbor_route(X_obs)

    Z_meas = build_measurement_field(cfg, Z_true)
    truth_interp = build_interpolator(cfg, Z_meas)
    y_obs = sample_field(cfg, truth_interp, X_obs, rng)

    if forecast_mean_grid is not None:
        forecast_mean_interp = build_interpolator(cfg, forecast_mean_grid)
        forecast_std_interp = build_interpolator(cfg, np.nan_to_num(forecast_std_grid, nan=0.0))
        forecast_meas_grid = build_measurement_field(cfg, forecast_mean_grid)
        forecast_meas_interp = build_interpolator(cfg, forecast_meas_grid)
    else:
        forecast_mean_interp = None
        forecast_std_interp = None
        forecast_meas_interp = None

    stop_min_samples = int(adapt.stop_min_samples) if int(adapt.stop_min_samples) > 0 else int(adapt.n0)
    risk_trace: List[Dict[str, float]] = []
    point_roles: List[str] = ["seed"] * len(X_obs)

    metric_stop_hits = 0
    metric_stop_trace: List[Dict[str, float]] = []

    def append_risk_trace() -> bool:
        need_eval = (weight_cfg is not None) or bool(getattr(adapt, "metric_stop_enabled", False))
        if not need_eval:
            return False

        res_now = evaluate_design(
            name="adaptive_step",
            geom=geom,
            ops=ops,
            Z_true=Z_true,
            X_obs=X_obs,
            y_obs=y_obs,
            n_restarts=adapt.gp_restarts,
            forecast_mean_grid=forecast_mean_grid,
            forecast_std_grid=forecast_std_grid,
            timing_cfg=timing_cfg,
        )

        if weight_cfg is not None:
            row = true_weighted_loss_from_result(res_now, ops, weight_cfg)
            row["step"] = float(len(X_obs))
            risk_trace.append(row)

        metric_stop_now = False
        if bool(getattr(adapt, "metric_stop_enabled", False)):
            zone_ok = (
                np.isfinite(res_now.zone_accuracy)
                and float(res_now.zone_accuracy) >= float(
                    getattr(adapt, "metric_stop_zone_accuracy_target", 0.84)
                )
            )

            hot_available = np.isfinite(res_now.hot_recall)
            hot_ok = (
                float(res_now.hot_recall) >= float(
                    getattr(adapt, "metric_stop_hot_recall_target", 0.74)
                )
                if hot_available else True
            )

            action_ok = (
                np.isfinite(res_now.action_recall)
                and float(res_now.action_recall) >= float(
                    getattr(adapt, "metric_stop_action_recall_target", 0.80)
                )
            )

            metric_stop_now = bool(zone_ok and hot_ok and action_ok)

            metric_stop_trace.append({
                "step": float(len(X_obs)),
                "zone_accuracy": float(res_now.zone_accuracy),
                "hot_recall": float(res_now.hot_recall),
                "action_recall": float(res_now.action_recall),
                "zone_ok": float(bool(zone_ok)),
                "hot_available": float(bool(hot_available)),
                "hot_ok": float(bool(hot_ok)),
                "action_ok": float(bool(action_ok)),
                "stop_now": float(bool(metric_stop_now)),
            })

        return bool(metric_stop_now)

    metric_stop_hits = 1 if append_risk_trace() else 0
    if bool(getattr(adapt, "metric_stop_enabled", False)) and metric_stop_hits >= int(max(getattr(adapt, "metric_stop_patience", 1), 1)):
        out = evaluate_design(
            "adaptive", geom, ops, Z_true, X_obs, y_obs,
            n_restarts=adapt.gp_restarts,
            forecast_mean_grid=forecast_mean_grid,
            forecast_std_grid=forecast_std_grid,
            timing_cfg=timing_cfg,
        )
        relabeled_roles = relabel_point_roles_by_distance(
            X_obs=np.asarray(out.X_obs, dtype=float),
            pred_zone=np.asarray(out.pred_zone),
            cfg=cfg,
            n_seed=int(adapt.n0),
            warmhot_dist_m=float(adapt.ROLE_WARMHOT_DIST_M),
            final_tail_dist_m=float(adapt.ROLE_FINAL_TAIL_DIST_M),
        )
        out.diag = {} if out.diag is None else dict(out.diag)
        out.diag["risk_trace"] = risk_trace
        out.diag["acq_mode"] = str(adapt.acq_mode)
        out.diag["point_roles_raw"] = point_roles
        out.diag["point_roles"] = relabeled_roles
        out.diag["metric_stop_trace"] = metric_stop_trace
        return out

    for _ in range(max(0, adapt.max_samples)):
        if forecast_mean_grid is None:
            gp = fit_gp(X_obs, y_obs, cfg, adapt.gp_restarts)
            x_post, score, dbg = choose_next_point_mile_pag_debug(
                gp, X_obs, domain_xy, hot_mask, ops, cfg, adapt, route_rank=route_rank
            )
        else:
            f_obs = forecast_meas_interp(np.column_stack([X_obs[:, 1], X_obs[:, 0]])).astype(float)
            residual_obs = y_obs - f_obs
            gp = fit_gp(X_obs, residual_obs, cfg, adapt.gp_restarts)
            x_post, score, dbg = choose_next_point_mile_pag_residual_debug(
                gp=gp,
                X_obs=X_obs,
                domain_xy=domain_xy,
                hot_prior_mask=hot_mask,
                ops=ops,
                cfg=cfg,
                adapt=adapt,
                forecast_mean_interp=forecast_mean_interp,
                forecast_std_interp=forecast_std_interp,
                route_rank=route_rank,
            )
                      
        if float(adapt.stop_acq) >= 0.0 and len(X_obs) >= stop_min_samples:
            finite = np.asarray(score, dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size == 0 or float(np.nanmax(finite)) < float(adapt.stop_acq):
                break
            
        if str(adapt.acq_mode).lower() == "oracle_onestep":
            if weight_cfg is None:
                raise ValueError("oracle_onestep requires weight_cfg (with fixed t_ref).")

            x_next, _ = choose_next_point_oracle_onestep(
                geom=geom,
                ops=ops,
                adapt=adapt,
                Z_true=Z_true,
                rng=rng,
                X_obs=X_obs,
                y_obs=y_obs,
                domain_xy=domain_xy,
                hot_mask=hot_mask,
                route_rank=route_rank,
                timing_cfg=timing_cfg,
                weight_cfg=weight_cfg,
                forecast_mean_grid=forecast_mean_grid,
                forecast_std_grid=forecast_std_grid,
            )

            role = _role_for_oracle_onestep(
                x_oracle=x_next,
                x_post=x_post,
                domain_xy=domain_xy,
                dbg=dbg,
                ops=ops,
                dist_gate_m=50.0,
            )
        else:
            x_next = x_post
            role = dbg["chosen_role"]

        y_next = sample_field(cfg, truth_interp, x_next[None, :], rng)[0]
        X_obs = np.vstack([X_obs, x_next[None, :]])
        y_obs = np.append(y_obs, y_next)
        point_roles.append(role)

        stop_now = append_risk_trace()
        if bool(getattr(adapt, "metric_stop_enabled", False)):
            metric_stop_hits = metric_stop_hits + 1 if stop_now else 0
            if metric_stop_hits >= int(max(getattr(adapt, "metric_stop_patience", 1), 1)):
                break

    out = evaluate_design(
        "adaptive", geom, ops, Z_true, X_obs, y_obs,
        n_restarts=adapt.gp_restarts,
        forecast_mean_grid=forecast_mean_grid,
        forecast_std_grid=forecast_std_grid,
        timing_cfg=timing_cfg,
    )

    relabeled_roles = relabel_point_roles_by_distance(
        X_obs=np.asarray(out.X_obs, dtype=float),
        pred_zone=np.asarray(out.pred_zone),
        cfg=cfg,
        n_seed=int(adapt.n0),
        warmhot_dist_m=float(adapt.ROLE_WARMHOT_DIST_M),
        final_tail_dist_m=float(adapt.ROLE_FINAL_TAIL_DIST_M),
    )
    
    out.diag = {} if out.diag is None else dict(out.diag)
    out.diag["risk_trace"] = risk_trace
    out.diag["acq_mode"] = str(adapt.acq_mode)
    out.diag["point_roles_raw"] = point_roles
    out.diag["point_roles"] = relabeled_roles
    out.diag["metric_stop_trace"] = metric_stop_trace
    return out


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------


def plot_smooth_track(ax: plt.Axes, xy: np.ndarray, color: str = "#d81b60", lw: float = 2.0, alpha: float = 0.95):
    pts = np.asarray(xy, dtype=float)
    if len(pts) < 2:
        return

    if len(pts) < 4:
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, alpha=alpha)
        return

    try:
        from scipy.interpolate import splprep, splev

        k = min(3, len(pts) - 1)
        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0.0, k=k)
        u_new = np.linspace(0.0, 1.0, max(200, len(pts) * 20))
        xs, ys = splev(u_new, tck)
        ax.plot(xs, ys, color=color, lw=lw, alpha=alpha)
    except Exception:
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, alpha=alpha)

def zone_cmap(ops: OperationalCriteria) -> ListedColormap:
    cmap = ListedColormap(list(ops.colors))
    cmap.set_bad(color="white")
    return cmap

def setup_axes(ax: plt.Axes, cfg: FieldConfig) -> None:
    ax.set_xlim(cfg.bounds[0], cfg.bounds[1])
    ax.set_ylim(cfg.bounds[2], cfg.bounds[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

def load_baseline_overlay_geom_from_shp(shp_path: Optional[str]) -> BaseGeometry:
    if shp_path is None or str(shp_path).strip() == "":
        return _empty_geom()
    try:
        gdf = gpd.read_file(shp_path)
        geoms = [to_geometry(g) for g in gdf.geometry if g is not None and not g.is_empty]
        if len(geoms) == 0:
            return _empty_geom()
        return unary_union(geoms)
    except Exception as e:
        print(f"[warn] Failed to load baseline overlay shapefile '{shp_path}': {e}")
        return _empty_geom()


def grid_mask_for_geom(cfg: "FieldConfig", geom_obj: BaseGeometry) -> np.ndarray:
    if geom_obj is None or geom_obj.is_empty:
        return np.ones((cfg.grid_n, cfg.grid_n), dtype=bool)

    xs = np.linspace(cfg.bounds[0], cfg.bounds[1], cfg.grid_n)
    ys = np.linspace(cfg.bounds[2], cfg.bounds[3], cfg.grid_n)
    XX, YY = np.meshgrid(xs, ys)

    pg = prep(to_geometry(geom_obj))
    mask = np.fromiter(
        (pg.covers(Point(float(x), float(y))) for x, y in zip(XX.ravel(), YY.ravel())),
        dtype=bool,
        count=XX.size,
    ).reshape(XX.shape)
    return mask


def point_mask_for_geom(xy: np.ndarray, geom_obj: BaseGeometry) -> np.ndarray:
    if geom_obj is None or geom_obj.is_empty:
        return np.ones(len(xy), dtype=bool)

    pg = prep(to_geometry(geom_obj))
    keep = np.fromiter(
        (pg.covers(Point(float(x), float(y))) for x, y in np.asarray(xy, dtype=float)),
        dtype=bool,
        count=len(xy),
    )
    return keep

def plot_polygon(ax: plt.Axes, poly: object, color: str, ls: str = "-", lw: float = 1.6) -> None:
    geom = to_geometry(poly)
    if geom.is_empty:
        return

    def _plot_geom(g: BaseGeometry) -> None:
        if isinstance(g, Polygon):
            ext = np.asarray(g.exterior.coords, dtype=float)
            ax.plot(ext[:, 0], ext[:, 1], color=color, ls=ls, lw=lw)
            for ring in g.interiors:
                hole = np.asarray(ring.coords, dtype=float)
                ax.plot(hole[:, 0], hole[:, 1], color=color, ls=ls, lw=max(0.9 * lw, 0.8))
        elif isinstance(g, LineString):
            arr = np.asarray(g.coords, dtype=float)
            ax.plot(arr[:, 0], arr[:, 1], color=color, ls=ls, lw=lw)
        elif hasattr(g, 'geoms'):
            for sub in g.geoms:
                _plot_geom(sub)

    _plot_geom(geom)

def truth_contour_label(ops: OperationalCriteria, idx: int) -> str:
    thr = float(ops.thresholds[idx])

    if len(ops.thresholds) == 1:
        return f"Truth contour @ {thr:g}"

    if idx == 0:
        return "Truth warm contour"
    if idx == len(ops.thresholds) - 1:
        return "Truth hot contour"
    return f"Truth contour @ {thr:g}"

def plot_truth_contours(
    ax: plt.Axes,
    cfg: FieldConfig,
    Z_true: np.ndarray,
    ops: OperationalCriteria,
    color: str = "black",
) -> List[int]:
    xs = np.linspace(cfg.bounds[0], cfg.bounds[1], cfg.grid_n)
    ys = np.linspace(cfg.bounds[2], cfg.bounds[3], cfg.grid_n)
    styles = ["-", "-.", ":", (0, (3, 1, 1, 1))]

    zplot = np.nan_to_num(Z_true, nan=cfg.background_level)
    zmin = float(np.nanmin(zplot))
    zmax = float(np.nanmax(zplot))

    drawn = []
    for i, thr in enumerate(ops.thresholds):
        thr = float(thr)

        if not (zmin < thr < zmax):
            continue

        ax.contour(
            xs, ys, zplot,
            levels=[thr],
            colors=[color],
            linewidths=1.3,
            linestyles=[styles[i % len(styles)]],
        )
        drawn.append(i)

    return drawn

def plot_truth_contours_dummy(cfg: FieldConfig, Z_true: np.ndarray, ops: OperationalCriteria) -> List[int]:
    zplot = np.nan_to_num(Z_true, nan=cfg.background_level)
    zmin = float(np.nanmin(zplot))
    zmax = float(np.nanmax(zplot))
    drawn = []
    for i, thr in enumerate(ops.thresholds):
        if zmin < float(thr) < zmax:
            drawn.append(i)
    return drawn

def plot_single_run_separate(
    geom: SurveyGeometry,
    ops: OperationalCriteria,
    Z_true: np.ndarray,
    meta: Dict[str, object],
    baseline: RunResult,
    adaptive: RunResult,
    out_dir: str,
) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    cfg = geom.cfg
    extent = [cfg.bounds[0], cfg.bounds[1], cfg.bounds[2], cfg.bounds[3]]
    hot_idx = top_zone_index(ops)
    truth_has_hot = bool(np.any(baseline.truth_zone == hot_idx))

    def fmt_hot(v: float) -> str:
        if (not truth_has_hot) or (not np.isfinite(v)):
            return "N/A"
        return f"{v:.2f}"

    drawn_truth_levels = plot_truth_contours_dummy(cfg, Z_true, ops)
    legend_handles = [
        Line2D(
            [0], [0],
            color="white",
            ls="--",
            lw=1.4,
            label="Survey boundary" if not meta.get("has_hot_prior", True) else "Warm prior",
        ),
    ]
    if meta.get("has_hot_prior", True):
        legend_handles.append(
            Line2D([0], [0], color="white", ls=":", lw=1.4, label="Hot prior")
        )
    styles = ["-", "-.", ":", (0, (3, 1, 1, 1))]
    for idx in drawn_truth_levels:
        legend_handles.append(
            Line2D(
                [0], [0],
                color="black",
                ls=styles[idx % len(styles)],
                lw=1.4,
                label=truth_contour_label(ops, idx),
            )
        )

    out = {}

    # 1) truth
    fig, ax = plt.subplots(figsize=(9, 5.5))
    Z_true_plot = np.nan_to_num(Z_true, nan=cfg.background_level)
    im = ax.imshow(Z_true_plot, origin="lower", extent=extent, cmap="jet")
    plot_polygon(ax, meta["pred_warm_poly"], "white", "--", 1.4)
    plot_polygon(ax, meta["pred_hot_poly"], "white", ":", 1.4)
    plot_truth_contours(ax, cfg, Z_true, ops, color="black")
    ax.set_title("Truth field")
    setup_axes(ax, cfg)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.legend(handles=legend_handles,
              loc="lower center",
              bbox_to_anchor=(0.5, 1.08),
              ncol = 2,
              framealpha=1.0,
              fancybox=False,
              edgecolor="0.2",
              borderaxespad=0.0,)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out["truth_png"] = os.path.join(out_dir, "truth_field.png")
    fig.savefig(out["truth_png"], bbox_inches="tight")
    plt.close(fig)

    # 2) baseline
    fig, ax = plt.subplots(figsize=(9, 5.5))

    overlay_geom = getattr(geom, "plot_overlay_geom", _empty_geom())
    baseline_plot = np.asarray(baseline.pred_zone, dtype=float).copy()
    baseline_plot[~np.isfinite(baseline_plot)] = 0.0
    baseline_plot[baseline_plot < 0] = 0.0

    overlay_mask = None
    if overlay_geom is not None and (not overlay_geom.is_empty):
        overlay_mask = grid_mask_for_geom(cfg, overlay_geom)
        baseline_plot = np.ma.masked_where(~overlay_mask, baseline_plot)

    ax.imshow(
        baseline_plot,
        origin="lower",
        extent=extent,
        cmap=zone_cmap(ops),
        vmin=0,
        vmax=len(ops.thresholds),
        interpolation="nearest",
    )

    if len(baseline.X_obs) > 0:
        xy_obs = np.asarray(baseline.X_obs, dtype=float)
        y_obs = np.asarray(baseline.y_obs, dtype=float)

        thr_warm = float(ops.thresholds[0])
        thr_hot = float(ops.thresholds[-1])

        keep_pts = point_mask_for_geom(xy_obs, overlay_geom) if (overlay_geom is not None and not overlay_geom.is_empty) else np.ones(len(xy_obs), dtype=bool)

        cold_mask = (y_obs < thr_warm) & keep_pts
        warm_mask = (y_obs >= thr_warm) & (y_obs < thr_hot) & keep_pts
        hot_mask = (y_obs >= thr_hot) & keep_pts

        if np.any(cold_mask):
            ax.scatter(
                xy_obs[cold_mask, 0],
                xy_obs[cold_mask, 1],
                c="black",
                marker="o",
                s=20,
                zorder=3,
                label="Cold point",
            )

        if np.any(warm_mask):
            ax.scatter(
                xy_obs[warm_mask, 0],
                xy_obs[warm_mask, 1],
                c="black",
                marker="^",
                s=34,
                zorder=3,
                label="Warm point",
            )

        if np.any(hot_mask):
            ax.scatter(
                xy_obs[hot_mask, 0],
                xy_obs[hot_mask, 1],
                c="black",
                marker="x",
                s=34,
                zorder=3,
                label="Hot point",
            )

        visible_xy = xy_obs[keep_pts]
        if len(visible_xy) > 1:
            plot_smooth_track(ax, visible_xy, color="#d81b60", lw=2.0, alpha=0.95)

    ax.legend(loc="upper right", framealpha=1.0, fancybox=False, edgecolor="0.2")

    plot_polygon(ax, meta["pred_warm_poly"], "white", "--", 1.2)
    plot_polygon(ax, meta["pred_hot_poly"], "white", ":", 1.2)
    plot_truth_contours(ax, cfg, Z_true, ops, color="black")

    if overlay_geom is not None and (not overlay_geom.is_empty):
        plot_polygon(ax, overlay_geom, "black", "-", 1.6)

    ax.set_title(
        f"Baseline\nacc={baseline.zone_accuracy:.2f}, hot recall={fmt_hot(baseline.hot_recall)}, "
        f"T={baseline.total_time_min:.1f} min"
    )

    # 3) adaptive
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.imshow(
        adaptive.pred_zone,
        origin="lower",
        extent=extent,
        cmap=zone_cmap(ops),
        vmin=0,
        vmax=len(ops.thresholds),
        interpolation="nearest",
    )

    roles = None
    if adaptive.diag is not None:
        roles = adaptive.diag.get("point_roles", None)

    if roles is None or len(roles) != len(adaptive.X_obs):
        roles = ["seed"] * len(adaptive.X_obs)
        
    roles_plot = ["explore" if r == "seed" else r for r in roles]

    role_colors = {
        "explore": "#1f77b4",
        "exploit": "#d9791f",
    }

    # route segments
    if len(adaptive.X_obs) > 1:
        for i in range(1, len(adaptive.X_obs)):
            role_i = roles_plot[i] if i < len(roles_plot) else "explore"
            c = role_colors.get(role_i, "black")
            ax.plot(
                adaptive.X_obs[i-1:i+1, 0],
                adaptive.X_obs[i-1:i+1, 1],
                color=c,
                lw=1.2,
                alpha=0.75,
                zorder=2,
            )

    roles_arr = np.asarray(roles_plot, dtype=object)
    for role_name in ["explore", "exploit"]:
        mask = roles_arr == role_name
        if np.any(mask):
            ax.scatter(
                adaptive.X_obs[mask, 0],
                adaptive.X_obs[mask, 1],
                c=role_colors[role_name],
                s=26,
                edgecolors="black",
                linewidths=0.4,
                zorder=3,
            )

    plot_polygon(ax, meta["pred_warm_poly"], "white", "--", 1.2)
    plot_polygon(ax, meta["pred_hot_poly"], "white", ":", 1.2)
    plot_truth_contours(ax, cfg, Z_true, ops, color="black")

    role_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=role_colors["explore"],
               markeredgecolor="black", markersize=8, label="Explore"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=role_colors["exploit"],
               markeredgecolor="black", markersize=8, label="Exploit"),
    ]
    ax.legend(
        handles=role_handles,
        loc="upper right",
        framealpha=1.0,
        fancybox=False,
        edgecolor="0.2",
    )

    ax.set_title(
        f"Adaptive MILE-PAG\nacc={adaptive.zone_accuracy:.2f}, hot recall={fmt_hot(adaptive.hot_recall)}, "
        f"T={adaptive.total_time_min:.1f} min"
    )
    setup_axes(ax, cfg)
    fig.tight_layout()
    out["adaptive_png"] = os.path.join(out_dir, "adaptive.png")
    fig.savefig(out["adaptive_png"], bbox_inches="tight")
    plt.close(fig)
    return out

def plot_mc_summary(df: pd.DataFrame, out_path: str) -> None:
    metrics = ["zone_accuracy", "hot_recall", "action_recall", "n_used"]
    titles = ["Zone accuracy", "Hot recall", "Action recall", "Samples used"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))

    for ax, metric, title in zip(axes, metrics, titles):
        means = df.groupby("design")[metric].mean().reindex(["baseline", "adaptive"])
        stds = df.groupby("design")[metric].std(ddof=0).fillna(0.0).reindex(["baseline", "adaptive"])

        vals = means.to_numpy(dtype=float)
        errs = stds.to_numpy(dtype=float)

        vals_plot = np.nan_to_num(vals, nan=0.0)
        errs_plot = np.where(np.isnan(vals), 0.0, errs)

        x = np.arange(2)
        bars = ax.bar(
            x,
            vals_plot,
            yerr=errs_plot,
            color=["#b8d3e6", "#0055a4"],
            edgecolor="black",
            linewidth=1.1,
            capsize=4,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(["Baseline", "Adaptive"])
        ax.set_title(title)

        if metric != "n_used":
            ax.set_ylim(0, 1.05)
        else:
            finite_vals = vals[np.isfinite(vals)]
            ymax = max(float(np.nanmax(finite_vals)) if finite_vals.size else 1.0, 1.0)
            ax.set_ylim(0, 1.18 * ymax)

        finite_vals = vals[np.isfinite(vals)]
        ymax_ref = max(float(np.nanmax(finite_vals)) if finite_vals.size else 1.0, 1.0)

        for i, v in enumerate(vals):
            if np.isnan(v):
                ypos = 0.03 if metric != "n_used" else 0.05 * ymax_ref
                ax.text(i, ypos, "N/A", ha="center")
                bars[i].set_alpha(0.35)
                bars[i].set_hatch("//")
            else:
                offset = 0.03 if metric != "n_used" else 0.05 * ymax_ref
                ax.text(i, v + offset, f"{v:.2f}", ha="center")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def plot_vsp_track(geom: SurveyGeometry, Z_true: np.ndarray, meta: Dict[str, object],
                   initial_xy: np.ndarray, adaptive: RunResult, out_path: str) -> None:
    cfg = geom.cfg
    extent = [cfg.bounds[0], cfg.bounds[1], cfg.bounds[2], cfg.bounds[3]]
    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    ax.imshow(Z_true, origin="lower", extent=extent, cmap="jet", alpha=0.8)
    plot_polygon(ax, meta["pred_warm_poly"], "white", "--", 1.4)
    plot_polygon(ax, meta["pred_hot_poly"], "white", ":", 1.4)
    ax.scatter(initial_xy[:, 0], initial_xy[:, 1], c="#1f77b4", s=70, edgecolors="white", linewidths=1.0, label=f"Initial points (N={len(initial_xy)})")
    added = adaptive.X_obs[len(initial_xy):]
    if len(added) > 0:
        ax.scatter(added[:, 0], added[:, 1], c="red", marker="*", s=220, edgecolors="black", label=f"AI added points (N={len(added)})")
        for i, (x, y) in enumerate(added, start=1):
            ax.text(x, y, f"#{i}", color="white", ha="center", va="center", fontsize=9, fontweight="bold")
    ax.legend(loc="upper right", framealpha=1.0, fancybox=False, edgecolor="0.2")
    ax.set_title("VSP + adaptive MILE-PAG")
    setup_axes(ax, cfg)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Weighted operational evaluation
# ---------------------------------------------------------------------


def build_weight_cfg(args: argparse.Namespace) -> DecisionWeightConfig:
    return DecisionWeightConfig(
        w_t=float(args.w_t),
        w_fr=float(args.w_fr),
        w_fh=float(args.w_fh),
        t_ref=float(args.t_ref),
    )

def top_zone_index(ops: OperationalCriteria) -> int:
    return int(len(ops.thresholds))

def aggregate_weighted_summary(df: pd.DataFrame, weights: DecisionWeightConfig) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()

    agg = df.groupby('design', as_index=False).agg(
        zone_accuracy_mean=('zone_accuracy', 'mean'),
        zone_accuracy_std=('zone_accuracy', 'std'),
        hot_recall_mean=('hot_recall', 'mean'),
        hot_recall_std=('hot_recall', 'std'),
        action_recall_mean=('action_recall', 'mean'),
        action_recall_std=('action_recall', 'std'),
        action_precision_mean=('action_precision', 'mean'),
        action_precision_std=('action_precision', 'std'),
        n_used_mean=('n_used', 'mean'),
        n_used_std=('n_used', 'std'),
        path_length_mean=('path_length_m', 'mean'),
        path_length_std=('path_length_m', 'std'),
        travel_time_mean=('travel_time_min', 'mean'),
        travel_time_std=('travel_time_min', 'std'),
        station_time_mean=('station_time_total_min', 'mean'),
        station_time_std=('station_time_total_min', 'std'),
        total_time_mean=('total_time_min', 'mean'),
        total_time_std=('total_time_min', 'std'),
        truth_has_hot_rate=('truth_has_hot', 'mean'),
        pred_has_hot_rate=('pred_has_hot', 'mean'),
        p_hot_overcall=('hot_overcall_area', 'mean'),   
    )

    for col in [
        'zone_accuracy_std', 'hot_recall_std', 'action_recall_std', 'action_precision_std',
        'n_used_std', 'path_length_std', 'travel_time_std', 'station_time_std', 'total_time_std'
    ]:
        agg[col] = agg[col].fillna(0.0)

    agg['p_hot_miss'] = (
        1.0 - agg['hot_recall_mean'].astype(float)
    ).clip(lower=0.0, upper=1.0).fillna(0.0)

    t_ref = float(weights.t_ref)
    if t_ref <= 0.0:
        if 'baseline' in set(agg['design']):
            t_ref = float(agg.loc[agg['design'] == 'baseline', 'total_time_mean'].iloc[0])
        else:
            t_ref = float(max(agg['total_time_mean'].mean(), 1.0))

    agg['t_ref_used'] = t_ref
    agg['T_n'] = agg['total_time_mean'] / max(t_ref, 1e-12)

    if 'baseline' in set(agg['design']):
        agg.loc[agg['design'] == 'baseline', 'T_n'] = 1.0

    agg['fr_penalty'] = float(weights.w_fr) * agg['p_hot_miss']
    agg['fh_penalty'] = float(weights.w_fh) * agg['p_hot_overcall']
    agg['time_penalty'] = float(weights.w_t) * agg['T_n']
    agg['J'] = agg['time_penalty'] + agg['fr_penalty'] + agg['fh_penalty']

    base_J = float(agg.loc[agg['design'] == 'baseline', 'J'].iloc[0]) \
        if 'baseline' in set(agg['design']) else float(agg['J'].iloc[0])
    base_J = max(base_J, 1e-12)

    agg['J_norm'] = agg['J'] / base_J
    agg['fr_penalty_norm'] = agg['fr_penalty'] / base_J
    agg['fh_penalty_norm'] = agg['fh_penalty'] / base_J
    agg['time_penalty_norm'] = agg['time_penalty'] / base_J
    return agg

def plot_weighted_summary(weighted_df: pd.DataFrame, out_path: str) -> None:
    if len(weighted_df) == 0:
        return
    order = [d for d in ['baseline', 'adaptive'] if d in set(weighted_df['design'])]
    sub = weighted_df.set_index('design').loc[order]
    x = np.arange(len(order))
    colors = ['#b8d3e6', '#0055a4']
    labels = ['Baseline' if d == 'baseline' else 'Adaptive' for d in order]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    panels = [
    ('p_hot_miss', 'Hot miss rate', (0.0, 1.05)),
    ('p_hot_overcall', 'Hot overcall rate', (0.0, 1.05)),
    ('T_n', 'Normalized total time $T_n$', None),
    ('J_norm', 'Weighted objective $J$ (norm.)', None),
    ]

    for ax, (col, title, ylim) in zip(axes, panels):
        vals = sub[col].to_numpy(dtype=float)
        ax.bar(x, vals, color=colors[:len(order)], edgecolor='black', linewidth=1.1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            ymax = max(float(np.nanmax(vals)), 1e-9)
            ax.set_ylim(0.0, 1.15 * ymax if ymax > 0 else 1.0)
        for i, v in enumerate(vals):
            offset = 0.03 if ax.get_ylim()[1] <= 1.2 else 0.04 * max(ax.get_ylim()[1], 1.0)
            ax.text(i, v + offset, f'{v:.2f}', ha='center')

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

def event_flags_from_result(res: RunResult, ops: OperationalCriteria) -> Dict[str, int]:
    hot_idx = top_zone_index(ops)
    truth_has_hot = int(np.any(res.truth_zone == hot_idx))
    pred_has_hot = int(np.any(res.pred_zone == hot_idx))

    action_truth = int(np.any(res.truth_zone >= 1))
    action_pred = int(np.any(res.pred_zone >= 1))

    return {
        "truth_has_hot": truth_has_hot,
        "pred_has_hot": pred_has_hot,
        "hot_miss_run": int(bool(truth_has_hot) and not bool(pred_has_hot)),
        "hot_overcall_run": int((not bool(truth_has_hot)) and bool(pred_has_hot)),
        "action_miss_run": int(bool(action_truth) and not bool(action_pred)),
        "action_overcall_run": int((not bool(action_truth)) and bool(action_pred)),
    }

def overcall_area_rates_from_result(
    res: RunResult,
    ops: OperationalCriteria,
) -> Dict[str, float]:
    """
    Area-based false positive rates.
    These are better proxies for 'overcalling' than the current binary run-level flag.
    """
    hot_idx = top_zone_index(ops)
    valid = res.truth_zone >= 0

    hot_truth = (res.truth_zone == hot_idx)
    hot_pred = (res.pred_zone == hot_idx)

    action_truth = (res.truth_zone >= 1)
    action_pred = (res.pred_zone >= 1)

    hot_fp = hot_pred & (~hot_truth) & valid
    action_fp = action_pred & (~action_truth) & valid

    non_hot_truth = (~hot_truth) & valid
    non_action_truth = (~action_truth) & valid

    hot_overcall_area = float(np.sum(hot_fp) / max(np.sum(non_hot_truth), 1))
    action_overcall_area = float(np.sum(action_fp) / max(np.sum(non_action_truth), 1))

    return {
        "hot_overcall_area": hot_overcall_area,
        "action_overcall_area": action_overcall_area,
    }

def true_weighted_loss_from_result(
    res: RunResult,
    ops: OperationalCriteria,
    weight_cfg: DecisionWeightConfig,
) -> Dict[str, float]:
    if float(weight_cfg.t_ref) <= 0.0:
        raise ValueError("Online J_n requires a fixed positive t_ref. Pass weight_cfg with t_ref > 0.")

    flags = event_flags_from_result(res, ops)
    over = overcall_area_rates_from_result(res, ops)

    T_n = float(res.total_time_min) / max(float(weight_cfg.t_ref), 1e-12)

    fr = float(flags["hot_miss_run"])
    fh = float(over["hot_overcall_area"])

    J_true = (
        float(weight_cfg.w_t) * T_n +
        float(weight_cfg.w_fr) * fr +
        float(weight_cfg.w_fh) * fh
    )

    return {
        "n_used": float(res.n_used),
        "path_length_m": float(res.path_length_m),
        "travel_time_min": float(res.travel_time_min),
        "station_time_total_min": float(res.station_time_total_min),
        "total_time_min": float(res.total_time_min),
        "T_n": float(T_n),
        "truth_has_hot": float(flags["truth_has_hot"]),
        "pred_has_hot": float(flags["pred_has_hot"]),
        "hot_miss_run": float(fr),
        "hot_overcall_run": float(flags["hot_overcall_run"]),
        "hot_overcall_area": float(over["hot_overcall_area"]),
        "action_overcall_area": float(over["action_overcall_area"]),
        "J_true": float(J_true),
    }

def evaluate_prefix_curve(
    geom: SurveyGeometry,
    ops: OperationalCriteria,
    Z_true: np.ndarray,
    X_obs: np.ndarray,
    y_obs: np.ndarray,
    n_restarts: int,
    forecast_mean_grid: Optional[np.ndarray] = None,
    forecast_std_grid: Optional[np.ndarray] = None,
    timing_cfg: Optional[TimingConfig] = None,
    design_name: str = "adaptive",
) -> pd.DataFrame:
    rows = []
    for n in range(1, len(X_obs) + 1):
        res_n = evaluate_design(
            name=f"{design_name}_n{n}",
            geom=geom,
            ops=ops,
            Z_true=Z_true,
            X_obs=np.asarray(X_obs[:n], dtype=float),
            y_obs=np.asarray(y_obs[:n], dtype=float),
            n_restarts=n_restarts,
            forecast_mean_grid=forecast_mean_grid,
            forecast_std_grid=forecast_std_grid,
            timing_cfg=timing_cfg,
        )
        flags = event_flags_from_result(res_n, ops)
        over = overcall_area_rates_from_result(res_n, ops)

        rows.append({
            "n_used": n,
            "zone_accuracy": res_n.zone_accuracy,
            "hot_recall": res_n.hot_recall,
            "action_recall": res_n.action_recall,
            "action_precision": res_n.action_precision,
            "path_length_m": res_n.path_length_m,
            "travel_time_min": res_n.travel_time_min,
            "station_time_total_min": res_n.station_time_total_min,
            "total_time_min": res_n.total_time_min,
            "hot_overcall_area": over["hot_overcall_area"],
            "action_overcall_area": over["action_overcall_area"],
            **flags,
        })
    return pd.DataFrame(rows)

def summarize_prefix_mc(prefix_df: pd.DataFrame, weights: DecisionWeightConfig) -> pd.DataFrame:
    tmp = prefix_df.copy()
    tmp["hot_miss_from_recall"] = (
        1.0 - tmp["hot_recall"].astype(float)
    ).clip(lower=0.0, upper=1.0).fillna(0.0)

    agg = tmp.groupby("n_used", as_index=False).agg(
        p_hot_miss=("hot_miss_from_recall", "mean"),
        p_hot_overcall=("hot_overcall_area", "mean"),
        zone_accuracy=("zone_accuracy", "mean"),
        hot_recall=("hot_recall", "mean"),
        action_recall=("action_recall", "mean"),
        action_precision=("action_precision", "mean"),
        total_time_mean=("total_time_min", "mean"),
        total_time_std=("total_time_min", "std"),
    )
    agg["total_time_std"] = agg["total_time_std"].fillna(0.0)

    if float(weights.t_ref) > 0:
        t_ref = float(weights.t_ref)
    else:
        t_ref = float(max(agg["total_time_mean"].max(), 1.0))

    agg["t_ref_used"] = t_ref
    agg["T_n"] = agg["total_time_mean"] / max(t_ref, 1e-12)
    agg["J"] = (
        float(weights.w_fr) * agg["p_hot_miss"]
        + float(weights.w_fh) * agg["p_hot_overcall"]
        + float(weights.w_t) * agg["T_n"]
    )
    return agg

def alpha_sensitivity_from_prefix(
    prefix_mc: pd.DataFrame,
    alpha_grid: np.ndarray,
    fh_ratio: float = 0.30,
) -> pd.DataFrame:
    rows = []
    for alpha in alpha_grid:
        tmp = prefix_mc.copy()
        tmp["J_alpha"] = (
            float(alpha) * tmp["p_hot_miss"]
            + float(alpha) * float(fh_ratio) * tmp["p_hot_overcall"]
            + (1.0 - float(alpha)) * tmp["T_n"]
        )
        best = tmp.iloc[int(np.argmin(tmp["J_alpha"].to_numpy(dtype=float)))]
        miss_pct = 100.0 * float(best["p_hot_miss"])
        rows.append({
            "alpha": float(alpha),
            "n_star": float(best["n_used"]),
            "fatal_miss_rate_pct": miss_pct,
            "hot_miss_from_recall_pct": miss_pct,  
            "risk_cost": float(best["J_alpha"]),
        })
    return pd.DataFrame(rows)

def lambda_heatmaps_from_prefix(
    prefix_mc: pd.DataFrame,
    lambda_t_grid: Sequence[float],
    lambda_fh_grid: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lambda_t_grid = np.asarray(lambda_t_grid, dtype=float)
    lambda_fh_grid = np.asarray(lambda_fh_grid, dtype=float)

    Z_nstar = np.zeros((len(lambda_fh_grid), len(lambda_t_grid)), dtype=float)
    Z_jstar = np.zeros((len(lambda_fh_grid), len(lambda_t_grid)), dtype=float)
    Z_miss = np.zeros((len(lambda_fh_grid), len(lambda_t_grid)), dtype=float)

    for i, lam_fh in enumerate(lambda_fh_grid):
        for j, lam_t in enumerate(lambda_t_grid):
            tmp = prefix_mc.copy()
            tmp["J_lambda"] = (
                tmp["p_hot_miss"]
                + float(lam_fh) * tmp["p_hot_overcall"]
                + float(lam_t) * tmp["T_n"]
            )

            best = tmp.iloc[int(np.argmin(tmp["J_lambda"].to_numpy(dtype=float)))]
            Z_nstar[i, j] = float(best["n_used"])
            Z_jstar[i, j] = float(best["J_lambda"])
            Z_miss[i, j] = 100.0 * float(best["p_hot_miss"])

    return lambda_t_grid, lambda_fh_grid, Z_nstar, Z_jstar, Z_miss

def plot_fatal_miss_curve(prefix_mc: pd.DataFrame, baseline_hot_miss_rate: float, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(prefix_mc["n_used"], 100.0 * prefix_mc["p_hot_miss"], lw=2.5, color="#1f77b4", label="Adaptive GP")
    ax.axhline(100.0 * baseline_hot_miss_rate, color="red", lw=2.0, ls="--", label="Baseline")
    ax.set_xlabel("Number of samples [n]")
    ax.set_ylabel("Fatal Miss Rate [%]")
    ax.set_title("Fatal Miss Rate Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def plot_pareto_front(
    df: pd.DataFrame,
    baseline_n: float,
    baseline_J: float,
    out_path: str,
    *,
    x_col: str = "n_used",
    j_col: str = "J",
    point_label: str = "All prefix points",
    title: str = "Pareto Front (best-so-far envelope)",
    x_label: str = "Sample count N",
    y_label: str = "Risk Cost J",
) -> None:
    if len(df) == 0:
        return
    if x_col not in df.columns:
        raise ValueError(f"Pareto plot requires column '{x_col}'")
    if j_col not in df.columns:
        raise ValueError(f"Pareto plot requires column '{j_col}'")

    tmp = df[[x_col, j_col]].copy()
    tmp = tmp.dropna(subset=[x_col, j_col]).sort_values(x_col)
    if len(tmp) == 0:
        return

    x = tmp[x_col].to_numpy(dtype=float)
    j = tmp[j_col].to_numpy(dtype=float)
    tmp["J_best_so_far"] = np.minimum.accumulate(j)

    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    ax.scatter(x, j, s=35, color="#1f77b4", alpha=0.7, label=point_label)
    ax.plot(x, tmp["J_best_so_far"], color="red", lw=2.2, label="Best-so-far envelope")
    ax.scatter([baseline_n], [baseline_J], marker="D", s=90, color="gray", label="Baseline")

    knee_idx = int(np.argmin(tmp["J_best_so_far"].to_numpy(dtype=float)))
    ax.scatter(
        [float(x[knee_idx])],
        [float(tmp.iloc[knee_idx]["J_best_so_far"])],
        marker="*", s=500, color="#cfa600", edgecolors="0.3", zorder=4, label="Knee point"
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def plot_alpha_sensitivity(alpha_df: pd.DataFrame, out_path: str) -> None:
    fig, ax1 = plt.subplots(figsize=(7.2, 5.2))
    ax2 = ax1.twinx()

    miss_col = (
        "fatal_miss_rate_pct"
        if "fatal_miss_rate_pct" in alpha_df.columns
        else "hot_miss_from_recall_pct"
    )
    if miss_col not in alpha_df.columns:
        raise KeyError(
            "plot_alpha_sensitivity requires 'fatal_miss_rate_pct' "
            "or 'hot_miss_from_recall_pct' in alpha_df"
        )

    ax1.plot(alpha_df["alpha"], alpha_df[miss_col], lw=2.5, color="#1f77b4")
    ax2.plot(alpha_df["alpha"], alpha_df["risk_cost"], lw=2.5, color="#b22222")

    ax1.set_xlabel("Risk Weight α")
    ax1.set_ylabel("Fatal Miss Rate [%]", color="#1f77b4")
    ax2.set_ylabel("Risk Cost J", color="#b22222")
    ax1.set_title("Alpha Sensitivity")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def plot_lambda_heatmap(
    lambda_t_grid: np.ndarray,
    lambda_fh_grid: np.ndarray,
    Z: np.ndarray,
    out_path: str,
    title: str,
    cbar_label: str,
    fmt: str = "%.1f",
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.8))

    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[
            float(lambda_t_grid.min()), float(lambda_t_grid.max()),
            float(lambda_fh_grid.min()), float(lambda_fh_grid.max())
        ],
        cmap="jet",
    )

    try:
        cs = ax.contour(
            lambda_t_grid,
            lambda_fh_grid,
            Z,
            colors="white",
            linewidths=1.0,
        )
        ax.clabel(cs, inline=True, fontsize=9, fmt=fmt)
    except Exception:
        pass

    for i, lam_fh in enumerate(lambda_fh_grid):
        for j, lam_t in enumerate(lambda_t_grid):
            ax.text(
                float(lam_t), float(lam_fh),
                fmt % Z[i, j],
                ha="center", va="center",
                color="black", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.55),
            )

    ax.set_xlabel(r"Time weight $\lambda_t = w_t / w_{FR}$")
    ax.set_ylabel(r"Overcall weight $\lambda_{FH} = w_{FH} / w_{FR}$")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label=cbar_label)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def plot_lambda_heatmap_paperstyle_from_summary(
    summary_df: pd.DataFrame,
    out_path: str,
    *,
    value_col: str = "N_star",
    title: str = r"2D Sensitivity Heatmap",
    cbar_label: str = r"Optimal sample count $N^*$",
    x_col: str = "lambda_t",
    y_col: str = "lambda_fh",
    x_label: str = r"Time weight $\lambda_t = w_t / w_{FR}$",
    y_label: str = r"Overcall weight $\lambda_{FH} = w_{FH} / w_{FR}$",
    contour_levels: Optional[Sequence[float]] = None,
    dense_n: int = 240,
    smooth_sigma: float = 1.0,
) -> None:
    if len(summary_df) == 0:
        return
    need_cols = {x_col, y_col, value_col}
    missing = [c for c in need_cols if c not in summary_df.columns]
    if missing:
        raise ValueError(f"Paper-style heatmap requires columns: {missing}")

    df = summary_df[[x_col, y_col, value_col]].copy()
    df = df.dropna(subset=[x_col, y_col, value_col])
    if len(df) == 0:
        return

    x_vals = np.sort(df[x_col].unique().astype(float))
    y_vals = np.sort(df[y_col].unique().astype(float))
    Z = (
        df.pivot(index=y_col, columns=x_col, values=value_col)
          .reindex(index=y_vals, columns=x_vals)
          .to_numpy(dtype=float)
    )

    x_dense = np.linspace(float(x_vals.min()), float(x_vals.max()), int(max(dense_n, 50)))
    y_dense = np.linspace(float(y_vals.min()), float(y_vals.max()), int(max(dense_n, 50)))
    YY, XX = np.meshgrid(y_dense, x_dense, indexing="ij")

    interp = RegularGridInterpolator(
        (y_vals, x_vals),
        Z,
        bounds_error=False,
        fill_value=None,
    )
    Z_dense = interp(np.column_stack([YY.ravel(), XX.ravel()])).reshape(YY.shape)
    if float(smooth_sigma) > 0:
        Z_dense = gaussian_filter(Z_dense, sigma=float(smooth_sigma), mode="nearest")

    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    im = ax.imshow(
        Z_dense,
        origin="lower",
        aspect="auto",
        extent=[float(x_dense.min()), float(x_dense.max()), float(y_dense.min()), float(y_dense.max())],
        cmap="jet",
    )

    if contour_levels is None:
        zmin = float(np.nanmin(Z_dense))
        zmax = float(np.nanmax(Z_dense))
        if zmax > zmin:
            contour_levels = np.linspace(zmin, zmax, 8)
        else:
            contour_levels = [zmin]
    elif np.isscalar(contour_levels):
        zmin = float(np.nanmin(Z_dense))
        zmax = float(np.nanmax(Z_dense))
        contour_levels = np.linspace(zmin, zmax, int(contour_levels))
    contour_levels = np.asarray(list(contour_levels), dtype=float)
    contour_levels = contour_levels[np.isfinite(contour_levels)]
    contour_levels = np.unique(contour_levels)

    try:
        if contour_levels.size >= 1:
            cs = ax.contour(
                x_dense,
                y_dense,
                Z_dense,
                levels=contour_levels,
                colors="white",
                linewidths=1.0,
            )
            ax.clabel(cs, inline=True, fontsize=9, fmt="%.2f")
    except Exception:
        pass

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label=cbar_label)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
# ---------------------------------------------------------------------
# Build from args
# ---------------------------------------------------------------------


def bounds_from_polygon(poly: object) -> Tuple[float, float, float, float]:
    geom = to_geometry(poly)
    if geom.is_empty:
        raise ValueError('Geometry is empty; cannot compute bounds.')
    xmin, ymin, xmax, ymax = geom.bounds
    return float(xmin), float(xmax), float(ymin), float(ymax)

def _largest_polygon_component(geom):
    geom = to_geometry(geom)
    if geom.is_empty:
        raise ValueError("Expanded survey geometry is empty")
    if isinstance(geom, Polygon):
        return geom
    if hasattr(geom, "geoms"):
        polys = [g for g in geom.geoms if isinstance(g, Polygon)]
        if len(polys) == 0:
            raise ValueError("Expanded survey geometry has no polygon component")
        return max(polys, key=lambda g: g.area)
    raise ValueError("Unsupported geometry type for survey domain")

def polygon_to_array(poly_geom: object) -> np.ndarray:
    geom = _largest_polygon_component(poly_geom)
    return np.asarray(geom.exterior.coords, dtype=float)

def make_buffered_survey_geom(prior_warm_geom: BaseGeometry, buffer_m: float) -> BaseGeometry:
    geom = to_geometry(prior_warm_geom)
    if float(buffer_m) == 0.0:
        return geom
    g = geom.buffer(float(buffer_m), join_style=2)
    if not g.is_valid:
        g = g.buffer(0)
    return to_geometry(g)

def build_geometry(args: argparse.Namespace) -> SurveyGeometry:
    warm_poly_raw, hot_poly_raw, dxf_summary, resolved_mode = infer_zone_polygons_from_dxf(
        args.zone_dxf,
        zone_mode=args.zone_mode,
    )
    warm_geom = to_geometry(warm_poly_raw)
    hot_geom = to_geometry(hot_poly_raw)
    has_hot_prior = not hot_geom.is_empty

    survey_summary = None
    label_col = None

    survey_geom = make_buffered_survey_geom(warm_geom, args.survey_buffer_m)
    survey_geom = to_geometry(survey_geom)

    if survey_geom.is_empty:
        raise ValueError('Survey geometry is empty after applying the selected survey-domain source.')

    cfg = FieldConfig(
        bounds=bounds_from_polygon(survey_geom),
        grid_n=args.grid_n,
        background_level=args.background,
        meas_sigma=args.meas_sigma,
        measurement_diameter_m=args.measurement_diameter_m,
        measurement_model=args.measurement_model,
    )

    r = 0.5 * cfg.measurement_diameter_m
    valid_center_geom = to_geometry(survey_geom).buffer(-r)
    if valid_center_geom.is_empty:
        raise ValueError("Measurement footprint is too large for the survey geometry")

    valid_center_hot_geom = valid_center_geom.intersection(hot_geom) if has_hot_prior else _empty_geom()

    summary_frames: List[pd.DataFrame] = []
    if dxf_summary is not None:
        dxf_summary = dxf_summary.copy()
        dxf_summary.insert(0, 'source', 'dxf')
        dxf_summary['selection'] = 'prior'
        dxf_summary['prior_rank'] = np.arange(len(dxf_summary), dtype=int) + 1
        dxf_summary['prior_role'] = 'unused'
        if len(dxf_summary) >= 1:
            dxf_summary.loc[dxf_summary.index[0], 'prior_role'] = 'warm_prior'
        if resolved_mode == 'dual' and len(dxf_summary) >= 2:
            dxf_summary.loc[dxf_summary.index[1], 'prior_role'] = 'hot_prior'
        summary_frames.append(dxf_summary)
    if survey_summary is not None:
        survey_summary = survey_summary.copy()
        survey_summary.insert(0, 'source', 'survey_shp')
        survey_summary['selection'] = 'survey_domain'
        survey_summary['label_column'] = label_col
        summary_frames.append(survey_summary)
    summary = pd.concat(summary_frames, ignore_index=True, sort=False) if len(summary_frames) > 0 else None

    sg = SurveyGeometry(
        survey_poly=survey_geom,
        survey_geom=survey_geom,
        warm_poly=warm_geom,
        hot_poly=hot_geom,
        warm_geom=warm_geom,
        hot_geom=hot_geom,
        valid_center_geom=valid_center_geom,
        valid_center_hot_geom=valid_center_hot_geom,
        cfg=cfg,
        boundary_summary=summary,
        has_hot_prior=has_hot_prior,
        zone_mode=resolved_mode,
    )
    try:
        sg.plot_overlay_geom = load_baseline_overlay_geom_from_shp(getattr(args, "survey_shp", None))
    except Exception:
        sg.plot_overlay_geom = _empty_geom()
    return sg

def build_ops(args: argparse.Namespace) -> OperationalCriteria:
    thresholds = tuple(float(x.strip()) for x in args.pag_thresholds.split(",") if x.strip())
    labels = tuple(s.strip() for s in args.pag_labels.split(",") if s.strip())
    colors = tuple(s.strip() for s in args.pag_colors.split(",") if s.strip())
    if len(labels) != len(thresholds) + 1:
        raise ValueError("pag_labels length must be len(thresholds)+1")
    if len(colors) != len(labels):
        raise ValueError("pag_colors length must match pag_labels")
    return OperationalCriteria(thresholds, args.pag_confidence, labels, colors)

def build_truth_cfg(args: argparse.Namespace) -> TruthConfig:
    return TruthConfig(
        truth_scale=args.truth_scale,
        truth_scale_min=args.truth_scale_min,
        truth_scale_max=args.truth_scale_max,
        truth_scale_random_mode=args.truth_scale_random_mode,
        origin_x=args.truth_origin_x,
        origin_y=args.truth_origin_y,
        n_sources_min=args.truth_sources_min,
        n_sources_max=args.truth_sources_max,
        plume_probability=args.truth_plume_probability,
        prior_center_bias=args.truth_prior_center_bias,
        amp_min=args.truth_amp_min,
        amp_max=args.truth_amp_max,
        value_scale=args.truth_value_scale,
        blur_sigma_cells=args.truth_blur_sigma,
        theta_jitter_deg=args.truth_theta_jitter_deg,
        shape_follow_strength=args.truth_shape_follow_strength,
        shape_fill_fraction=args.truth_shape_fill_fraction,
        prior_softness=args.truth_prior_softness,
        meander_strength=args.truth_meander_strength,
        roughness_strength=args.truth_roughness_strength,
        roughness_sigma_cells=args.truth_roughness_sigma_cells,
        
        lock_hot_anchor=bool(args.truth_lock_hot_anchor),
        hot_anchor_x=float(args.truth_hot_anchor_x),
        hot_anchor_y=float(args.truth_hot_anchor_y),
        meteo_tilt_max_deg=float(args.truth_meteo_tilt_max_deg),
        preserve_hot_core_during_warp=bool(args.truth_preserve_hot_core_during_warp),
    )


def build_baseline_cfg(args: argparse.Namespace) -> BaselineConfig:
    return BaselineConfig(
        warm_spacing_m=args.baseline_warm_spacing_m,
        hot_spacing_m=args.baseline_hot_spacing_m,
        vsp_serpentine_axis=args.baseline_vsp_serpentine_axis,
        vsp_line_tol_m=args.baseline_vsp_line_tol_m,
        vsp_end_corner=args.baseline_vsp_end_corner,
        max_samples=args.baseline_max_samples,
        candidate_source=args.baseline_candidate_source,
        vsp_apply_valid_center=args.baseline_vsp_apply_valid_center,
        cross_phase_prune_enabled=bool(getattr(args, 'baseline_cross_phase_prune_enabled', True)),
        cross_phase_prune_radius_m=float(getattr(args, 'baseline_cross_phase_prune_radius_m', 22.0)),
    )


def build_adaptive_cfg(args: argparse.Namespace) -> AdaptiveConfig:
    return AdaptiveConfig(
        n0=args.gp_n0,
        max_samples=args.gp_max_samples,
        candidate_spacing_m=args.gp_candidate_spacing_m,
        min_spacing_m=args.gp_min_spacing_m,
        epsilon=args.gp_epsilon,
        gamma=args.gp_gamma,
        candidate_stride=args.gp_candidate_stride,
        hot_prior_weight=args.gp_hot_prior_weight,
        gp_restarts=args.gp_restarts,
        max_step_m=args.gp_max_step_m,
        travel_weight=args.gp_travel_weight,
        frontier_weight=args.gp_frontier_weight,
        turn_weight=args.gp_turn_weight,
        global_every=args.gp_global_every,
        order_initial_points=not args.gp_keep_seed_order,
        candidate_source=args.gp_candidate_source,
        vsp_row_tol_m=args.gp_vsp_row_tol_m,
        vsp_serpentine_axis=args.gp_vsp_serpentine_axis,
        vsp_lookahead=args.gp_vsp_lookahead,
        vsp_skip_weight=args.gp_vsp_skip_weight,
        stop_acq=args.gp_stop_acq,
        stop_min_samples=args.gp_stop_min_samples,
        acq_mode=args.gp_acq_mode,
        oracle_top_k=args.gp_oracle_top_k,
        oracle_n_repl=args.gp_oracle_n_repl,
        
        metric_stop_enabled=args.gp_metric_stop,
        metric_stop_zone_accuracy_target=args.gp_metric_zone_accuracy_target,
        metric_stop_hot_recall_target=args.gp_metric_hot_recall_target,
        metric_stop_action_recall_target=args.gp_metric_action_recall_target,
        metric_stop_patience=args.gp_metric_stop_patience,  

    )

def parse_str_list_arg(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]

def _pad_last(values, n, default):
    values = list(values)
    if len(values) == 0:
        values = [default]
    while len(values) < n:
        values.append(values[-1])
    return values[:n]

def build_forecast_cfg(args: argparse.Namespace) -> ForecastConfig:
    return ForecastConfig(
        enabled=(str(args.forecast_model).lower() == "puff"),
        horizon_h=float(args.forecast_horizon_h),
        dt_min=float(args.forecast_dt_min),
        n_members=int(args.forecast_members),
        dose_dep_weight=float(args.forecast_dose_dep_weight),
        source_jitter_m=float(args.forecast_source_jitter_m),
        random_walk_mps=float(args.forecast_random_walk_mps),
        min_std=float(args.forecast_min_std),
    )

def build_source_term(args: argparse.Namespace) -> SourceTerm:
    return SourceTerm(
        start_h=0.0,
        end_h=float(args.source_release_h),
        release_rate=float(args.source_rate),
        dry_dep_vd=float(args.source_dry_dep_vd),
        wet_scavenging=float(args.source_wet_scavenging),
        decay_lambda=float(args.source_decay_lambda),
        source_sigma_m=float(args.source_sigma_m),
    )

def build_meteo_schedule(args: argparse.Namespace) -> List[MeteoStep]:
    dirs = parse_float_list_arg(args.meteo_dirs_deg)
    speeds = parse_float_list_arg(args.meteo_speeds_mps)
    stabs = parse_str_list_arg(args.meteo_stabilities)
    rains = parse_float_list_arg(args.meteo_rain_mmph)

    n = max(len(dirs), len(speeds), len(stabs), len(rains), 1)
    dirs = _pad_last(dirs, n, 250.0)
    speeds = _pad_last(speeds, n, 5.0)
    stabs = _pad_last(stabs, n, "D")
    rains = _pad_last(rains, n, 0.0)

    horizon_h = float(args.forecast_horizon_h)
    block_h = horizon_h / float(n)

    out = []
    for i in range(n):
        out.append(
            MeteoStep(
                t0_h=i * block_h,
                t1_h=(i + 1) * block_h,
                wind_dir_deg=float(dirs[i]),
                wind_speed_mps=float(speeds[i]),
                stability=str(stabs[i]).upper(),
                rain_mmph=float(rains[i]),
            )
        )
    return out

def sample_truth_cfg_for_trial(truth_cfg: TruthConfig, rng: np.random.Generator) -> TruthConfig:
    cfg = copy.deepcopy(truth_cfg)
    mode = str(getattr(cfg, "truth_scale_random_mode", "fixed")).lower()
    lo = getattr(cfg, "truth_scale_min", None)
    hi = getattr(cfg, "truth_scale_max", None)

    if mode == "per_trial" and lo is not None and hi is not None:
        lo_f, hi_f = sorted([float(lo), float(hi)])
        cfg.truth_scale = float(rng.uniform(lo_f, hi_f))

    return cfg

# ---------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------

def parse_float_list_arg(raw: Optional[str]) -> List[float]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, np.ndarray)):
        return [float(x) for x in raw]
    vals = []
    for token in str(raw).split(','):
        token = token.strip()
        if not token:
            continue
        vals.append(float(token))
    return vals

def run_single(
    geom: SurveyGeometry,
    ops: OperationalCriteria,
    truth_cfg: TruthConfig,
    base_cfg: BaselineConfig,
    adapt_cfg: AdaptiveConfig,
    seed: int,
    baseline_vsp_xy: Optional[np.ndarray] = None,
    adaptive_vsp_xy: Optional[np.ndarray] = None,
    source_term: Optional[SourceTerm] = None,
    meteo_steps: Optional[Sequence[MeteoStep]] = None,
    forecast_cfg: Optional[ForecastConfig] = None,
    timing_cfg: Optional[TimingConfig] = None,
    truth_model: str = "legacy",
    truth_weather_mode: str = "none",
    truth_meteo_advect_scale: float = 0.05,
    truth_meteo_blur_scale: float = 0.35,
    weight_cfg: Optional[DecisionWeightConfig] = None,
) -> Tuple[np.ndarray, Dict[str, object], RunResult, RunResult]:
    rng = np.random.default_rng(seed)

    truth_cfg_local = sample_truth_cfg_for_trial(truth_cfg, rng)

    Z_true, forecast_mean_grid, forecast_std_grid, meta = generate_truth_and_forecast(
        geom=geom,
        truth_cfg=truth_cfg_local,
        ops=ops,
        source_term=source_term,
        meteo_steps=meteo_steps,
        fcst_cfg=forecast_cfg,
        rng=rng,
        truth_model=truth_model,
        truth_weather_mode=truth_weather_mode,
        truth_meteo_advect_scale=truth_meteo_advect_scale,
        truth_meteo_blur_scale=truth_meteo_blur_scale,
    )

    baseline = run_baseline(
        geom, ops, base_cfg, Z_true, rng, baseline_vsp_xy,
        forecast_mean_grid=forecast_mean_grid,
        forecast_std_grid=forecast_std_grid,
        timing_cfg=timing_cfg,
    )

    weight_cfg_local = None
    if weight_cfg is not None:
        weight_cfg_local = copy.deepcopy(weight_cfg)
        if float(weight_cfg_local.t_ref) <= 0.0:
            weight_cfg_local.t_ref = float(max(baseline.total_time_min, 1e-12))

    adaptive = run_adaptive(
        geom, ops, adapt_cfg, Z_true, rng, adaptive_vsp_xy,
        forecast_mean_grid=forecast_mean_grid,
        forecast_std_grid=forecast_std_grid,
        timing_cfg=timing_cfg,
        weight_cfg=weight_cfg_local,
    )

    meta = dict(meta)
    meta["truth_scale_used"] = float(truth_cfg_local.truth_scale)
    meta["truth_model_used"] = str(truth_model).lower()

    return Z_true, meta, baseline, adaptive

def build_timing_cfg(args: argparse.Namespace) -> TimingConfig:
    return TimingConfig(
        travel_speed_kmph=float(args.travel_speed_kmph),
        station_time_min=float(args.station_time_min),
    )

def resolve_baseline_csv_path(args: argparse.Namespace) -> Optional[str]:
    path = getattr(args, 'baseline_csv', None)
    if path is not None and str(path).strip():
        return str(path)
    path = getattr(args, 'vsp_csv', None)
    if path is not None and str(path).strip():
        return str(path)
    return None

def load_baseline_plan(args: argparse.Namespace) -> Optional[pd.DataFrame]:
    csv_path = resolve_baseline_csv_path(args)
    if csv_path is None:
        return None
    plan = read_vsp_csv(csv_path, return_frame=True)
    if bool(getattr(args, 'baseline_cross_phase_prune_enabled', True)) and baseline_csv_supports_phase_guided_search(plan):
        clean_df, removed_tables, prune_stats = prune_baseline_all_phases(
            plan,
            radius_m=float(getattr(args, 'baseline_cross_phase_prune_radius_m', 22.0)),
        )
        clean_df.attrs['cross_phase_removed_tables'] = removed_tables
        clean_df.attrs['cross_phase_prune_stats'] = prune_stats
        clean_df.attrs['source_csv_path'] = csv_path
        return clean_df
    plan.attrs['source_csv_path'] = csv_path
    return plan

def save_truth_payload_npz(
    out_path: str,
    Z_true: np.ndarray,
    meta: Dict[str, object],
    *,
    truth_seed: Optional[int] = None,
    survey_seed: Optional[int] = None,
    payload_family: Optional[str] = None,
) -> None:
    forecast_mean_grid = meta.get("forecast_mean_grid", None)
    forecast_std_grid = meta.get("forecast_std_grid", None)
    truth_scale_used = meta.get("truth_scale_used", np.nan)

    kwargs = dict(
        Z_true=np.asarray(Z_true, dtype=float),
        forecast_mean_grid=(
            np.asarray(forecast_mean_grid, dtype=float)
            if forecast_mean_grid is not None else np.array([], dtype=float)
        ),
        forecast_std_grid=(
            np.asarray(forecast_std_grid, dtype=float)
            if forecast_std_grid is not None else np.array([], dtype=float)
        ),
        truth_scale_used=float(truth_scale_used) if np.isfinite(truth_scale_used) else np.nan,
    )

    if truth_seed is not None:
        kwargs["truth_seed"] = np.int64(int(truth_seed))
    if survey_seed is not None:
        kwargs["survey_seed"] = np.int64(int(survey_seed))
    if payload_family is not None:
        kwargs["payload_family"] = np.array(str(payload_family))

    np.savez_compressed(out_path, **kwargs)

def split_master_seed(master_seed: int) -> Tuple[int, int]:
    ss = np.random.SeedSequence(int(master_seed))
    child_truth, child_survey = ss.spawn(2)
    truth_seed = int(child_truth.generate_state(1, dtype=np.uint64)[0])
    survey_seed = int(child_survey.generate_state(1, dtype=np.uint64)[0])
    return truth_seed, survey_seed

def generate_truth_payload_for_export(
    geom: SurveyGeometry,
    ops: OperationalCriteria,
    truth_cfg: TruthConfig,
    *,
    truth_seed: int,
    source_term: Optional[SourceTerm] = None,
    meteo_steps: Optional[Sequence[MeteoStep]] = None,
    forecast_cfg: Optional[ForecastConfig] = None,
    truth_model: str = "legacy",
    truth_weather_mode: str = "none",
    truth_meteo_advect_scale: float = 0.05,
    truth_meteo_blur_scale: float = 0.35,
) -> Tuple[np.ndarray, Dict[str, object]]:
    truth_rng = np.random.default_rng(int(truth_seed))
    truth_cfg_local = sample_truth_cfg_for_trial(truth_cfg, truth_rng)

    Z_true, _forecast_mean_grid, _forecast_std_grid, meta = generate_truth_and_forecast(
        geom=geom,
        truth_cfg=truth_cfg_local,
        ops=ops,
        source_term=source_term,
        meteo_steps=meteo_steps,
        fcst_cfg=forecast_cfg,
        rng=truth_rng,
        truth_model=truth_model,
        truth_weather_mode=truth_weather_mode,
        truth_meteo_advect_scale=truth_meteo_advect_scale,
        truth_meteo_blur_scale=truth_meteo_blur_scale,
    )
    meta = dict(meta)
    meta["truth_scale_used"] = float(truth_cfg_local.truth_scale)
    meta["truth_model_used"] = str(truth_model).lower()
    return Z_true, meta

def run_eval_mode_payload_only(args: argparse.Namespace) -> Dict[str, str]:
    os.makedirs(args.out_dir, exist_ok=True)
    geom = build_geometry(args)
    ops = build_ops(args)
    truth_cfg = build_truth_cfg(args)
    source_term = build_source_term(args)
    forecast_cfg = build_forecast_cfg(args)
    meteo_steps = build_meteo_schedule(args)

    out: Dict[str, str] = {}

    rep_truth_seed, rep_survey_seed = split_master_seed(int(args.seed))
    Z_true0, meta0 = generate_truth_payload_for_export(
        geom, ops, truth_cfg,
        truth_seed=rep_truth_seed,
        source_term=source_term,
        meteo_steps=meteo_steps,
        forecast_cfg=forecast_cfg,
        truth_model=args.truth_model,
        truth_weather_mode=args.truth_weather_mode,
        truth_meteo_advect_scale=args.truth_meteo_advect_scale,
        truth_meteo_blur_scale=args.truth_meteo_blur_scale,
    )
    rep_path = os.path.join(args.out_dir, "payload_rep.npz")
    save_truth_payload_npz(
        rep_path,
        Z_true=Z_true0,
        meta=meta0,
        truth_seed=rep_truth_seed,
        survey_seed=rep_survey_seed,
        payload_family="rep",
    )
    out["payload_rep"] = rep_path

    for i in range(args.eval_mc):
        master_seed_i = int(args.seed) + 1000 + i
        truth_seed_i, survey_seed_i = split_master_seed(master_seed_i)
        Z_true, meta = generate_truth_payload_for_export(
            geom, ops, truth_cfg,
            truth_seed=truth_seed_i,
            source_term=source_term,
            meteo_steps=meteo_steps,
            forecast_cfg=forecast_cfg,
            truth_model=args.truth_model,
            truth_weather_mode=args.truth_weather_mode,
            truth_meteo_advect_scale=args.truth_meteo_advect_scale,
            truth_meteo_blur_scale=args.truth_meteo_blur_scale,
        )
        save_truth_payload_npz(
            os.path.join(args.out_dir, f"payload_run_{i+1:04d}.npz"),
            Z_true=Z_true,
            meta=meta,
            truth_seed=truth_seed_i,
            survey_seed=survey_seed_i,
            payload_family="mc",
        )

    for i in range(args.eval_mc):
        master_seed_i = int(args.seed) + 5000 + i
        truth_seed_i, survey_seed_i = split_master_seed(master_seed_i)
        Z_true, meta = generate_truth_payload_for_export(
            geom, ops, truth_cfg,
            truth_seed=truth_seed_i,
            source_term=source_term,
            meteo_steps=meteo_steps,
            forecast_cfg=forecast_cfg,
            truth_model=args.truth_model,
            truth_weather_mode=args.truth_weather_mode,
            truth_meteo_advect_scale=args.truth_meteo_advect_scale,
            truth_meteo_blur_scale=args.truth_meteo_blur_scale,
        )
        save_truth_payload_npz(
            os.path.join(args.out_dir, f"payload_prefix_{i+1:04d}.npz"),
            Z_true=Z_true,
            meta=meta,
            truth_seed=truth_seed_i,
            survey_seed=survey_seed_i,
            payload_family="prefix",
        )

    return out

def run_eval_mode(args: argparse.Namespace) -> Dict[str, str]:
    if getattr(args, "payload_only", False):
        return run_eval_mode_payload_only(args)
    os.makedirs(args.out_dir, exist_ok=True)
    geom = build_geometry(args)
    ops = build_ops(args)
    truth_cfg = build_truth_cfg(args)
    base_cfg = build_baseline_cfg(args)
    adapt_cfg = build_adaptive_cfg(args)
    source_term = build_source_term(args)
    forecast_cfg = build_forecast_cfg(args)
    meteo_steps = build_meteo_schedule(args)
    timing_cfg = build_timing_cfg(args)
    weight_cfg = build_weight_cfg(args)
    baseline_vsp_xy = load_baseline_plan(args)
    adaptive_vsp_xy = None

    Z_true0, meta0, baseline0, adaptive0 = run_single(
        geom, ops, truth_cfg, base_cfg, adapt_cfg, args.seed,
        baseline_vsp_xy, adaptive_vsp_xy,
        source_term=source_term,
        meteo_steps=meteo_steps,
        forecast_cfg=forecast_cfg,
        timing_cfg=timing_cfg,
        truth_weather_mode=args.truth_weather_mode,
        truth_meteo_advect_scale=args.truth_meteo_advect_scale,
        truth_meteo_blur_scale=args.truth_meteo_blur_scale,
        truth_model=args.truth_model,
        weight_cfg=weight_cfg,    
    )
    
    save_truth_payload_npz(
    os.path.join(args.out_dir, "truth_payload_rep.npz"),
    Z_true=Z_true0,
    meta=meta0,
    )
    
    weight_cfg_fixed = copy.deepcopy(weight_cfg)
    if float(weight_cfg_fixed.t_ref) <= 0.0:
        weight_cfg_fixed.t_ref = float(max(baseline0.total_time_min, 1e-12))
   
    single_artifacts = plot_single_run_separate(
    geom, ops, Z_true0, meta0, baseline0, adaptive0, args.out_dir
    )
    
    single_artifacts.update(
        export_baseline_phaseguided_artifacts(
            args.out_dir,
            baseline0,
            baseline_vsp_xy if isinstance(baseline_vsp_xy, pd.DataFrame) else None,
        )
    )
    
    rows = []
    hot_idx = top_zone_index(ops)
    for i in range(args.eval_mc):
        Z_true, meta, baseline, adaptive = run_single(
            geom, ops, truth_cfg, base_cfg, adapt_cfg, args.seed + 1000 + i,
            baseline_vsp_xy, adaptive_vsp_xy,
            source_term=source_term,
            meteo_steps=meteo_steps,
            forecast_cfg=forecast_cfg,
            timing_cfg=timing_cfg,
            truth_weather_mode=args.truth_weather_mode,
            truth_meteo_advect_scale=args.truth_meteo_advect_scale,
            truth_meteo_blur_scale=args.truth_meteo_blur_scale,
            truth_model=args.truth_model,
            weight_cfg=weight_cfg_fixed,
        )
       
        save_truth_payload_npz(
            os.path.join(args.out_dir, f"truth_payload_run_{i+1:04d}.npz"),
            Z_true=Z_true,
            meta=meta,
        )   
        
        for res in [baseline, adaptive]:
            truth_has_hot = int(np.any(res.truth_zone == hot_idx))
            pred_has_hot = int(np.any(res.pred_zone == hot_idx))
            hot_miss_run = int(bool(truth_has_hot) and not bool(pred_has_hot))
            hot_overcall_run = int((not bool(truth_has_hot)) and bool(pred_has_hot))
            over = overcall_area_rates_from_result(res, ops)
            rows.append({
                'run': i + 1,
                'design': res.name,
                'truth_scale_used': float(meta.get('truth_scale_used', truth_cfg.truth_scale)),
                'zone_accuracy': res.zone_accuracy,
                'hot_recall': res.hot_recall,
                'action_recall': res.action_recall,
                'action_precision': res.action_precision,
                'n_used': res.n_used,
                'path_length_m': res.path_length_m,
                'travel_time_min': res.travel_time_min,
                'station_time_total_min': res.station_time_total_min,
                'total_time_min': res.total_time_min,
                'truth_has_hot': truth_has_hot,
                'pred_has_hot': pred_has_hot,
                'hot_miss_run': hot_miss_run,
                'hot_overcall_run': hot_overcall_run,
                'hot_overcall_area': over['hot_overcall_area'],
                'action_overcall_area': over['action_overcall_area'],
            })

    df = pd.DataFrame(rows)
    summary_csv = os.path.join(args.out_dir, 'mc_summary.csv')
    df.to_csv(summary_csv, index=False)
    agg = df.groupby('design', as_index=False).agg(
        zone_accuracy_mean=('zone_accuracy', 'mean'),
        zone_accuracy_std=('zone_accuracy', 'std'),
        hot_recall_mean=('hot_recall', 'mean'),
        hot_recall_std=('hot_recall', 'std'),
        action_recall_mean=('action_recall', 'mean'),
        action_recall_std=('action_recall', 'std'),
        action_precision_mean=('action_precision', 'mean'),
        action_precision_std=('action_precision', 'std'),
        n_used_mean=('n_used', 'mean'),
        n_used_std=('n_used', 'std'),
        path_length_mean=('path_length_m', 'mean'),
        path_length_std=('path_length_m', 'std'),
        travel_time_mean=('travel_time_min', 'mean'),
        travel_time_std=('travel_time_min', 'std'),
        station_time_mean=('station_time_total_min', 'mean'),
        station_time_std=('station_time_total_min', 'std'),
        total_time_mean=('total_time_min', 'mean'),
        total_time_std=('total_time_min', 'std'),
    )
    agg_csv = os.path.join(args.out_dir, 'mc_summary_aggregated.csv')
    agg.to_csv(agg_csv, index=False)
    fig_path = os.path.join(args.out_dir, 'mc_summary.png')
    plot_mc_summary(df, fig_path)

    weighted_df = aggregate_weighted_summary(df, weight_cfg_fixed)
    weighted_csv = os.path.join(args.out_dir, 'mc_weighted_summary.csv')
    weighted_df.to_csv(weighted_csv, index=False)
    weighted_png = os.path.join(args.out_dir, 'mc_weighted_summary.png')
    plot_weighted_summary(weighted_df, weighted_png)

    # ------------------------------------------------------------
    # Prefix-based adaptive diagnostics
    # ------------------------------------------------------------
    prefix_rows = []

    for i in range(args.eval_mc):
        Z_true, meta, baseline, adaptive = run_single(
            geom, ops, truth_cfg, base_cfg, adapt_cfg, args.seed + 5000 + i,
            baseline_vsp_xy, adaptive_vsp_xy,
            source_term=source_term,
            meteo_steps=meteo_steps,
            forecast_cfg=forecast_cfg,
            timing_cfg=timing_cfg,
            truth_weather_mode=args.truth_weather_mode,
            truth_meteo_advect_scale=args.truth_meteo_advect_scale,
            truth_meteo_blur_scale=args.truth_meteo_blur_scale,
            truth_model=args.truth_model,
            weight_cfg=weight_cfg_fixed,
        )

        pfx = evaluate_prefix_curve(
            geom=geom,
            ops=ops,
            Z_true=Z_true,
            X_obs=adaptive.X_obs,
            y_obs=adaptive.y_obs,
            n_restarts=adapt_cfg.gp_restarts,
            forecast_mean_grid=meta.get("forecast_mean_grid", None),
            forecast_std_grid=meta.get("forecast_std_grid", None),
            timing_cfg=timing_cfg,
            design_name="adaptive",
        )
        pfx["run"] = i + 1
        prefix_rows.append(pfx)

    if len(prefix_rows) > 0:
        prefix_df = pd.concat(prefix_rows, ignore_index=True)
        prefix_mc = summarize_prefix_mc(prefix_df, weight_cfg_fixed)

        prefix_csv = os.path.join(args.out_dir, "adaptive_prefix_curve.csv")
        prefix_df.to_csv(prefix_csv, index=False)

        prefix_mc_csv = os.path.join(args.out_dir, "adaptive_prefix_curve_aggregated.csv")
        prefix_mc.to_csv(prefix_mc_csv, index=False)

        baseline_hot_miss_rate = float(
            (1.0 - df.loc[df["design"] == "baseline", "hot_recall"].astype(float))
            .clip(lower=0.0, upper=1.0)
            .fillna(0.0)
            .mean()
        )
        plot_fatal_miss_curve(
            prefix_mc,
            baseline_hot_miss_rate=baseline_hot_miss_rate,
            out_path=os.path.join(args.out_dir, "fatal_miss_curve.png"),
        )

        baseline_n = float(df.loc[df["design"] == "baseline", "n_used"].mean())
        baseline_hot_recall_mean = float(
            df.loc[df["design"] == "baseline", "hot_recall"].astype(float).mean()
        )
        baseline_hot_miss = float(
            np.clip(1.0 - baseline_hot_recall_mean, 0.0, 1.0)
        )
        baseline_hot_overcall = float(
            df.loc[df["design"] == "baseline", "hot_overcall_area"].astype(float).mean()
        )

        baseline_J = (
            float(weight_cfg_fixed.w_fr) * baseline_hot_miss
            + float(weight_cfg_fixed.w_fh) * baseline_hot_overcall
            + float(weight_cfg_fixed.w_t) * 1.0
            )

        plot_pareto_front(
            prefix_mc,
            baseline_n=baseline_n,
            baseline_J=baseline_J,
            out_path=os.path.join(args.out_dir, "pareto_front.png"),
            x_col="n_used",
            j_col="J",
            point_label="All prefix points",
            title="Pareto Front (MC prefix best-so-far envelope)",
            x_label="Sample count N",
            y_label="Risk Cost J",
        )

        alpha_grid = np.linspace(0.1, 0.9, 17)
        alpha_df = alpha_sensitivity_from_prefix(prefix_mc, alpha_grid=alpha_grid, fh_ratio=weight_cfg.w_fh)
        alpha_df.to_csv(os.path.join(args.out_dir, "alpha_sensitivity.csv"), index=False)
        plot_alpha_sensitivity(alpha_df, os.path.join(args.out_dir, "alpha_sensitivity.png"))

        lambda_t_grid = np.array(parse_float_list_arg(args.lambda_t_grid), dtype=float)
        lambda_fh_grid = np.array(parse_float_list_arg(args.lambda_fh_grid), dtype=float)

        lam_t, lam_fh, Z_nstar, Z_jstar, Z_miss = lambda_heatmaps_from_prefix(
            prefix_mc=prefix_mc,
            lambda_t_grid=lambda_t_grid,
            lambda_fh_grid=lambda_fh_grid,
        )

        rows_lambda = []
        for i_fh, fh in enumerate(lam_fh):
            for j_t, lt in enumerate(lam_t):
                rows_lambda.append({
                    "lambda_t": float(lt),
                    "lambda_fh": float(fh),
                    "N_star": float(Z_nstar[i_fh, j_t]),
                    "J_star": float(Z_jstar[i_fh, j_t]),
                    "fatal_miss_rate_pct": float(Z_miss[i_fh, j_t]),
                })
        lambda_df = pd.DataFrame(rows_lambda)
        lambda_df.to_csv(os.path.join(args.out_dir, "lambda_heatmap_summary.csv"), index=False)

        plot_lambda_heatmap_paperstyle_from_summary(
            lambda_df,
            out_path=os.path.join(args.out_dir, "lambda_heatmap_nstar_paperstyle.png"),
            value_col="N_star",
            title=r"2D Sensitivity of Optimal Sample Count $N^*$",
            cbar_label=r"Optimal sample count $N^*$",
        )

        plot_lambda_heatmap_paperstyle_from_summary(
            lambda_df,
            out_path=os.path.join(args.out_dir, "lambda_heatmap_jstar_paperstyle.png"),
            value_col="J_star",
            title=r"2D Sensitivity of Minimum Risk Cost $J^*$",
            cbar_label=r"Minimum risk cost $J^*$",
        )

        plot_lambda_heatmap_paperstyle_from_summary(
            lambda_df,
            out_path=os.path.join(args.out_dir, "lambda_heatmap_miss_paperstyle.png"),
            value_col="fatal_miss_rate_pct",
            title="2D Sensitivity of Fatal Miss Rate",
            cbar_label="Fatal miss rate [%]",
        )

    if geom.boundary_summary is not None:
        geom.boundary_summary.to_csv(os.path.join(args.out_dir, 'zone_boundary_summary.csv'), index=False)
    return {
        **single_artifacts,
        'summary_csv': summary_csv,
        'agg_csv': agg_csv,
        'summary_png': fig_path,
        'weighted_csv': weighted_csv,
        'weighted_png': weighted_png,
        'risk_trace_csv': os.path.join(args.out_dir, 'adaptive_risk_trace.csv'),
        'pareto_front_png': os.path.join(args.out_dir, 'pareto_front.png'),
    }

def run_paper_figures_mode(args: argparse.Namespace) -> Dict[str, str]:
    os.makedirs(args.out_dir, exist_ok=True)
    geom = build_geometry(args)
    ops = build_ops(args)
    truth_cfg = build_truth_cfg(args)
    base_cfg = build_baseline_cfg(args)
    adapt_cfg = build_adaptive_cfg(args)
    source_term = build_source_term(args)
    forecast_cfg = build_forecast_cfg(args)
    meteo_steps = build_meteo_schedule(args)
    timing_cfg = build_timing_cfg(args)
    weight_cfg = build_weight_cfg(args)

    baseline_vsp_xy = load_baseline_plan(args)
    adaptive_vsp_xy = None

    Z_true, meta, baseline, adaptive = run_single(
        geom, ops, truth_cfg, base_cfg, adapt_cfg, args.seed,
        baseline_vsp_xy, adaptive_vsp_xy,
        source_term=source_term,
        meteo_steps=meteo_steps,
        forecast_cfg=forecast_cfg,
        timing_cfg=timing_cfg,
        truth_weather_mode=args.truth_weather_mode,
        truth_meteo_advect_scale=args.truth_meteo_advect_scale,
        truth_meteo_blur_scale=args.truth_meteo_blur_scale,
        truth_model=args.truth_model,
        weight_cfg=weight_cfg,
    )

    artifacts = plot_single_run_separate(
        geom, ops, Z_true, meta, baseline, adaptive, args.out_dir
    )
    artifacts.update(run_eval_mode(args))
    return artifacts

def plot_footprint_sweep_summary(df: pd.DataFrame, out_path: str) -> None:
    if len(df) == 0:
        return
    metrics = [
        ('zone_accuracy_mean', 'Zone accuracy'),
        ('hot_recall_mean', 'Hot recall'),
        ('n_used_mean', 'Samples used'),
        ('J_norm', 'Weighted objective J (norm.)'),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    for ax, (metric, title) in zip(axes, metrics):
        for design, label, color, marker in [
            ('baseline', 'Baseline', '#b8d3e6', 'o'),
            ('adaptive', 'Adaptive', '#0055a4', 's'),
        ]:
            sub = df[df['design'] == design].sort_values('measurement_diameter_m')
            if len(sub) == 0 or metric not in sub.columns:
                continue
            ax.plot(sub['measurement_diameter_m'], sub[metric], marker=marker, color=color, linewidth=1.8, label=label)
        ax.set_title(title)
        ax.set_xlabel('Measurement diameter [m]')
        if metric in {'zone_accuracy_mean', 'hot_recall_mean'}:
            ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel('Metric value')
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def run_footprint_sweep_mode(args: argparse.Namespace) -> Dict[str, str]:
    diameters = parse_float_list_arg(args.measurement_diameter_sweep_m)
    if len(diameters) == 0:
        diameters = [float(args.measurement_diameter_m)]
    diameters = sorted({float(d) for d in diameters})
    os.makedirs(args.out_dir, exist_ok=True)

    if getattr(args, "payload_only", False):
        artifacts_last: Dict[str, str] = {}
        for d in diameters:
            sub_args = copy.deepcopy(args)
            sub_args.mode = 'eval'
            sub_args.measurement_diameter_m = float(d)
            safe_name = str(d).replace('.', 'p')
            sub_args.out_dir = os.path.join(args.out_dir, f'footprint_{safe_name}_m')
            artifacts_last = run_eval_mode_payload_only(sub_args)
        return artifacts_last

    rows = []
    manifest_rows = []
    for d in diameters:
        sub_args = copy.deepcopy(args)
        sub_args.mode = 'eval'
        sub_args.measurement_diameter_m = float(d)
        safe_name = str(d).replace('.', 'p')
        sub_args.out_dir = os.path.join(args.out_dir, f'footprint_{safe_name}_m')
        artifacts = run_eval_mode(sub_args)

        agg_path = os.path.join(sub_args.out_dir, 'mc_summary_aggregated.csv')
        weighted_path = os.path.join(sub_args.out_dir, 'mc_weighted_summary.csv')
        agg_df = pd.read_csv(agg_path)
        weighted_df = pd.read_csv(weighted_path)
        merged = pd.merge(
            agg_df,
            weighted_df[['design', 'p_hot_miss', 'p_hot_overcall', 'T_n', 'J', 'J_norm']],
            on='design',
            how='left'
        )
        merged.insert(0, 'measurement_diameter_m', float(d))
        rows.append(merged)

        manifest_rows.append({
            'measurement_diameter_m': float(d),
            'out_dir': sub_args.out_dir,
            **artifacts,
        })

    sweep_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    summary_csv = os.path.join(args.out_dir, 'footprint_sweep_summary.csv')
    sweep_df.to_csv(summary_csv, index=False)
    summary_png = os.path.join(args.out_dir, 'footprint_sweep_summary.png')
    plot_footprint_sweep_summary(sweep_df, summary_png)
    manifest_csv = os.path.join(args.out_dir, 'footprint_sweep_manifest.csv')
    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)
    return {
        'summary_csv': summary_csv,
        'summary_png': summary_png,
        'manifest_csv': manifest_csv,
    }


def run_vsp_track_mode(args: argparse.Namespace) -> Dict[str, str]:
    vsp_csv_path = args.vsp_csv if args.vsp_csv else resolve_baseline_csv_path(args)
    if not vsp_csv_path:
        raise ValueError("vsp_track requires --vsp_csv (or a fallback --baseline_csv)")

    os.makedirs(args.out_dir, exist_ok=True)
    geom = build_geometry(args)
    ops = build_ops(args)
    truth_cfg = build_truth_cfg(args)
    adapt_cfg = build_adaptive_cfg(args)
    timing_cfg = build_timing_cfg(args)
    
    source_term = build_source_term(args)
    forecast_cfg = build_forecast_cfg(args)
    meteo_steps = build_meteo_schedule(args)

    vsp_xy = read_vsp_csv(vsp_csv_path)
    pg = prep(geom.valid_center_geom)
    keep = np.fromiter((pg.covers(Point(float(x), float(y))) for x, y in vsp_xy), dtype=bool, count=len(vsp_xy))
    vsp_xy = vsp_xy[keep]
    if len(vsp_xy) == 0:
        raise ValueError("No VSP points remain after warm-zone valid-center filtering")

    rng = np.random.default_rng(args.seed)
    Z_true, forecast_mean_grid, forecast_std_grid, meta = generate_truth_and_forecast(
        geom=geom,
        truth_cfg=truth_cfg,
        ops=ops,
        source_term=source_term,
        meteo_steps=meteo_steps,
        fcst_cfg=forecast_cfg,
        rng=rng,
        truth_model=args.truth_model,
        truth_weather_mode=args.truth_weather_mode,
        truth_meteo_advect_scale=args.truth_meteo_advect_scale,
        truth_meteo_blur_scale=args.truth_meteo_blur_scale,
    )
    weight_cfg = build_weight_cfg(args)
    if float(weight_cfg.t_ref) <= 0.0:
        tm0 = compute_time_metrics(path_length(vsp_xy), len(vsp_xy), timing_cfg)
        weight_cfg.t_ref = float(max(tm0["total_time_min"], 1e-12))

    adaptive = run_adaptive(
        geom, ops, adapt_cfg, Z_true, rng, vsp_xy,
        forecast_mean_grid=forecast_mean_grid,
        forecast_std_grid=forecast_std_grid,
        timing_cfg=timing_cfg,
        weight_cfg=weight_cfg,
    )

    track_png = os.path.join(args.out_dir, "vsp_track.png")
    plot_vsp_track(geom, Z_true, meta, vsp_xy, adaptive, track_png)

    added = adaptive.X_obs[max(int(adapt_cfg.n0), 0):]
    waypoint_csv = os.path.join(args.out_dir, "ai_waypoints.csv")
    pd.DataFrame(added, columns=["x", "y"]).to_csv(waypoint_csv, index=False)
    return {"track_png": track_png, "waypoint_csv": waypoint_csv}



# ---------------------------------------------------------------------
# Integrated payload rebuild helpers
# ---------------------------------------------------------------------


def stage_source(source: str, out_dir: str) -> Path:
    src = Path(source)
    if src.is_dir():
        return src
    if src.is_file() and src.suffix.lower() == ".zip":
        stage_dir = Path(out_dir) / "_staged_payload_source"
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        stage_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(src, "r") as zf:
            zf.extractall(stage_dir)
        return stage_dir
    raise FileNotFoundError(f"payload_source must be a directory or zip: {source}")

def discover_run_dirs(root: Path) -> List[Tuple[str, Path, Optional[float]]]:
    footprint_dirs: List[Tuple[str, Path, Optional[float]]] = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and p.name.startswith("footprint_"):
            m = re.match(r"footprint_(\d+)p(\d+)_m", p.name)
            diam = None
            if m:
                diam = float(f"{m.group(1)}.{m.group(2)}")
            footprint_dirs.append((p.name, p, diam))
    if footprint_dirs:
        return footprint_dirs
    return [(root.name, root, None)]

def load_truth_payload(npz_path: Path) -> Dict[str, object]:
    arr = np.load(npz_path, allow_pickle=True)
    out: Dict[str, object] = {
        "Z_true": np.asarray(arr["Z_true"], dtype=float),
        "forecast_mean_grid": None,
        "forecast_std_grid": None,
        "truth_scale_used": float(arr["truth_scale_used"]) if "truth_scale_used" in arr else np.nan,
        "truth_seed": int(arr["truth_seed"]) if "truth_seed" in arr else -1,
        "survey_seed": int(arr["survey_seed"]) if "survey_seed" in arr else -1,
        "payload_family": str(arr["payload_family"]) if "payload_family" in arr else "",
    }
    if "forecast_mean_grid" in arr and arr["forecast_mean_grid"].size > 0:
        out["forecast_mean_grid"] = np.asarray(arr["forecast_mean_grid"], dtype=float)
    if "forecast_std_grid" in arr and arr["forecast_std_grid"].size > 0:
        out["forecast_std_grid"] = np.asarray(arr["forecast_std_grid"], dtype=float)
    return out

def collect_payload_sets(src_dir: Path) -> Tuple[Path, List[Path], List[Path]]:
    rep_path = src_dir / "payload_rep.npz"
    run_paths = sorted(src_dir.glob("payload_run_*.npz"))
    prefix_paths = sorted(src_dir.glob("payload_prefix_*.npz"))
    if not rep_path.exists():
        raise FileNotFoundError(f"Missing {rep_path}")
    if len(run_paths) == 0:
        raise FileNotFoundError(f"No payload_run_*.npz found in {src_dir}")
    if len(prefix_paths) == 0:
        raise FileNotFoundError(f"No payload_prefix_*.npz found in {src_dir}")
    return rep_path, run_paths, prefix_paths

def export_pareto_front_30_weight_csvs_local(
    prefix_mc: pd.DataFrame,
    baseline_n: float,
    baseline_hot_miss: float,
    baseline_hot_overcall: float,
    out_dir: str,
    *,
    w_fr: float = 1.0,
    w_fh_list: Optional[List[float]] = None,
    w_t_list: Optional[List[float]] = None,
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if len(prefix_mc) == 0:
        return out
    needed = {"n_used", "p_hot_miss", "p_hot_overcall", "T_n"}
    if not needed.issubset(prefix_mc.columns):
        return out

    if w_fh_list is None:
        w_fh_list = [0.0, 0.1, 0.3, 0.5, 1.0]
    if w_t_list is None:
        w_t_list = [0.1, 0.3, 0.5, 1.0, 2.0, 4.0]

    prefix_base = (
        prefix_mc[["n_used", "p_hot_miss", "p_hot_overcall", "T_n"]]
        .copy()
        .dropna(subset=["n_used", "p_hot_miss", "p_hot_overcall", "T_n"])
        .sort_values("n_used")
    )
    if len(prefix_base) == 0:
        return out

    curve_rows: List[pd.DataFrame] = []
    baseline_rows: List[Dict[str, float]] = []
    for w_fh in w_fh_list:
        for w_t in w_t_list:
            sub = prefix_base.copy()
            sub["w_fr"] = float(w_fr)
            sub["w_fh"] = float(w_fh)
            sub["w_t"] = float(w_t)
            sub["J_combo"] = (
                float(w_fr) * sub["p_hot_miss"]
                + float(w_fh) * sub["p_hot_overcall"]
                + float(w_t) * sub["T_n"]
            )
            curve_rows.append(sub)
            baseline_rows.append({
                "w_fr": float(w_fr),
                "w_fh": float(w_fh),
                "w_t": float(w_t),
                "baseline_n": float(baseline_n),
                "baseline_J": (
                    float(w_fr) * float(baseline_hot_miss)
                    + float(w_fh) * float(baseline_hot_overcall)
                    + float(w_t) * 1.0
                ),
            })

    curves_df = pd.concat(curve_rows, ignore_index=True)
    baseline_df = pd.DataFrame(baseline_rows)
    curves_csv = os.path.join(out_dir, "pareto_front_30_weight_curves.csv")
    baseline_csv = os.path.join(out_dir, "pareto_front_30_weight_baseline.csv")
    curves_df.to_csv(curves_csv, index=False)
    baseline_df.to_csv(baseline_csv, index=False)
    out["pareto_30_curves_csv"] = curves_csv
    out["pareto_30_baseline_csv"] = baseline_csv
    return out

def build_context_for_rebuild(args_obj: argparse.Namespace):
    geom = build_geometry(args_obj)
    ops = build_ops(args_obj)
    truth_cfg = build_truth_cfg(args_obj)
    base_cfg = build_baseline_cfg(args_obj)
    source_term = build_source_term(args_obj)
    forecast_cfg = build_forecast_cfg(args_obj)
    meteo_steps = build_meteo_schedule(args_obj)
    timing_cfg = build_timing_cfg(args_obj)
    weight_cfg = build_weight_cfg(args_obj)
    baseline_vsp_xy = load_baseline_plan(args_obj)
    return geom, ops, truth_cfg, base_cfg, source_term, forecast_cfg, meteo_steps, timing_cfg, weight_cfg, baseline_vsp_xy

def run_truth_baseline_adaptive_from_payload(args_obj: argparse.Namespace, payload: Dict[str, object]):
    geom, ops, truth_cfg, base_cfg, source_term, forecast_cfg, meteo_steps, timing_cfg, weight_cfg, baseline_vsp_xy = build_context_for_rebuild(args_obj)
    adapt_cfg = build_adaptive_cfg(args_obj)
    adaptive_vsp_xy = None

    survey_seed = int(payload.get("survey_seed", -1))
    if survey_seed < 0:
        raise ValueError("payload does not contain a valid survey_seed")
    rng = np.random.default_rng(survey_seed)

    Z_true = np.asarray(payload["Z_true"], dtype=float)
    forecast_mean_grid = payload.get("forecast_mean_grid", None)
    forecast_std_grid = payload.get("forecast_std_grid", None)
    meta = {
        "truth_scale_used": float(payload.get("truth_scale_used", np.nan)),
        "truth_model_used": str(args_obj.truth_model).lower(),
        "pred_survey_poly": geom.survey_poly,
        "pred_warm_poly": geom.warm_poly,
        "pred_hot_poly": geom.hot_poly,
        "has_hot_prior": geom.has_hot_prior,
        "forecast_mean_grid": forecast_mean_grid,
        "forecast_std_grid": forecast_std_grid,
    }

    baseline = run_baseline(
        geom, ops, base_cfg, Z_true, rng, baseline_vsp_xy,
        forecast_mean_grid=forecast_mean_grid,
        forecast_std_grid=forecast_std_grid,
        timing_cfg=timing_cfg,
    )

    weight_cfg_local = copy.deepcopy(weight_cfg)
    if float(weight_cfg_local.t_ref) <= 0.0:
        weight_cfg_local.t_ref = float(max(baseline.total_time_min, 1e-12))

    adaptive = run_adaptive(
        geom, ops, adapt_cfg, Z_true, rng, adaptive_vsp_xy,
        forecast_mean_grid=forecast_mean_grid,
        forecast_std_grid=forecast_std_grid,
        timing_cfg=timing_cfg,
        weight_cfg=weight_cfg_local,
    )
    
    return {
        "geom": geom,
        "ops": ops,
        "timing_cfg": timing_cfg,
        "weight_cfg": weight_cfg_local,
        "baseline_vsp_xy": baseline_vsp_xy,
        "Z_true": Z_true,
        "meta": meta,
        "baseline": baseline,
        "adaptive": adaptive,
    }

def row_from_result(res: RunResult, ops: OperationalCriteria, meta: Dict[str, object], run_idx: int) -> Dict[str, object]:
    hot_idx = top_zone_index(ops)
    over = overcall_area_rates_from_result(res, ops)
    truth_has_hot = int(np.any(res.truth_zone == hot_idx))
    pred_has_hot = int(np.any(res.pred_zone == hot_idx))
    hot_miss_run = int(bool(truth_has_hot) and not bool(pred_has_hot))
    hot_overcall_run = int((not bool(truth_has_hot)) and bool(pred_has_hot))
    return {
        "run": run_idx,
        "design": res.name,
        "truth_scale_used": float(meta.get("truth_scale_used", np.nan)),
        "zone_accuracy": float(res.zone_accuracy),
        "hot_recall": float(res.hot_recall),
        "action_recall": float(res.action_recall),
        "action_precision": float(res.action_precision),
        "n_used": int(res.n_used),
        "path_length_m": float(res.path_length_m),
        "travel_time_min": float(res.travel_time_min),
        "station_time_total_min": float(res.station_time_total_min),
        "total_time_min": float(res.total_time_min),
        "truth_has_hot": truth_has_hot,
        "pred_has_hot": pred_has_hot,
        "hot_miss_run": hot_miss_run,
        "hot_overcall_run": hot_overcall_run,
        "hot_overcall_area": float(over.get("hot_overcall_area", np.nan)),
        "action_overcall_area": float(over.get("action_overcall_area", np.nan)),
    }

def regenerate_prefix_from_payloads(args_obj: argparse.Namespace, prefix_paths: Sequence[Path], out_dir: Path, weight_cfg_fixed: DecisionWeightConfig):
    prefix_rows = []
    adapt_cfg = build_adaptive_cfg(args_obj)
    for i, p in enumerate(prefix_paths, start=1):
        payload = load_truth_payload(p)
        ctx = run_truth_baseline_adaptive_from_payload(args_obj, payload)
        adaptive = ctx["adaptive"]
        pfx = evaluate_prefix_curve(
            geom=ctx["geom"],
            ops=ctx["ops"],
            Z_true=ctx["Z_true"],
            X_obs=adaptive.X_obs,
            y_obs=adaptive.y_obs,
            n_restarts=adapt_cfg.gp_restarts,
            forecast_mean_grid=ctx["meta"].get("forecast_mean_grid", None),
            forecast_std_grid=ctx["meta"].get("forecast_std_grid", None),
            timing_cfg=ctx["timing_cfg"],
            design_name="adaptive",
        )
        pfx["run"] = i
        prefix_rows.append(pfx)

    prefix_df = pd.concat(prefix_rows, ignore_index=True)
    prefix_mc = summarize_prefix_mc(prefix_df, weight_cfg_fixed)
    prefix_df.to_csv(out_dir / "adaptive_prefix_curve.csv", index=False)
    prefix_mc.to_csv(out_dir / "adaptive_prefix_curve_aggregated.csv", index=False)
    return prefix_df, prefix_mc

def regenerate_all_artifacts(args_obj: argparse.Namespace, src_dir: Path, out_dir: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    out_dir.mkdir(parents=True, exist_ok=True)

    rep_path, run_paths, prefix_paths = collect_payload_sets(src_dir)

    rep_payload = load_truth_payload(rep_path)
    ctx0 = run_truth_baseline_adaptive_from_payload(args_obj, rep_payload)
    geom = ctx0["geom"]
    ops = ctx0["ops"]
    Z_true0 = ctx0["Z_true"]
    meta0 = ctx0["meta"]
    baseline0 = ctx0["baseline"]
    adaptive0 = ctx0["adaptive"]
    weight_cfg_fixed = copy.deepcopy(ctx0["weight_cfg"])

    plot_single_run_separate(geom, ops, Z_true0, meta0, baseline0, adaptive0, str(out_dir))
    out["truth_field_png"] = str(out_dir / "truth_field.png")
    out["baseline_png"] = str(out_dir / "baseline.png")
    out["adaptive_png"] = str(out_dir / "adaptive.png")

    if adaptive0.diag is not None and "risk_trace" in adaptive0.diag:
        risk_df = pd.DataFrame(adaptive0.diag["risk_trace"]).copy()
        risk_df.to_csv(out_dir / "adaptive_risk_trace.csv", index=False)
        out["adaptive_risk_trace_csv"] = str(out_dir / "adaptive_risk_trace.csv")

    rows: List[Dict[str, object]] = []
    for i, p in enumerate(run_paths, start=1):
        payload_i = load_truth_payload(p)
        ctx_i = run_truth_baseline_adaptive_from_payload(args_obj, payload_i)
        rows.append(row_from_result(ctx_i["baseline"], ops, ctx_i["meta"], i))
        rows.append(row_from_result(ctx_i["adaptive"], ops, ctx_i["meta"], i))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "mc_summary.csv", index=False)
    out["mc_summary_csv"] = str(out_dir / "mc_summary.csv")

    agg = df.groupby("design", as_index=False).agg(
        zone_accuracy_mean=("zone_accuracy", "mean"),
        zone_accuracy_std=("zone_accuracy", "std"),
        hot_recall_mean=("hot_recall", "mean"),
        hot_recall_std=("hot_recall", "std"),
        action_recall_mean=("action_recall", "mean"),
        action_recall_std=("action_recall", "std"),
        action_precision_mean=("action_precision", "mean"),
        action_precision_std=("action_precision", "std"),
        n_used_mean=("n_used", "mean"),
        n_used_std=("n_used", "std"),
        path_length_mean=("path_length_m", "mean"),
        path_length_std=("path_length_m", "std"),
        travel_time_mean=("travel_time_min", "mean"),
        travel_time_std=("travel_time_min", "std"),
        station_time_mean=("station_time_total_min", "mean"),
        station_time_std=("station_time_total_min", "std"),
        total_time_mean=("total_time_min", "mean"),
        total_time_std=("total_time_min", "std"),
    )
    agg.to_csv(out_dir / "mc_summary_aggregated.csv", index=False)
    out["mc_summary_aggregated_csv"] = str(out_dir / "mc_summary_aggregated.csv")

    weighted_df = aggregate_weighted_summary(df, weight_cfg_fixed)
    weighted_df.to_csv(out_dir / "mc_weighted_summary.csv", index=False)
    out["mc_weighted_summary_csv"] = str(out_dir / "mc_weighted_summary.csv")

    prefix_df, prefix_mc = regenerate_prefix_from_payloads(args_obj, prefix_paths, out_dir, weight_cfg_fixed)
    out["adaptive_prefix_curve_csv"] = str(out_dir / "adaptive_prefix_curve.csv")
    out["adaptive_prefix_curve_aggregated_csv"] = str(out_dir / "adaptive_prefix_curve_aggregated.csv")

    lam_t = np.array(parse_float_list_arg(args_obj.lambda_t_grid), dtype=float)
    lam_fh = np.array(parse_float_list_arg(args_obj.lambda_fh_grid), dtype=float)
    lam_t, lam_fh, Z_nstar, Z_jstar, Z_miss = lambda_heatmaps_from_prefix(
        prefix_mc=prefix_mc,
        lambda_t_grid=lam_t,
        lambda_fh_grid=lam_fh,
    )

    rows_lambda = []
    for i_fh, fh in enumerate(lam_fh):
        for j_t, lt in enumerate(lam_t):
            rows_lambda.append({
                "lambda_t": float(lt),
                "lambda_fh": float(fh),
                "N_star": float(Z_nstar[i_fh, j_t]),
                "J_star": float(Z_jstar[i_fh, j_t]),
                "fatal_miss_rate_pct": float(Z_miss[i_fh, j_t]),
            })
    lambda_df = pd.DataFrame(rows_lambda)
    lambda_df.to_csv(out_dir / "lambda_heatmap_summary.csv", index=False)
    out["lambda_heatmap_summary_csv"] = str(out_dir / "lambda_heatmap_summary.csv")

    baseline_n = float(df.loc[df["design"] == "baseline", "n_used"].mean())
    baseline_hot_recall_mean = float(df.loc[df["design"] == "baseline", "hot_recall"].astype(float).mean())
    baseline_hot_miss = float(np.clip(1.0 - baseline_hot_recall_mean, 0.0, 1.0))
    baseline_hot_overcall = float(df.loc[df["design"] == "baseline", "hot_overcall_area"].astype(float).mean())
    out.update(
        export_pareto_front_30_weight_csvs_local(
            prefix_mc,
            baseline_n=baseline_n,
            baseline_hot_miss=baseline_hot_miss,
            baseline_hot_overcall=baseline_hot_overcall,
            out_dir=str(out_dir),
            w_fr=float(weight_cfg_fixed.w_fr),
            w_fh_list=[0.0, 0.1, 0.3, 0.5, 1.0],
            w_t_list=[0.1, 0.3, 0.5, 1.0, 2.0, 4.0],
        )
    )
    return out

def run_rebuild_stage(args: argparse.Namespace) -> Dict[str, str]:
    payload_source = args.payload_source if args.payload_source else args.out_dir
    staged_root = stage_source(payload_source, args.out_dir)
    runs = discover_run_dirs(staged_root)

    artifacts_all: Dict[str, str] = {}
    multi = len(runs) > 1
    for name, src_dir, diam in runs:
        args_sub = copy.deepcopy(args)
        if diam is not None:
            args_sub.measurement_diameter_m = float(diam)
        args_sub.mode = "eval"
        out_subdir = Path(args.out_dir) / (name if multi else "")
        out_subdir.mkdir(parents=True, exist_ok=True)
        arts = regenerate_all_artifacts(args_sub, src_dir, out_subdir)
        if multi:
            for k, v in arts.items():
                artifacts_all[f"{name}_{k}"] = v
        else:
            artifacts_all.update(arts)
    return artifacts_all

def run_payload_stage(args: argparse.Namespace) -> Dict[str, str]:
    args_payload = copy.deepcopy(args)
    args_payload.payload_only = True
    if args_payload.mode == "eval":
        return run_eval_mode(args_payload)
    if args_payload.mode == "footprint_sweep":
        return run_footprint_sweep_mode(args_payload)
    raise ValueError("run_stage=payload_only/full_pipeline only supports --mode eval or --mode footprint_sweep")

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hybrid operational zoning simulator: DXF contamination priors, shapefile survey boundary, baseline CSV routing, unchanged GP adaptive sampling")
    p.add_argument("--mode", choices=["paper_figures", "eval", "vsp_track", "footprint_sweep"], default="eval")
    p.add_argument("--zone_dxf", type=str, required=True, help="DXF file used for the contamination prior / legacy warm-hot geometry.")
    p.add_argument("--survey_shp", "--zone_shp", dest="survey_shp", type=str, default=None, help="Survey boundary .shp file. If omitted, the hard survey domain falls back to a buffered DXF warm prior.")
    p.add_argument("--survey_shx", "--zone_shx", dest="survey_shx", type=str, default=None, help="Optional explicit path to the .shx companion file for --survey_shp.")
    p.add_argument("--survey_dbf", "--zone_dbf", dest="survey_dbf", type=str, default=None, help="Optional explicit path to the .dbf companion file for --survey_shp.")
    p.add_argument("--shp_label_col", type=str, default=None, help="Attribute column in the survey shapefile used for area selection. Default: auto-detect.")
    p.add_argument("--shp_survey_labels", type=str, default="all", help="Comma-separated shapefile labels used as the hard survey domain. Default: all polygons.")
    p.add_argument("--shp_warm_labels", type=str, default="all", help="Retained for compatibility. Warm prior selection now comes from --zone_dxf in hybrid mode.")
    p.add_argument("--shp_hot_labels", type=str, default=None, help="Retained for compatibility. Hot prior selection now comes from --zone_dxf in hybrid mode.")
    p.add_argument("--zone_mode", choices=["auto", "single", "dual"], default="auto", help="How to interpret the DXF prior polygons: single ignores a hot prior, dual requires warm+hot, auto uses a hot prior when present.")
    p.add_argument("--baseline_csv", type=str, default=None, help="Baseline station plan CSV. This is the primary baseline input.")
    p.add_argument("--vsp_csv", type=str, default=None, help="Optional CSV for GP/VSP experiments such as --mode vsp_track. If --baseline_csv is omitted, baseline will fall back to this file.")
    p.add_argument("--out_dir", type=str, default="ops_hybrid_outputs")
    p.add_argument("--seed", type=int, default=None, help="Master random seed. Omit to auto-generate a different seed each script run.")

    p.add_argument("--grid_n", type=int, default=201)
    p.add_argument("--background", type=float, default=0.02)
    p.add_argument("--meas_sigma", type=float, default=0.05)
    p.add_argument("--measurement_diameter_m", type=float, default=22.0)
    p.add_argument("--measurement_diameter_sweep_m", type=str, default="", help="Comma-separated footprint diameters for mode=footprint_sweep, e.g. 5,10,15,22")
    p.add_argument("--measurement_model", choices=["point", "disk_avg"], default="disk_avg")
    p.add_argument("--truth_model", choices=["legacy", "puff"], default="legacy", help="legacy: irregular truth field. puff: truth itself generated from puff ensemble")

    p.add_argument("--forecast_horizon_h", type=float, default=96.0)
    p.add_argument("--forecast_dt_min", type=float, default=30.0)
    p.add_argument("--forecast_members", type=int, default=24)
    p.add_argument("--forecast_dose_dep_weight", type=float, default=0.35)
    p.add_argument("--forecast_source_jitter_m", type=float, default=20.0)
    p.add_argument("--forecast_random_walk_mps", type=float, default=0.25)
    p.add_argument("--forecast_min_std", type=float, default=0.05)
    p.add_argument("--forecast_model", choices=["none", "puff"], default="none")

    p.add_argument("--meteo_dirs_deg", type=str, default="260,245,230,250", help="Comma-separated wind directions (meteorological, from-direction)")
    p.add_argument("--meteo_speeds_mps", type=str, default="4.5,5.0,4.0,5.5")
    p.add_argument("--meteo_stabilities", type=str, default="D,D,E,D")
    p.add_argument("--meteo_rain_mmph", type=str, default="0,0,0.5,0")

    p.add_argument("--survey_buffer_m", type=float, default=45.0, help="Fallback buffer around the DXF warm prior [m] when --survey_shp is omitted. Ignored when a survey shapefile is provided.")

    p.add_argument("--source_release_h", type=float, default=1.0)
    p.add_argument("--source_rate", type=float, default=1.0)
    p.add_argument("--source_dry_dep_vd", type=float, default=0.002)
    p.add_argument("--source_wet_scavenging", type=float, default=0.03)
    p.add_argument("--source_decay_lambda", type=float, default=0.0)
    p.add_argument("--source_sigma_m", type=float, default=12.0)
    p.add_argument("--pag_thresholds", type=str, default="1.0,5.0")
    p.add_argument("--pag_labels", type=str, default="Cold,Warm,Hot")
    p.add_argument("--pag_colors", type=str, default="#9fd39f,#f4c06a,#e96b63")
    p.add_argument("--pag_confidence", type=float, default=0.975)

    p.add_argument("--truth_scale", type=float, default=0.6)
    p.add_argument("--truth_scale_min", type=float, default=None, help="Lower bound for random truth_scale. Used only when --truth_scale_random_mode is per_run or per_trial.")
    p.add_argument("--truth_scale_max", type=float, default=None, help="Upper bound for random truth_scale. Used only when --truth_scale_random_mode is per_run or per_trial.")
    p.add_argument("--truth_scale_random_mode", choices=["fixed", "per_run", "per_trial"], default="fixed", help="fixed: always use --truth_scale. per_run: draw one random truth_scale for the whole script execution. per_trial: draw a new truth_scale for each MC trial / run_single call.")
    p.add_argument("--truth_origin_x", type=float, default=0.0)
    p.add_argument("--truth_origin_y", type=float, default=0.0)
    p.add_argument("--truth_sources_min", type=int, default=1)
    p.add_argument("--truth_sources_max", type=int, default=2)
    p.add_argument("--truth_plume_probability", type=float, default=1.0)
    p.add_argument("--truth_prior_center_bias", type=float, default=4.0)
    p.add_argument("--truth_amp_min", type=float, default=3.6)
    p.add_argument("--truth_amp_max", type=float, default=5.6)
    p.add_argument("--truth_value_scale", type=float, default=0.5,help="Global multiplier on excess contamination above background. 1.0 keeps the current field; 0.6 scales every non-background value to 60%%.")
    p.add_argument("--truth_blur_sigma", type=float, default=2.0)
    p.add_argument("--truth_theta_jitter_deg", type=float, default=0)
    p.add_argument("--truth_shape_follow_strength", type=float, default=0.75)
    p.add_argument("--truth_shape_fill_fraction", type=float, default=0.10)
    p.add_argument("--truth_prior_softness", type=float, default=0.45)
    p.add_argument("--truth_meander_strength", type=float, default=0.20)
    p.add_argument("--truth_roughness_strength", type=float, default=0.20)
    p.add_argument("--truth_roughness_sigma_cells", type=float, default=6.0)
    p.add_argument("--truth_weather_mode", choices=["none", "warp"], default="none")
    p.add_argument("--truth_meteo_advect_scale", type=float, default=0.05)
    p.add_argument("--truth_meteo_blur_scale", type=float, default=0.35)
    p.add_argument("--truth_hot_anchor_x", type=float, default=0.0)
    p.add_argument("--truth_hot_anchor_y", type=float, default=0.0)
    p.add_argument("--truth_meteo_tilt_max_deg", type=float, default=5.0, help="Maximum weather-induced warm-plume tilt from the base axis")
    p.add_argument("--truth_lock_hot_anchor", action=argparse.BooleanOptionalAction, default=True, help="Lock truth hot core around (truth_hot_anchor_x, truth_hot_anchor_y)")
    p.add_argument("--truth_preserve_hot_core_during_warp", action=argparse.BooleanOptionalAction, default=True, help="Keep hot core fixed while warping the warm tail" )

    p.add_argument("--baseline_warm_spacing_m", type=float, default=19.0)
    p.add_argument("--baseline_hot_spacing_m", type=float, default=13.0)
    p.add_argument("--baseline_candidate_source", choices=["auto", "baseline_csv", "vsp", "hex", "grid"], default="auto", help="Baseline point source. auto prefers --baseline_csv (or --vsp_csv fallback); GP/adaptive logic stays unchanged.")
    p.add_argument("--baseline_vsp_serpentine_axis", choices=["none", "row", "col"], default="col", help="Serpentine direction for baseline ordering. In grid mode this also controls fixed sweep ordering.")
    p.add_argument("--baseline_vsp_line_tol_m", type=float, default=22.0, help="Line/column grouping tolerance for VSP serpentine ordering.")
    p.add_argument("--baseline_vsp_end_corner", choices=["upper_right", "upper_left", "lower_right", "lower_left"], default="upper_right")
    p.add_argument("--baseline_vsp_apply_valid_center", action="store_true", help="When baseline candidate_source=vsp/baseline_csv, filter raw stations by the footprint-valid center region. Default is off, so CSV stations remain fixed as footprint changes.")
    p.add_argument("--baseline_max_samples", type=int, default=0, help="Limit baseline to the first N ordered points. 0 means use all baseline points.")

    p.add_argument("--gp_n0", type=int, default=3)
    p.add_argument("--gp_max_samples", type=int, default=80, help="Number of adaptive samples added after the initial gp_n0 seed points.")
    p.add_argument("--gp_candidate_spacing_m", type=float, default=20.0)
    p.add_argument("--gp_min_spacing_m", type=float, default=22.0)
    p.add_argument("--gp_epsilon", type=float, default=0.05)
    p.add_argument("--gp_gamma", type=float, default=0.01)
    p.add_argument("--gp_candidate_stride", type=int, default=1)
    p.add_argument("--gp_hot_prior_weight", type=float, default=0.10)
    p.add_argument("--gp_restarts", type=int, default=2)
    p.add_argument("--gp_max_step_m", type=float, default=34.0)
    p.add_argument("--gp_travel_weight", type=float, default=0.30)
    p.add_argument("--gp_frontier_weight", type=float, default=0.20)
    p.add_argument("--gp_turn_weight", type=float, default=0.08)
    p.add_argument("--gp_global_every", type=int, default=6)
    p.add_argument("--gp_keep_seed_order", action="store_true")
    p.add_argument("--gp_candidate_source", choices=["auto", "grid", "vsp"], default="grid", help="Adaptive GP candidate domain. Use grid for fair GP-vs-VSP comparison; use vsp only for route-constrained experiments. auto behaves like grid.")
    p.add_argument("--gp_use_vsp_init", action="store_true", help="Seed adaptive GP with provided VSP points. Off by default so GP remains original during comparison.")
    p.add_argument("--gp_vsp_row_tol_m", type=float, default=8.0, help="Serpentine line tolerance in meters. In row mode it groups by y; in col mode it groups by x.")
    p.add_argument("--gp_vsp_serpentine_axis", choices=["row", "col"], default="row")
    p.add_argument("--gp_vsp_lookahead", type=int, default=10)
    p.add_argument("--gp_vsp_skip_weight", type=float, default=0.18)
    p.add_argument("--gp_stop_acq", type=float, default=-1.0, help="Early-stop adaptive sampling when the best acquisition score falls below this value. Negative disables stopping.")
    p.add_argument("--gp_stop_min_samples", type=int, default=0, help="Minimum adaptive samples before early stopping can activate. 0 means use gp_n0.")
    p.add_argument("--gp_acq_mode", choices=["posterior", "oracle_onestep"], default="posterior")
    p.add_argument("--gp_oracle_top_k", type=int, default=25)
    p.add_argument("--gp_oracle_n_repl", type=int, default=3)
    p.add_argument("--gp_metric_stop", action=argparse.BooleanOptionalAction, default=False, help="Stop adaptive sampling when current design metrics meet all specified targets.")
    p.add_argument("--gp_metric_zone_accuracy_target", type=float, default=0.84)
    p.add_argument("--gp_metric_hot_recall_target", type=float, default=0.74)
    p.add_argument("--gp_metric_action_recall_target", type=float, default=0.80)
    p.add_argument("--gp_metric_stop_patience", type=int, default=2,help="Number of consecutive metric-stop checks that must pass before stopping.")

    p.add_argument("--w_t", type=float, default=1.0, help="Weight on normalized sampling effort T_tilde")
    p.add_argument("--w_fr", type=float, default=1.0, help="Weight on hot-miss probability")
    p.add_argument("--w_fh", type=float, default=0.30, help="Weight on hot overcall probability")
    p.add_argument("--t_ref", type=float, default=0.0, help="Reference sample count for T_tilde = n_avg / t_ref. 0 means use baseline mean sample count.")

    p.add_argument("--eval_mc", type=int, default=20)
    p.add_argument("--payload_only", action="store_true", help="For eval/footprint_sweep, export only payload_rep.npz, payload_run_*.npz, and payload_prefix_*.npz.")
    p.add_argument("--run_stage", choices=["legacy", "payload_only", "rebuild_only", "full_pipeline"], default="legacy", help="legacy: original behavior. payload_only: export payloads only. rebuild_only: read payloads and rebuild final PNG/CSV outputs. full_pipeline: run payload export then rebuild in one command.")
    p.add_argument("--payload_source", type=str, default=None, help="Directory or zip containing payload_rep.npz, payload_run_*.npz, and payload_prefix_*.npz. Used by --run_stage rebuild_only. Defaults to --out_dir.")
    p.add_argument("--travel_speed_kmph", type=float, default=15.0, help="Travel speed between consecutive stations in km/h. Coordinates are interpreted in meters.")
    p.add_argument("--station_time_min", type=float, default=1.0,help="Fixed reconnaissance / dwell time per visited station in minutes.")
    p.add_argument("--lambda_t_grid", type=str, default="0.05,0.1,0.2,0.4,0.8", help="Comma-separated lambda_t grid for lambda heatmap")
    p.add_argument("--lambda_fh_grid", type=str, default="0.05,0.1,0.2,0.3,0.5", help="Comma-separated lambda_FH grid for lambda heatmap")
    
    return p

def main() -> None:
    args = build_argparser().parse_args()

    if args.seed is None:
        args.seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])
        print(f"[info] Auto-generated master seed: {args.seed}")
    else:
        print(f"[info] Using master seed: {args.seed}")
    if args.truth_scale_random_mode in ("per_run", "per_trial"):
        if args.truth_scale_min is None or args.truth_scale_max is None:
            raise ValueError("--truth_scale_min and --truth_scale_max are required when --truth_scale_random_mode is per_run or per_trial")
        lo, hi = sorted([float(args.truth_scale_min), float(args.truth_scale_max)])
        if args.truth_scale_random_mode == "per_run":
            rng_scale = np.random.default_rng(args.seed + 424242)
            args.truth_scale = float(rng_scale.uniform(lo, hi))
            args.truth_scale_random_mode = "fixed"
            print(f"[info] Randomized truth_scale for this execution: {args.truth_scale:.4f}")
        else:
            print(f"[info] truth_scale will be re-sampled per trial from U({lo:.4f}, {hi:.4f})")

    if args.run_stage == "legacy":
        if args.mode == "eval":
            artifacts = run_eval_mode(args)
        elif args.mode == "vsp_track":
            artifacts = run_vsp_track_mode(args)
        elif args.mode == "footprint_sweep":
            artifacts = run_footprint_sweep_mode(args)
        else:
            artifacts = run_paper_figures_mode(args)
    elif args.run_stage == "payload_only":
        artifacts = run_payload_stage(args)
    elif args.run_stage == "rebuild_only":
        artifacts = run_rebuild_stage(args)
    elif args.run_stage == "full_pipeline":
        payload_arts = run_payload_stage(args)
        rebuild_args = copy.deepcopy(args)
        if rebuild_args.payload_source is None:
            rebuild_args.payload_source = args.out_dir
        rebuild_arts = run_rebuild_stage(rebuild_args)
        artifacts = {**payload_arts, **rebuild_arts}
    else:
        raise ValueError(f"Unknown run_stage: {args.run_stage}")

    print("Saved artifacts:")
    for k, v in artifacts.items():
        print(f"  {k:>28s} : {v}")


if __name__ == "__main__":
    main()