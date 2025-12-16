import io
import time
import contextlib
from typing import Any, Dict, List, Tuple, Optional, Union

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import openseespy.opensees as ops

# ============================================================
#  GEOMETRIA / VINCOLI
# ============================================================
L = 4.0
H = 6.0

CORDOLI_Y = [(2.7, 3.0), (5.7, 6.0)]
MARGIN = 0.30
PIER_MIN = 0.30

Rect = Tuple[float, float, float, float]  # (x1,x2,y1,y2)

# ============================================================
#  DEFAULTS
# ============================================================
DEFAULTS = {
    "stress": 0,
    "verbose": 0,

    "max_dx": 0.30,
    "max_dy": 0.30,
    "dU": 0.0006,
    "max_steps": 100,
    "target_mm": 15.0,

    "Ptot": 100e3,
    "testTol": 1.0e-4,
    "testIter": 15,
    "algo": "Newton",
    "system": "BandGeneral",
    "numberer": "RCM",
    "constraints": "Plain",

    "n_bins_x": 4,
    "n_bins_y": 12,

    "grid_zx": 3,
    "grid_zy": 5,
    "min_solid_ratio": 0.85,
    "greedy_topN": 25,

    # selezione rinforzo
    "frcm_mode": "intersection",
    "frcm_min_overlap": 0.02,

    # ✅ equivalente muratura + FRCM (BOOST)
    #   - aumenta rigidezza e resistenza nelle zone rinforzate
    "frcm_E_mult": 2.5,     # E_eq = E_mur * frcm_E_mult
    "frcm_sig_mult": 3.0,   # sig0/sigInf eq = mur * frcm_sig_mult
}

CLAMPS = {
    "max_dx": (0.05, 0.50),
    "max_dy": (0.05, 0.50),
    "dU": (0.0001, 0.0020),
    "max_steps": (10, 400),
    "target_mm": (2.0, 60.0),
    "Ptot": (1e3, 1e7),
    "testTol": (1e-8, 1e-2),
    "testIter": (5, 80),

    "grid_zx": (2, 16),
    "grid_zy": (2, 24),
    "min_solid_ratio": (0.10, 0.95),
    "greedy_topN": (5, 80),

    "frcm_min_overlap": (0.0, 1.0),

    # ✅ clamp boost
    "frcm_E_mult": (1.0, 10.0),
    "frcm_sig_mult": (1.0, 20.0),
}

# ============================================================
#  PAYLOAD UNWRAP (Lovable / n8n wrappers)
# ============================================================
def _unwrap_payload(obj: Any) -> Dict[str, Any]:
    while isinstance(obj, list) and len(obj) == 1:
        obj = obj[0]

    if not isinstance(obj, dict):
        return {}

    for k in ("data", "body", "input", "payload"):
        v = obj.get(k)
        if isinstance(v, dict):
            obj = v
            break
        if isinstance(v, list) and len(v) == 1 and isinstance(v[0], dict):
            obj = v[0]
            break

    return obj if isinstance(obj, dict) else {}

# ============================================================
#  PARSING / CLAMP
# ============================================================
def _as_int(v: Any, default: int) -> int:
    if v is None:
        return int(default)
    if isinstance(v, bool):
        return 1 if v else 0
    try:
        return int(v)
    except Exception:
        return int(default)

def _as_float(v: Any, default: float) -> float:
    if v is None:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)

def _clamp(name: str, val: Union[int, float]) -> Union[int, float]:
    if name not in CLAMPS:
        return val
    lo, hi = CLAMPS[name]
    if isinstance(val, int):
        return int(max(lo, min(val, hi)))
    return float(max(lo, min(val, hi)))

def _merge_params(payload: Dict[str, Any], query: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(DEFAULTS)

    # body
    for k in DEFAULTS:
        if k in payload:
            out[k] = payload[k]

    # query vince
    for k in DEFAULTS:
        if k in query and query[k] is not None:
            out[k] = query[k]

    out["stress"] = 1 if _as_int(out["stress"], 0) == 1 else 0
    out["verbose"] = 1 if _as_int(out.get("verbose", 0), 0) == 1 else 0

    for k in [
        "max_dx", "max_dy", "dU", "target_mm", "Ptot", "testTol", "min_solid_ratio",
        "frcm_min_overlap", "frcm_E_mult", "frcm_sig_mult"
    ]:
        out[k] = _clamp(k, _as_float(out.get(k), DEFAULTS[k]))

    for k in ["max_steps", "testIter", "n_bins_x", "n_bins_y", "grid_zx", "grid_zy", "greedy_topN"]:
        out[k] = _clamp(k, _as_int(out.get(k), DEFAULTS[k]))

    out["algo"] = str(out.get("algo", DEFAULTS["algo"]))
    out["system"] = str(out.get("system", DEFAULTS["system"]))
    out["numberer"] = str(out.get("numberer", DEFAULTS["numberer"]))
    out["constraints"] = str(out.get("constraints", DEFAULTS["constraints"]))
    out["frcm_mode"] = str(out.get("frcm_mode", DEFAULTS["frcm_mode"]))

    return out

# ============================================================
#  RECT UTILS
# ============================================================
def _normalize_rects(obj: Any) -> List[Rect]:
    out: List[Rect] = []
    if obj is None:
        return out
    if not isinstance(obj, list):
        return out
    for o in obj:
        if isinstance(o, (list, tuple)) and len(o) == 4:
            out.append((float(o[0]), float(o[1]), float(o[2]), float(o[3])))
        elif isinstance(o, dict) and all(k in o for k in ("x1", "x2", "y1", "y2")):
            out.append((float(o["x1"]), float(o["x2"]), float(o["y1"]), float(o["y2"])))
    return out

def _rect_key_from_zone(z: Dict[str, Any]) -> Rect:
    return (float(z["x1"]), float(z["x2"]), float(z["y1"]), float(z["y2"]))

def _same_rect(a: Rect, b: Rect, tol: float = 1e-9) -> bool:
    return all(abs(ai - bi) <= tol for ai, bi in zip(a, b))

def inside_opening(x: float, y: float, openings: List[Rect]) -> bool:
    for (x1, x2, y1, y2) in openings:
        if (x > x1) and (x < x2) and (y > y1) and (y < y2):
            return True
    return False

def openings_valid(openings: List[Rect]) -> bool:
    for (x1, x2, y1, y2) in openings:
        if x1 < PIER_MIN or x2 > (L - PIER_MIN):
            return False
        if not (MARGIN <= x1 < x2 <= L - MARGIN):
            return False
        if not (MARGIN <= y1 < y2 <= H - MARGIN):
            return False
        for (yc1, yc2) in CORDOLI_Y:
            if not (y2 <= yc1 - MARGIN or y1 >= yc2 + MARGIN):
                return False

    n = len(openings)
    for i in range(n):
        x1i, x2i, y1i, y2i = openings[i]
        for j in range(i + 1, n):
            x1j, x2j, y1j, y2j = openings[j]

            dx_gap = max(0.0, max(x1i, x1j) - min(x2i, x2j))
            dy_gap = max(0.0, max(y1i, y1j) - min(y2i, y2j))
            overlap_y = not (y2i <= y1j or y2j <= y1i)

            if overlap_y and dx_gap < PIER_MIN:
                return False
            if dx_gap < MARGIN and dy_gap < MARGIN:
                return False

    return True

def rect_intersection_area(a: Rect, b: Rect) -> float:
    ax1, ax2, ay1, ay2 = a
    bx1, bx2, by1, by2 = b
    ix1, ix2 = max(ax1, bx1), min(ax2, bx2)
    iy1, iy2 = max(ay1, by1), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)

# ============================================================
#  ZONE UTILS
# ============================================================
def generate_candidate_zones(openings: List[Rect], nZX: int, nZY: int, min_solid_ratio: float) -> Dict[str, Any]:
    """
    ✅ MODIFICA MINIMA (chiesta):
    - Candidate zones = SOLO rettangoli con continuità verticale terra-cielo (0..H)
    - Tagli verticali solo su bordi aperture + (0,L)
    - Ogni fascia (x1,x2) è candidata solo se NON è interrotta da nessuna apertura in quella fascia.
    - Pochissime zone -> screening molto più veloce.
    Nota: nZX/nZY restano in firma per compatibilità con API, ma qui non servono.
    """
    zones: List[Dict[str, Any]] = []

    # Tagli verticali: 0, L e bordi aperture
    xs = _unique_sorted([0.0, L] + [x for (x1, x2, _, _) in openings for x in (x1, x2)])

    zid = 0
    for i, (x1, x2) in enumerate(zip(xs[:-1], xs[1:])):
        if x2 <= x1:
            continue

        # Se esiste un'apertura che copre tutta la fascia (x1,x2), allora la continuità verticale è interrotta
        interrupted = False
        for (ox1, ox2, oy1, oy2) in openings:
            if (ox1 <= x1 + 1e-9) and (ox2 >= x2 - 1e-9):
                interrupted = True
                break

        if interrupted:
            continue

        # Fascia continua terra-cielo
        zones.append({
            "id": f"z_{zid}",
            "i": i,
            "j": 0,
            "x1": float(x1),
            "x2": float(x2),
            "y1": 0.0,
            "y2": float(H),
            "solid_ratio": 1.0,
        })
        zid += 1

    return {"grid": {"mode": "vertical_continuous", "nZX": len(zones), "nZY": 1}, "zones": zones}

def make_heatmap_cells(zones: List[Dict[str, Any]], value_key: str) -> List[Dict[str, Any]]:
    return [{
        "id": z["id"], "i": z["i"], "j": z["j"],
        "x1": z["x1"], "x2": z["x2"], "y1": z["y1"], "y2": z["y2"],
        "solid_ratio": z.get("solid_ratio"),
        "value": z.get(value_key),
    } for z in zones]

def _is_selected_zone(z: Dict[str, Any], selected_rects: List[Rect]) -> bool:
    rz = _rect_key_from_zone(z)
    return any(_same_rect(rz, rs) for rs in selected_rects)

# ============================================================
#  MESH CONFORME
# ============================================================
def _unique_sorted(vals: List[float], tol: float = 1e-6) -> List[float]:
    vals = sorted(vals)
    out: List[float] = []
    for v in vals:
        if not out or abs(v - out[-1]) > tol:
            out.append(v)
    return out

def _refine_intervals(coords: List[float], max_step: float) -> List[float]:
    refined = [coords[0]]
    for a, b in zip(coords[:-1], coords[1:]):
        seg = b - a
        if seg <= max_step:
            refined.append(b)
        else:
            n = int(np.ceil(seg / max_step))
            for k in range(1, n + 1):
                refined.append(a + seg * k / n)
    return _unique_sorted(refined)

def build_conforming_grid(openings: List[Rect], max_dx: float, max_dy: float) -> Tuple[List[float], List[float]]:
    xs = [0.0, L, MARGIN, L - MARGIN]
    ys = [0.0, H, MARGIN, H - MARGIN]

    for (yc1, yc2) in CORDOLI_Y:
        ys += [yc1, yc2]

    for (x1, x2, y1, y2) in openings:
        xs += [x1, x2]
        ys += [y1, y2]

    xs = _refine_intervals(_unique_sorted(xs), max_dx)
    ys = _refine_intervals(_unique_sorted(ys), max_dy)
    return xs, ys

# ============================================================
#  MATERIALI + SELEZIONE FRCM
# ============================================================
def K_from_E_nu(E: float, nu: float) -> float:
    return E / (3.0 * (1.0 - 2.0 * nu))

def G_from_E_nu(E: float, nu: float) -> float:
    return E / (2.0 * (1.0 + nu))

def quad_bbox(x1: float, x2: float, y1: float, y2: float) -> Rect:
    return (min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2))

def rect_area(r: Rect) -> float:
    x1, x2, y1, y2 = r
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def overlaps_any_zone(elem_rect: Rect, zones: List[Dict[str, float]], min_overlap_ratio: float) -> bool:
    a_elem = rect_area(elem_rect)
    if a_elem <= 0:
        return False
    for z in zones:
        zr = (float(z["x1"]), float(z["x2"]), float(z["y1"]), float(z["y2"]))
        inter = rect_intersection_area(elem_rect, zr)
        if inter / a_elem >= float(min_overlap_ratio):
            return True
    return False

# ============================================================
#  BUILD MODEL (✅ FRCM = muratura + FRCM equivalente)
# ============================================================
def build_wall_J2_conforming_with_frcm(
    openings: List[Rect],
    frcm_zones: List[Dict[str, float]],
    max_dx: float,
    max_dy: float,
    Ptot: float,
    frcm_mode: str = "intersection",
    frcm_min_overlap: float = 0.02,
    frcm_E_mult: float = 2.5,
    frcm_sig_mult: float = 3.0,
) -> Dict[str, Any]:
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)

    xs, ys = build_conforming_grid(openings, max_dx=max_dx, max_dy=max_dy)

    node_tags: Dict[Tuple[int, int], int] = {}
    tag = 1
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            if inside_opening(x, y, openings):
                continue
            ops.node(tag, x, y)
            node_tags[(i, j)] = tag
            tag += 1

    for i in range(len(xs)):
        key = (i, 0)
        if key in node_tags:
            ops.fix(node_tags[key], 1, 1)

    # ----------------------------
    # Materiali base
    # ----------------------------
    E_mur, nu_mur = 1.5e9, 0.15
    sig0_m, sigInf_m, delta_m, H_m = 0.5e6, 2.0e6, 8.0, 0.0

    E_cord, nu_cord = 30e9, 0.20
    sig0_c, sigInf_c, delta_c, H_c = 6.0e6, 25.0e6, 6.0, 0.0

    # (materiale FRCM puro rimane definito, ma non lo useremo più come "solo FRCM")
    E_frcm, nu_frcm = 3.0e9, 0.18
    sig0_f, sigInf_f, delta_f, H_f = 1.2e6, 4.0e6, 8.0, 0.0

    K_m, G_m = K_from_E_nu(E_mur, nu_mur), G_from_E_nu(E_mur, nu_mur)
    K_c, G_c = K_from_E_nu(E_cord, nu_cord), G_from_E_nu(E_cord, nu_cord)
    K_f, G_f = K_from_E_nu(E_frcm, nu_frcm), G_from_E_nu(E_frcm, nu_frcm)

    # J2 3D
    ops.nDMaterial("J2Plasticity", 10, K_m, G_m, sig0_m, sigInf_m, delta_m, H_m)
    ops.nDMaterial("J2Plasticity", 20, K_c, G_c, sig0_c, sigInf_c, delta_c, H_c)
    ops.nDMaterial("J2Plasticity", 30, K_f, G_f, sig0_f, sigInf_f, delta_f, H_f)

    # PlaneStress wrappers
    ops.nDMaterial("PlaneStress", 1, 10)  # muratura
    ops.nDMaterial("PlaneStress", 2, 20)  # cordoli
    ops.nDMaterial("PlaneStress", 3, 30)  # FRCM puro (non usato per rinforzo equivalente)

    # ----------------------------
    # ✅ Equivalente muratura + FRCM
    # ----------------------------
    frcm_E_mult = float(frcm_E_mult)
    frcm_sig_mult = float(frcm_sig_mult)

    E_eq = E_mur * frcm_E_mult
    nu_eq = nu_mur  # stabilità
    sig0_eq = sig0_m * frcm_sig_mult
    sigInf_eq = sigInf_m * frcm_sig_mult

    K_eq, G_eq = K_from_E_nu(E_eq, nu_eq), G_from_E_nu(E_eq, nu_eq)

    ops.nDMaterial("J2Plasticity", 40, K_eq, G_eq, sig0_eq, sigInf_eq, delta_m, H_m)
    ops.nDMaterial("PlaneStress", 4, 40)  # ✅ questo è il rinforzo vero

    # ----------------------------
    # Regola selezione
    # ----------------------------
    frcm_mode = (frcm_mode or "intersection").strip().lower()
    if frcm_mode not in ("centroid", "intersection"):
        frcm_mode = "intersection"

    t = 0.25
    eleTag = 1
    ele_mat: Dict[int, int] = {}
    frcm_hits = 0

    for j in range(len(ys) - 1):
        y1e, y2e = ys[j], ys[j + 1]
        yc = 0.5 * (y1e + y2e)

        in_cord = any((yc >= y1c) and (yc <= y2c) for (y1c, y2c) in CORDOLI_Y)

        for i in range(len(xs) - 1):
            x1e, x2e = xs[i], xs[i + 1]
            keys = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            if not all(k in node_tags for k in keys):
                continue

            # base: cordolo vince sempre
            this_mat = 2 if in_cord else 1

            # ✅ rinforzo SOLO su muratura (mai su cordoli)
            if (not in_cord) and frcm_zones:
                if frcm_mode == "intersection":
                    elem_rect = quad_bbox(x1e, x2e, y1e, y2e)
                    if overlaps_any_zone(elem_rect, frcm_zones, min_overlap_ratio=frcm_min_overlap):
                        this_mat = 4  # ✅ muratura + FRCM equivalente
                        frcm_hits += 1

            n1 = node_tags[keys[0]]
            n2 = node_tags[keys[1]]
            n3 = node_tags[keys[2]]
            n4 = node_tags[keys[3]]
            ops.element("quad", eleTag, n1, n2, n3, n4, t, "PlaneStress", this_mat, 0.0, 0.0, 0.0)

            ele_mat[eleTag] = this_mat
            eleTag += 1

    # carico in sommità
    j_top = len(ys) - 1
    top_nodes = [node_tags[(i, j_top)] for i in range(len(xs)) if (i, j_top) in node_tags]
    if not top_nodes:
        raise RuntimeError("Nessun nodo in sommità: layout aperture troppo aggressivo.")

    control_node = top_nodes[len(top_nodes) // 2]

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)

    Pnode = float(Ptot) / len(top_nodes)
    for nd in top_nodes:
        ops.load(nd, Pnode, 0.0)

    return {
        "node_tags": node_tags,
        "control_node": control_node,
        "xs": xs,
        "ys": ys,
        "n_nodes": len(node_tags),
        "n_eles": eleTag - 1,
        "ele_mat": ele_mat,
        "frcm_hits": int(frcm_hits),
        "frcm_mode": frcm_mode,
        "frcm_min_overlap": float(frcm_min_overlap),
        "frcm_E_mult": float(frcm_E_mult),
        "frcm_sig_mult": float(frcm_sig_mult),
    }

# ============================================================
#  PUSHOVER CORE
# ============================================================
def shear_at_target_disp(disp_mm: np.ndarray, shear_kN: np.ndarray, target_mm: float) -> Optional[float]:
    if len(disp_mm) < 2:
        return None
    if float(np.max(disp_mm)) < target_mm:
        return None
    return float(np.interp(target_mm, disp_mm, shear_kN))

def run_pushover_case(openings: List[Rect], frcm_zones: List[Dict[str, float]], params: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()

    if not openings_valid(openings):
        return {
            "status": "error",
            "message": "openings_invalid",
            "disp_mm": [],
            "shear_kN": [],
            "V_target": None,
            "mesh": None,
            "timing_s": {"total": float(time.time() - t0)},
            "debug": {"params_used": params},
        }

    model = build_wall_J2_conforming_with_frcm(
        openings=openings,
        frcm_zones=frcm_zones or [],
        max_dx=float(params["max_dx"]),
        max_dy=float(params["max_dy"]),
        Ptot=float(params["Ptot"]),
        frcm_mode=str(params.get("frcm_mode", "intersection")),
        frcm_min_overlap=float(params.get("frcm_min_overlap", 0.02)),
        frcm_E_mult=float(params.get("frcm_E_mult", DEFAULTS["frcm_E_mult"])),
        frcm_sig_mult=float(params.get("frcm_sig_mult", DEFAULTS["frcm_sig_mult"])),
    )

    node_tags = model["node_tags"]
    control_node = model["control_node"]
    base_nodes = [nd for (i, j), nd in node_tags.items() if j == 0]

    mesh_info = {
        "max_dx": float(params["max_dx"]),
        "max_dy": float(params["max_dy"]),
        "n_x_lines": len(model["xs"]),
        "n_y_lines": len(model["ys"]),
        "n_nodes": model["n_nodes"],
        "n_eles": model["n_eles"],
        "frcm_hits": model.get("frcm_hits", 0),
        "frcm_mode": model.get("frcm_mode"),
        "frcm_min_overlap": model.get("frcm_min_overlap"),
        "frcm_E_mult": model.get("frcm_E_mult"),
        "frcm_sig_mult": model.get("frcm_sig_mult"),
    }

    ops.constraints(str(params["constraints"]))
    ops.numberer(str(params["numberer"]))
    ops.system(str(params["system"]))
    ops.test("NormUnbalance", float(params["testTol"]), int(params["testIter"]))
    ops.algorithm("Newton")
    ops.integrator("DisplacementControl", int(control_node), 1, float(params["dU"]))
    ops.analysis("Static")

    max_steps = int(params["max_steps"])
    target_mm = float(params["target_mm"])

    disp_mm: List[float] = []
    shear_kN: List[float] = []

    for step in range(max_steps):
        ok = ops.analyze(1)
        if ok < 0:
            return {
                "status": "error",
                "message": f"analysis_failed_step_{step}",
                "disp_mm": disp_mm,
                "shear_kN": shear_kN,
                "V_target": None,
                "mesh": mesh_info,
                "timing_s": {"total": float(time.time() - t0)},
                "debug": {"params_used": params},
            }

        u = float(ops.nodeDisp(control_node, 1))
        ops.reactions()

        Vb = 0.0
        for nd in base_nodes:
            Vb += float(ops.nodeReaction(nd, 1))

        disp_mm.append(u * 1000.0)
        shear_kN.append(-Vb / 1000.0)

        if disp_mm[-1] >= target_mm:
            break

    disp_arr = np.array(disp_mm, dtype=float)
    shear_arr = np.array(shear_kN, dtype=float)
    Vt = shear_at_target_disp(disp_arr, shear_arr, target_mm)

    return {
        "status": "ok" if Vt is not None else "error",
        "message": None if Vt is not None else "analysis_not_reached_target",
        "disp_mm": disp_arr.tolist(),
        "shear_kN": shear_arr.tolist(),
        "V_target": Vt,
        "mesh": mesh_info,
        "timing_s": {"total": float(time.time() - t0)},
        "debug": {"params_used": params},
    }

# ============================================================
#  FASTAPI
# ============================================================
app = FastAPI(
    title="Wall Pushover + FRCM Screening (Conforming Mesh)",
    version="8.0.0",
    description="Parsing robusto + /frcm/add fix + FRCM equivalente muratura+FRCM (PlaneStress 4)."
)

@app.get("/")
def root():
    return {"status": "ok", "service": "pushover-frcm-screening", "defaults": DEFAULTS}

# ============================================================
#  /frcm/screening
# ============================================================
@app.post("/frcm/screening")
async def frcm_screening(
    request: Request,
    max_dx: Optional[float] = None,
    max_dy: Optional[float] = None,
    dU: Optional[float] = None,
    max_steps: Optional[int] = None,
    target_mm: Optional[float] = None,
    Ptot: Optional[float] = None,
    testTol: Optional[float] = None,
    testIter: Optional[int] = None,
    stress: Optional[int] = None,
    verbose: Optional[int] = None,
    grid_zx: Optional[int] = None,
    grid_zy: Optional[int] = None,
    min_solid_ratio: Optional[float] = None,
    frcm_mode: Optional[str] = None,
    frcm_min_overlap: Optional[float] = None,
    frcm_E_mult: Optional[float] = None,
    frcm_sig_mult: Optional[float] = None,
):
    raw = await request.json()
    payload = _unwrap_payload(raw)
    if not payload:
        return JSONResponse(status_code=400, content={"error": "Payload non valido."})

    query = {
        "max_dx": max_dx, "max_dy": max_dy, "dU": dU, "max_steps": max_steps,
        "target_mm": target_mm, "Ptot": Ptot, "testTol": testTol, "testIter": testIter,
        "stress": stress, "verbose": verbose,
        "grid_zx": grid_zx, "grid_zy": grid_zy, "min_solid_ratio": min_solid_ratio,
        "frcm_mode": frcm_mode, "frcm_min_overlap": frcm_min_overlap,
        "frcm_E_mult": frcm_E_mult, "frcm_sig_mult": frcm_sig_mult,
    }
    params = _merge_params(payload, query)

    openings = _normalize_rects(payload.get("openings") or payload.get("project_openings") or payload.get("existing_openings"))
    if not openings:
        return JSONResponse(status_code=400, content={"error": "Manca 'openings'."})
    if not openings_valid(openings):
        return JSONResponse(status_code=400, content={"error": "openings_invalid"})

    t_req0 = time.time()

    cand = generate_candidate_zones(
        openings=openings,
        nZX=int(params["grid_zx"]),
        nZY=int(params["grid_zy"]),
        min_solid_ratio=float(params["min_solid_ratio"]),
    )

    baseline = run_pushover_case(openings=openings, frcm_zones=[], params=params)
    if baseline["status"] != "ok":
        return JSONResponse(content={
            "meta": {"params_used": params, "timing_s": {"request_total": float(time.time() - t_req0)}},
            "baseline": baseline,
            "screening": {"grid": cand["grid"], "zones": [], "heatmap": []},
            "recommendation": None,
        })

    Vbase = float(baseline["V_target"])
    results: List[Dict[str, Any]] = []

    for z in cand["zones"]:
        frcm = [{"x1": z["x1"], "x2": z["x2"], "y1": z["y1"], "y2": z["y2"]}]
        out = run_pushover_case(openings=openings, frcm_zones=frcm, params=params)
        Vz = out.get("V_target")
        deltaV = (float(Vz) - Vbase) if (Vz is not None) else None
        results.append({**z, "V_target": Vz, "deltaV_single": deltaV})

    valid = [r for r in results if isinstance(r.get("deltaV_single"), (int, float))]
    best = max(valid, key=lambda r: float(r["deltaV_single"])) if valid else None

    return JSONResponse(content={
        "meta": {"params_used": params, "timing_s": {"request_total": float(time.time() - t_req0)}},
        "baseline": baseline,
        "screening": {"grid": cand["grid"], "zones": results, "heatmap": make_heatmap_cells(results, "deltaV_single")},
        "recommendation": (None if best is None else {"best_zone": best}),
    })

# ============================================================
#  /frcm/add
# ============================================================
@app.post("/frcm/add")
async def frcm_add(
    request: Request,
    max_dx: Optional[float] = None,
    max_dy: Optional[float] = None,
    dU: Optional[float] = None,
    max_steps: Optional[int] = None,
    target_mm: Optional[float] = None,
    Ptot: Optional[float] = None,
    testTol: Optional[float] = None,
    testIter: Optional[int] = None,
    stress: Optional[int] = None,
    verbose: Optional[int] = None,
    grid_zx: Optional[int] = None,
    grid_zy: Optional[int] = None,
    min_solid_ratio: Optional[float] = None,
    greedy_topN: Optional[int] = None,
    frcm_mode: Optional[str] = None,
    frcm_min_overlap: Optional[float] = None,
    frcm_E_mult: Optional[float] = None,
    frcm_sig_mult: Optional[float] = None,
):
    raw = await request.json()
    payload = _unwrap_payload(raw)
    if not payload:
        return JSONResponse(status_code=400, content={"error": "Payload non valido."})

    query = {
        "max_dx": max_dx, "max_dy": max_dy, "dU": dU, "max_steps": max_steps,
        "target_mm": target_mm, "Ptot": Ptot, "testTol": testTol, "testIter": testIter,
        "stress": stress, "verbose": verbose,
        "grid_zx": grid_zx, "grid_zy": grid_zy, "min_solid_ratio": min_solid_ratio,
        "greedy_topN": greedy_topN,
        "frcm_mode": frcm_mode, "frcm_min_overlap": frcm_min_overlap,
        "frcm_E_mult": frcm_E_mult, "frcm_sig_mult": frcm_sig_mult,
    }
    params = _merge_params(payload, query)

    openings = _normalize_rects(payload.get("openings") or payload.get("project_openings") or payload.get("existing_openings"))
    if not openings:
        return JSONResponse(status_code=400, content={"error": "Manca 'openings'."})
    if not openings_valid(openings):
        return JSONResponse(status_code=400, content={"error": "openings_invalid"})

    selected_raw = (
        payload.get("selected_zones")
        or payload.get("selectedZones")
        or (payload.get("next") or {}).get("selected_zones")
        or (payload.get("next") or {}).get("selectedZones")
    )
    selected_rects: List[Rect] = _normalize_rects(selected_raw)
    selected: List[Dict[str, float]] = [{"x1": x1, "x2": x2, "y1": y1, "y2": y2} for (x1, x2, y1, y2) in selected_rects]

    ranked_raw = payload.get("ranked_candidates") or payload.get("rankedCandidates")
    ranked_list = ranked_raw if isinstance(ranked_raw, list) else []

    t_req0 = time.time()

    cand = generate_candidate_zones(
        openings=openings,
        nZX=int(params["grid_zx"]),
        nZY=int(params["grid_zy"]),
        min_solid_ratio=float(params["min_solid_ratio"]),
    )
    zones = cand["zones"]

    ranked_ids_received: List[str] = []
    if ranked_list:
        zones_by_id = {z["id"]: z for z in zones}
        ranked_zones = []
        for r in ranked_list:
            zid = (r or {}).get("id")
            if zid in zones_by_id:
                ranked_ids_received.append(zid)
                z = zones_by_id[zid].copy()
                z["deltaV_single_hint"] = (r or {}).get("deltaV_single")
                ranked_zones.append(z)
        ranked_zones.sort(key=lambda z: float(z.get("deltaV_single_hint") or -1e18), reverse=True)
        zones_eval = ranked_zones
    else:
        zones_eval = zones

    excluded_ids = [z["id"] for z in zones_eval if _is_selected_zone(z, selected_rects)]
    zones_eval = [z for z in zones_eval if not _is_selected_zone(z, selected_rects)]
    zones_eval = zones_eval[: int(params["greedy_topN"])]

    current = run_pushover_case(openings=openings, frcm_zones=[dict(s) for s in selected], params=params)
    Vcur = float(current["V_target"]) if current["status"] == "ok" and current.get("V_target") is not None else None

    trials: List[Dict[str, Any]] = []
    best_trial: Optional[Dict[str, Any]] = None

    if Vcur is not None:
        for z in zones_eval:
            trial_zones = [dict(s) for s in selected] + [{"x1": z["x1"], "x2": z["x2"], "y1": z["y1"], "y2": z["y2"]}]
            out = run_pushover_case(openings=openings, frcm_zones=trial_zones, params=params)
            Vt = out.get("V_target")
            deltaV = (float(Vt) - float(Vcur)) if (Vt is not None) else None
            rec = {**z, "V_target": Vt, "deltaV_marginal": deltaV, "mesh": out.get("mesh"), "status": out.get("status")}
            trials.append(rec)
            if deltaV is not None and (best_trial is None or float(deltaV) > float(best_trial["deltaV_marginal"])):
                best_trial = {**rec, "case": out}

    resp = {
        "meta": {"params_used": params, "timing_s": {"request_total": float(time.time() - t_req0)}},
        "current": current,
        "added": (None if best_trial is None else {
            "id": best_trial["id"], "i": best_trial["i"], "j": best_trial["j"],
            "x1": best_trial["x1"], "x2": best_trial["x2"], "y1": best_trial["y1"], "y2": best_trial["y2"],
            "deltaV_marginal": best_trial.get("deltaV_marginal"),
        }),
        "next": (None if best_trial is None else {
            "selected_zones": [dict(s) for s in selected] + [{"x1": best_trial["x1"], "x2": best_trial["x2"], "y1": best_trial["y1"], "y2": best_trial["y2"]}],
            "case": best_trial["case"],
        }),
        "greedy": {"evaluated": trials, "heatmap": make_heatmap_cells(trials, "deltaV_marginal")},
        "debug": {
            "payload_keys": list(payload.keys()),
            "selected_raw_type": str(type(selected_raw)),
            "ranked_raw_type": str(type(ranked_raw)),
            "selected_rects_parsed": selected_rects,
            "ranked_ids_received": ranked_ids_received,
            "excluded_ids": excluded_ids,
            "evaluated_ids": [t["id"] for t in trials],
        }
    }
    return JSONResponse(content=resp)

# ============================================================
#  /pushover
# ============================================================
@app.post("/pushover")
async def pushover_simple(
    request: Request,
    max_dx: Optional[float] = None,
    max_dy: Optional[float] = None,
    dU: Optional[float] = None,
    max_steps: Optional[int] = None,
    target_mm: Optional[float] = None,
    Ptot: Optional[float] = None,
    testTol: Optional[float] = None,
    testIter: Optional[int] = None,
    stress: Optional[int] = None,
    verbose: Optional[int] = None,
    frcm_mode: Optional[str] = None,
    frcm_min_overlap: Optional[float] = None,
    frcm_E_mult: Optional[float] = None,
    frcm_sig_mult: Optional[float] = None,
):
    raw = await request.json()
    payload = _unwrap_payload(raw)
    if not payload:
        return JSONResponse(status_code=400, content={"error": "Payload non valido."})

    query = {
        "max_dx": max_dx, "max_dy": max_dy, "dU": dU, "max_steps": max_steps,
        "target_mm": target_mm, "Ptot": Ptot, "testTol": testTol, "testIter": testIter,
        "stress": stress, "verbose": verbose,
        "frcm_mode": frcm_mode, "frcm_min_overlap": frcm_min_overlap,
        "frcm_E_mult": frcm_E_mult, "frcm_sig_mult": frcm_sig_mult,
    }
    params = _merge_params(payload, query)

    openings = _normalize_rects(payload.get("openings"))
    frcm = _normalize_rects(payload.get("frcm_zones"))
    if not openings:
        return JSONResponse(status_code=400, content={"error": "Manca 'openings'."})
    if not openings_valid(openings):
        return JSONResponse(status_code=400, content={"error": "openings_invalid"})

    frcm_zones = [{"x1": a, "x2": b, "y1": c, "y2": d} for (a, b, c, d) in frcm]
    out = run_pushover_case(openings=openings, frcm_zones=frcm_zones, params=params)
    return JSONResponse(content={"meta": {"params_used": params}, "result": out})
