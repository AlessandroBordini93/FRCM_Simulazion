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


# ============================================================
#  DEFAULTS
# ============================================================
DEFAULTS = {
    "stress": 0,
    "max_dx": 0.20,
    "max_dy": 0.20,
    "target_mm": 15.0,
    "dU": 0.0002,
    "max_steps": 250,
    "Ptot": 100e3,
    "testTol": 1.0e-4,
    "testIter": 15,
    "algo": "Newton",
    "system": "BandGeneral",
    "numberer": "RCM",
    "constraints": "Plain",
    "n_bins_x": 4,
    "n_bins_y": 12,
    "verbose": 0,

    # screening / greedy
    "grid_zx": 8,
    "grid_zy": 12,
    "min_solid_ratio": 0.60,
    "greedy_topN": 25,
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
}


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

    # query vince sul body
    for k in DEFAULTS:
        if k in query and query[k] is not None:
            out[k] = query[k]

    out["stress"] = 1 if _as_int(out["stress"], 0) == 1 else 0
    out["verbose"] = 1 if _as_int(out.get("verbose", 0), 0) == 1 else 0

    # numerici
    for k in ["max_dx", "max_dy", "dU", "target_mm", "Ptot", "testTol", "min_solid_ratio"]:
        out[k] = _clamp(k, _as_float(out[k], DEFAULTS[k]))
    for k in ["max_steps", "testIter", "n_bins_x", "n_bins_y", "grid_zx", "grid_zy", "greedy_topN"]:
        out[k] = _clamp(k, _as_int(out[k], DEFAULTS[k]))

    # stringhe
    out["algo"] = str(out.get("algo", DEFAULTS["algo"]))
    out["system"] = str(out.get("system", DEFAULTS["system"]))
    out["numberer"] = str(out.get("numberer", DEFAULTS["numberer"]))
    out["constraints"] = str(out.get("constraints", DEFAULTS["constraints"]))

    return out


# ============================================================
#  RECT / OPENINGS UTILS
# ============================================================
Rect = Tuple[float, float, float, float]  # (x1,x2,y1,y2)

def _normalize_openings(obj: Any) -> List[Rect]:
    """
    Supporta:
      - [[x1,x2,y1,y2], ...]
      - [{"x1":..,"x2":..,"y1":..,"y2":..}, ...]
    """
    out: List[Rect] = []
    if obj is None:
        return out
    if not isinstance(obj, list):
        return out
    for o in obj:
        if isinstance(o, (list, tuple)) and len(o) == 4:
            out.append((float(o[0]), float(o[1]), float(o[2]), float(o[3])))
        elif isinstance(o, dict):
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
#  ZONE UTILS (heatmap candidates)
# ============================================================
def generate_candidate_zones(openings: List[Rect],
                             nZX: int, nZY: int,
                             min_solid_ratio: float) -> Dict[str, Any]:
    zones: List[Dict[str, Any]] = []
    dx = L / nZX
    dy = H / nZY
    for j in range(nZY):
        for i in range(nZX):
            x1, x2 = i * dx, (i + 1) * dx
            y1, y2 = j * dy, (j + 1) * dy
            area = dx * dy

            void = 0.0
            for op in openings:
                void += rect_intersection_area((x1, x2, y1, y2), op)

            solid_ratio = max(0.0, (area - void) / area)
            if solid_ratio >= min_solid_ratio:
                zones.append({
                    "id": f"z_{i}_{j}",
                    "i": i, "j": j,
                    "x1": x1, "x2": x2, "y1": y1, "y2": y2,
                    "solid_ratio": float(solid_ratio),
                })

    return {"grid": {"nZX": nZX, "nZY": nZY}, "zones": zones}


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

def build_conforming_grid(openings: List[Rect],
                          max_dx: float, max_dy: float) -> Tuple[List[float], List[float]]:
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
#  MATERIALI
# ============================================================
def K_from_E_nu(E: float, nu: float) -> float:
    return E / (3.0 * (1.0 - 2.0 * nu))

def G_from_E_nu(E: float, nu: float) -> float:
    return E / (2.0 * (1.0 + nu))

def point_in_any_zone(xc: float, yc: float, zones: List[Dict[str, float]]) -> bool:
    for z in zones:
        x1, x2 = float(z["x1"]), float(z["x2"])
        y1, y2 = float(z["y1"]), float(z["y2"])
        if (xc >= x1) and (xc <= x2) and (yc >= y1) and (yc <= y2):
            return True
    return False


# ============================================================
#  BUILD MODEL (muratura + cordoli + FRCM proxy)
# ============================================================
def build_wall_J2_conforming_with_frcm(
    openings: List[Rect],
    frcm_zones: List[Dict[str, float]],
    max_dx: float, max_dy: float,
    Ptot: float
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

    # vincoli base
    for i in range(len(xs)):
        key = (i, 0)
        if key in node_tags:
            ops.fix(node_tags[key], 1, 1)

    # materiali
    # Muratura
    E_mur, nu_mur = 1.5e9, 0.15
    sig0_m, sigInf_m, delta_m, H_m = 0.5e6, 2.0e6, 8.0, 0.0

    # Cordoli
    E_cord, nu_cord = 30e9, 0.20
    sig0_c, sigInf_c, delta_c, H_c = 6.0e6, 25.0e6, 6.0, 0.0

    # FRCM proxy
    E_frcm, nu_frcm = 3.0e9, 0.18
    sig0_f, sigInf_f, delta_f, H_f = 1.2e6, 4.0e6, 8.0, 0.0

    K_m, G_m = K_from_E_nu(E_mur, nu_mur), G_from_E_nu(E_mur, nu_mur)
    K_c, G_c = K_from_E_nu(E_cord, nu_cord), G_from_E_nu(E_cord, nu_cord)
    K_f, G_f = K_from_E_nu(E_frcm, nu_frcm), G_from_E_nu(E_frcm, nu_frcm)

    # J2 3D tags
    ops.nDMaterial("J2Plasticity", 10, K_m, G_m, sig0_m, sigInf_m, delta_m, H_m)
    ops.nDMaterial("J2Plasticity", 20, K_c, G_c, sig0_c, sigInf_c, delta_c, H_c)
    ops.nDMaterial("J2Plasticity", 30, K_f, G_f, sig0_f, sigInf_f, delta_f, H_f)

    # PlaneStress wrappers
    ops.nDMaterial("PlaneStress", 1, 10)  # mur
    ops.nDMaterial("PlaneStress", 2, 20)  # cord
    ops.nDMaterial("PlaneStress", 3, 30)  # frcm

    t = 0.25
    eleTag = 1
    ele_mat: Dict[int, int] = {}

    for j in range(len(ys) - 1):
        yc = 0.5 * (ys[j] + ys[j + 1])

        in_cord = False
        for (y1c, y2c) in CORDOLI_Y:
            if (yc >= y1c) and (yc <= y2c):
                in_cord = True
                break

        for i in range(len(xs) - 1):
            keys = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            if not all(k in node_tags for k in keys):
                continue

            xc = 0.5 * (xs[i] + xs[i + 1])

            # regola: cordolo vince, frcm solo su muratura
            this_mat = 2 if in_cord else 1
            if (not in_cord) and frcm_zones and point_in_any_zone(xc, yc, frcm_zones):
                this_mat = 3

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
    }


# ============================================================
#  STRESS PROFILES (opzionale)
# ============================================================
def _compute_stress_grid_profiles(n_bins_x: int, n_bins_y: int) -> Dict[str, Any]:
    ele_tags = ops.getEleTags()
    if not ele_tags:
        return {"tau_profile_y": {"y": [], "tau_mean": []},
                "sigma_profile_y": {"y": [], "sigma_c_mean": []},
                "zones": []}

    y_edges = np.linspace(0.0, H, n_bins_y + 1)
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    x_edges = np.linspace(0.0, L, n_bins_x + 1)

    tau_sum_y = np.zeros(n_bins_y)
    tau_cnt_y = np.zeros(n_bins_y)
    sigc_sum_y = np.zeros(n_bins_y)
    sigc_cnt_y = np.zeros(n_bins_y)

    for ele in ele_tags:
        stress = ops.eleResponse(ele, "stress")
        if stress is None:
            continue

        tau_vals = []
        sigy_vals = []
        for k in range(0, len(stress), 3):
            sigy_vals.append(float(stress[k + 1]))
            tau_vals.append(abs(float(stress[k + 2])))

        if not tau_vals:
            continue

        tau_mean_el = float(sum(tau_vals) / len(tau_vals))
        sigma_c_el = float(abs(min(sigy_vals)))

        nds = ops.eleNodes(ele)
        ys_el = [float(ops.nodeCoord(nd)[1]) for nd in nds]
        xs_el = [float(ops.nodeCoord(nd)[0]) for nd in nds]
        yc = sum(ys_el) / len(ys_el)
        xc = sum(xs_el) / len(xs_el)

        j = int(np.searchsorted(y_edges, yc) - 1)
        i = int(np.searchsorted(x_edges, xc) - 1)
        if j < 0 or j >= n_bins_y or i < 0 or i >= n_bins_x:
            continue

        tau_sum_y[j] += tau_mean_el
        tau_cnt_y[j] += 1
        sigc_sum_y[j] += sigma_c_el
        sigc_cnt_y[j] += 1

    y_out = [float(y_centers[j]) for j in range(n_bins_y) if tau_cnt_y[j] > 0]
    tau_mean = [float(tau_sum_y[j] / tau_cnt_y[j]) for j in range(n_bins_y) if tau_cnt_y[j] > 0]
    y_sig = [float(y_centers[j]) for j in range(n_bins_y) if sigc_cnt_y[j] > 0]
    sigc_mean = [float(sigc_sum_y[j] / sigc_cnt_y[j]) for j in range(n_bins_y) if sigc_cnt_y[j] > 0]

    return {
        "tau_profile_y": {"y": y_out, "tau_mean": tau_mean},
        "sigma_profile_y": {"y": y_sig, "sigma_c_mean": sigc_mean},
        "zones": []
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

def run_pushover_case(openings: List[Rect],
                      frcm_zones: List[Dict[str, float]],
                      params: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()

    if not openings_valid(openings):
        return {
            "status": "error",
            "message": "openings_invalid",
            "disp_mm": [],
            "shear_kN": [],
            "V_target": None,
            "timing_s": {"total": float(time.time() - t0)},
        }

    verbose = int(params.get("verbose", 0)) == 1
    buf = io.StringIO() if verbose else None

    # ---- BUILD ----
    t_build0 = time.time()
    model = build_wall_J2_conforming_with_frcm(
        openings=openings,
        frcm_zones=frcm_zones or [],
        max_dx=float(params["max_dx"]),
        max_dy=float(params["max_dy"]),
        Ptot=float(params["Ptot"]),
    )
    t_build = time.time() - t_build0

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
    }

    # ---- SETUP ----
    t_setup0 = time.time()
    ops.constraints(str(params["constraints"]))
    ops.numberer(str(params["numberer"]))
    ops.system(str(params["system"]))
    ops.test("NormUnbalance", float(params["testTol"]), int(params["testIter"]))
    ops.algorithm("Newton")
    ops.integrator("DisplacementControl", int(control_node), 1, float(params["dU"]))
    ops.analysis("Static")
    t_setup = time.time() - t_setup0

    # ---- LOOP ----
    max_steps = int(params["max_steps"])
    target_mm = float(params["target_mm"])

    disp_mm: List[float] = []
    shear_kN: List[float] = []

    t_loop0 = time.time()
    for step in range(max_steps):
        if verbose:
            buf.truncate(0); buf.seek(0)
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                ok = ops.analyze(1)
        else:
            ok = ops.analyze(1)

        if ok < 0:
            return {
                "status": "error",
                "message": f"analysis_failed_step_{step}",
                "disp_mm": disp_mm,
                "shear_kN": shear_kN,
                "V_target": None,
                "debug": (buf.getvalue().strip()[:800] if verbose else None),
                "mesh": mesh_info,
                "timing_s": {
                    "build": float(t_build),
                    "setup": float(t_setup),
                    "loop": float(time.time() - t_loop0),
                    "stress_profiles": 0.0,
                    "total": float(time.time() - t0),
                },
            }

        u = float(ops.nodeDisp(control_node, 1))  # m
        ops.reactions()

        Vb = 0.0
        for nd in base_nodes:
            Vb += float(ops.nodeReaction(nd, 1))

        disp_mm.append(u * 1000.0)
        shear_kN.append(-Vb / 1000.0)

        if disp_mm[-1] >= target_mm:
            break

    t_loop = time.time() - t_loop0

    disp_arr = np.array(disp_mm, dtype=float)
    shear_arr = np.array(shear_kN, dtype=float)
    Vt = shear_at_target_disp(disp_arr, shear_arr, target_mm)

    out: Dict[str, Any] = {
        "status": "ok" if Vt is not None else "error",
        "message": None if Vt is not None else "analysis_not_reached_target",
        "disp_mm": disp_arr.tolist(),
        "shear_kN": shear_arr.tolist(),
        "V_target": Vt,
        "mesh": mesh_info,
        "timing_s": {
            "build": float(t_build),
            "setup": float(t_setup),
            "loop": float(t_loop),
            "stress_profiles": 0.0,
            "total": float(time.time() - t0),
        },
    }

    if int(params["stress"]) == 1:
        t_st0 = time.time()
        out["stress_profiles"] = _compute_stress_grid_profiles(int(params["n_bins_x"]), int(params["n_bins_y"]))
        out["timing_s"]["stress_profiles"] = float(time.time() - t_st0)

    return out


# ============================================================
#  SCREENING / GREEDY
# ============================================================
def make_heatmap_cells(zones: List[Dict[str, Any]], value_key: str) -> List[Dict[str, Any]]:
    cells = []
    for z in zones:
        cells.append({
            "id": z["id"],
            "i": z["i"],
            "j": z["j"],
            "x1": z["x1"], "x2": z["x2"], "y1": z["y1"], "y2": z["y2"],
            "solid_ratio": z.get("solid_ratio"),
            "value": z.get(value_key),
        })
    return cells

def _is_selected_zone(z: Dict[str, Any], selected_rects: List[Rect]) -> bool:
    rz = _rect_key_from_zone(z)
    return any(_same_rect(rz, rs) for rs in selected_rects)


# ============================================================
#  FASTAPI
# ============================================================
app = FastAPI(
    title="Wall Pushover + FRCM Screening (Conforming Mesh)",
    version="4.0.1",
    description="Baseline + screening zone FRCM + greedy add, con timing in JSON."
)

@app.get("/")
def root():
    return {"status": "ok", "service": "pushover-frcm-screening", "defaults": DEFAULTS}


@app.post("/frcm/screening")
async def frcm_screening(
    request: Request,
    # query params opzionali
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
):
    payload = await request.json()
    if isinstance(payload, list):
        if not payload:
            return JSONResponse(status_code=400, content={"error": "Payload list vuota."})
        payload = payload[0]
    if not isinstance(payload, dict):
        return JSONResponse(status_code=400, content={"error": "Payload non valido: atteso oggetto JSON."})

    query = {
        "max_dx": max_dx, "max_dy": max_dy, "dU": dU, "max_steps": max_steps,
        "target_mm": target_mm, "Ptot": Ptot, "testTol": testTol, "testIter": testIter,
        "stress": stress, "verbose": verbose,
        "grid_zx": grid_zx, "grid_zy": grid_zy, "min_solid_ratio": min_solid_ratio,
    }
    params = _merge_params(payload, query)

    openings = _normalize_openings(payload.get("openings") or payload.get("project_openings") or payload.get("existing_openings"))
    if not openings:
        return JSONResponse(status_code=400, content={"error": "Manca 'openings' (o project_openings/existing_openings)."})
    if not openings_valid(openings):
        return JSONResponse(status_code=400, content={"error": "openings_invalid"})

    t_req0 = time.time()

    # 1) candidate zones
    t_z0 = time.time()
    cand = generate_candidate_zones(
        openings=openings,
        nZX=int(params["grid_zx"]),
        nZY=int(params["grid_zy"]),
        min_solid_ratio=float(params["min_solid_ratio"]),
    )
    t_z = time.time() - t_z0

    # 2) baseline
    baseline = run_pushover_case(openings=openings, frcm_zones=[], params=params)
    if baseline["status"] != "ok":
        return JSONResponse(content={
            "meta": {"params_used": params, "timing_s": {"request_total": float(time.time() - t_req0), "zones_gen": float(t_z)}},
            "baseline": baseline,
            "screening": {"grid": cand["grid"], "zones": [], "heatmap": []},
            "recommendation": None,
        })

    Vbase = float(baseline["V_target"])

    # 3) single-zone screening
    zones = cand["zones"]
    t_sc0 = time.time()
    results: List[Dict[str, Any]] = []

    for z in zones:
        frcm = [{"x1": z["x1"], "x2": z["x2"], "y1": z["y1"], "y2": z["y2"]}]
        out = run_pushover_case(openings=openings, frcm_zones=frcm, params=params)
        Vz = out.get("V_target")
        deltaV = (float(Vz) - Vbase) if (Vz is not None) else None

        results.append({
            **z,
            "V_target": Vz,
            "deltaV_single": deltaV,
            "case": {
                "status": out["status"],
                "message": out["message"],
                "mesh": out.get("mesh"),
                "timing_s": out.get("timing_s"),
            }
        })

    t_sc = time.time() - t_sc0

    valid = [r for r in results if isinstance(r.get("deltaV_single"), (int, float))]
    best = max(valid, key=lambda r: float(r["deltaV_single"])) if valid else None

    best_case = None
    if best is not None:
        frcm_best = [{"x1": best["x1"], "x2": best["x2"], "y1": best["y1"], "y2": best["y2"]}]
        best_case = run_pushover_case(openings=openings, frcm_zones=frcm_best, params=params)

    heatmap = make_heatmap_cells(results, "deltaV_single")

    resp = {
        "meta": {
            "params_used": params,
            "timing_s": {
                "zones_gen": float(t_z),
                "screening_loop": float(t_sc),
                "request_total": float(time.time() - t_req0),
            }
        },
        "baseline": baseline,
        "screening": {
            "grid": cand["grid"],
            "zones": results,
            "heatmap": heatmap,
        },
        "recommendation": (None if best is None else {
            "best_zone": {
                "id": best["id"], "i": best["i"], "j": best["j"],
                "x1": best["x1"], "x2": best["x2"], "y1": best["y1"], "y2": best["y2"],
                "solid_ratio": best["solid_ratio"],
                "deltaV_single": best["deltaV_single"],
            },
            "best_case": best_case,
        }),
    }
    return JSONResponse(content=resp)


@app.post("/frcm/add")
async def frcm_add(
    request: Request,
    # query params opzionali (devono restare coerenti con screening)
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
):
    payload = await request.json()
    if isinstance(payload, list):
        if not payload:
            return JSONResponse(status_code=400, content={"error": "Payload list vuota."})
        payload = payload[0]
    if not isinstance(payload, dict):
        return JSONResponse(status_code=400, content={"error": "Payload non valido: atteso oggetto JSON."})

    query = {
        "max_dx": max_dx, "max_dy": max_dy, "dU": dU, "max_steps": max_steps,
        "target_mm": target_mm, "Ptot": Ptot, "testTol": testTol, "testIter": testIter,
        "stress": stress, "verbose": verbose,
        "grid_zx": grid_zx, "grid_zy": grid_zy, "min_solid_ratio": min_solid_ratio,
        "greedy_topN": greedy_topN,
    }
    params = _merge_params(payload, query)

    openings = _normalize_openings(payload.get("openings") or payload.get("project_openings") or payload.get("existing_openings"))
    if not openings:
        return JSONResponse(status_code=400, content={"error": "Manca 'openings' (o project_openings/existing_openings)."})
    if not openings_valid(openings):
        return JSONResponse(status_code=400, content={"error": "openings_invalid"})

    # ---- selected zones (robusto) ----
    selected_rects: List[Rect] = _normalize_openings(payload.get("selected_zones"))
    selected: List[Dict[str, float]] = [
        {"x1": x1, "x2": x2, "y1": y1, "y2": y2}
        for (x1, x2, y1, y2) in selected_rects
    ]

    t_req0 = time.time()

    # rigenera candidates (coerente con screening)
    t_z0 = time.time()
    cand = generate_candidate_zones(
        openings=openings,
        nZX=int(params["grid_zx"]),
        nZY=int(params["grid_zy"]),
        min_solid_ratio=float(params["min_solid_ratio"]),
    )
    t_z = time.time() - t_z0
    zones = cand["zones"]

    # se payload porta già ranking dallo screening, lo usiamo per scegliere topN
    ranked = payload.get("ranked_candidates")
    if isinstance(ranked, list) and ranked:
        zones_by_id = {z["id"]: z for z in zones}
        ranked_zones = []
        for r in ranked:
            zid = r.get("id")
            if zid in zones_by_id:
                z = zones_by_id[zid].copy()
                z["deltaV_single_hint"] = r.get("deltaV_single")
                ranked_zones.append(z)
        ranked_zones.sort(key=lambda z: float(z.get("deltaV_single_hint") or -1e18), reverse=True)
        zones_eval = ranked_zones
    else:
        zones_eval = zones

    topN = int(params["greedy_topN"])

    # ✅ filtro CORRETTO: non riproporre zone già selezionate
    zones_eval = [z for z in zones_eval if not _is_selected_zone(z, selected_rects)]
    zones_eval = zones_eval[:topN]

    # ✅ current: deve essere con i rinforzi selezionati (copy difensiva)
    current = run_pushover_case(openings=openings, frcm_zones=[dict(s) for s in selected], params=params)
    if current["status"] != "ok":
        return JSONResponse(content={
            "meta": {"params_used": params, "timing_s": {"request_total": float(time.time() - t_req0), "zones_gen": float(t_z)}},
            "current": current,
            "added": None,
            "next": None,
        })

    Vcur = float(current["V_target"])

    # prova aggiunta di ciascuna candidata (solo topN)
    t_g0 = time.time()
    trials: List[Dict[str, Any]] = []
    best_trial: Optional[Dict[str, Any]] = None

    for z in zones_eval:
        trial_zones = [dict(s) for s in selected] + [{"x1": z["x1"], "x2": z["x2"], "y1": z["y1"], "y2": z["y2"]}]
        out = run_pushover_case(openings=openings, frcm_zones=trial_zones, params=params)
        Vt = out.get("V_target")
        deltaV = (float(Vt) - Vcur) if (Vt is not None) else None

        rec = {
            **z,
            "V_target": Vt,
            "deltaV_marginal": deltaV,
            "timing_s": out.get("timing_s"),
            "mesh": out.get("mesh"),
            "status": out.get("status"),
            "message": out.get("message"),
        }
        trials.append(rec)

        if deltaV is not None:
            if (best_trial is None) or (float(deltaV) > float(best_trial["deltaV_marginal"])):
                best_trial = {**rec, "case": out}

    t_g = time.time() - t_g0

    heatmap = make_heatmap_cells(trials, "deltaV_marginal")

    if best_trial is None:
        return JSONResponse(content={
            "meta": {
                "params_used": params,
                "timing_s": {
                    "zones_gen": float(t_z),
                    "greedy_eval": float(t_g),
                    "request_total": float(time.time() - t_req0),
                }
            },
            "current": current,
            "added": None,
            "next": None,
            "greedy": {"evaluated": trials, "heatmap": heatmap, "topN": topN},
        })

    added_zone = {
        "id": best_trial["id"], "i": best_trial["i"], "j": best_trial["j"],
        "x1": best_trial["x1"], "x2": best_trial["x2"],
        "y1": best_trial["y1"], "y2": best_trial["y2"],
        "solid_ratio": best_trial.get("solid_ratio"),
        "deltaV_marginal": best_trial.get("deltaV_marginal"),
    }

    new_selected = [dict(s) for s in selected] + [{"x1": added_zone["x1"], "x2": added_zone["x2"], "y1": added_zone["y1"], "y2": added_zone["y2"]}]
    new_case = best_trial["case"]

    resp = {
        "meta": {
            "params_used": params,
            "timing_s": {
                "zones_gen": float(t_z),
                "greedy_eval": float(t_g),
                "request_total": float(time.time() - t_req0),
            }
        },
        "current": current,
        "added": added_zone,
        "next": {
            "selected_zones": new_selected,   # ✅ sempre dict, non list di list
            "case": new_case,
        },
        "greedy": {
            "topN": topN,
            "evaluated": trials,
            "heatmap": heatmap,
        }
    }
    return JSONResponse(content=resp)


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
):
    payload = await request.json()
    if isinstance(payload, list):
        if not payload:
            return JSONResponse(status_code=400, content={"error": "Payload list vuota."})
        payload = payload[0]
    if not isinstance(payload, dict):
        return JSONResponse(status_code=400, content={"error": "Payload non valido: atteso oggetto JSON."})

    query = {
        "max_dx": max_dx, "max_dy": max_dy, "dU": dU, "max_steps": max_steps,
        "target_mm": target_mm, "Ptot": Ptot, "testTol": testTol, "testIter": testIter,
        "stress": stress, "verbose": verbose,
    }
    params = _merge_params(payload, query)

    openings = _normalize_openings(payload.get("openings"))
    frcm = _normalize_openings(payload.get("frcm_zones"))

    if not openings:
        return JSONResponse(status_code=400, content={"error": "Manca 'openings'."})

    frcm_zones = [{"x1": a, "x2": b, "y1": c, "y2": d} for (a, b, c, d) in frcm]

    t0 = time.time()
    out = run_pushover_case(openings=openings, frcm_zones=frcm_zones, params=params)
    return JSONResponse(content={
        "meta": {"params_used": params, "timing_s": {"request_total": float(time.time() - t0)}},
        "result": out
    })
