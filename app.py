from flask import Flask, request, jsonify
import os
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

app = Flask(__name__)

@app.route("/")
def home():
    return "HC Optimizer API Running"

@app.route("/solve", methods=["POST"])
def solve():
    data = request.get_json()

    r1     = np.array(data["r1"])      # full demand row (keep the zeros)
    active = data["activeShifts"]      # shift start positions (zeros already filtered out)

    window_size = data.get("windowSize", 9)
    break_start = data.get("breakStart", 3)
    break_end   = data.get("breakEnd", 5)
    min_shift   = data.get("minShift", 17)
    last_shift_weight = data.get("lastShiftWeight", 100000)

    num_periods = len(r1)
    num_shifts  = len(active)

    # =====================================================
    # VARIABLE INDEX
    # =====================================================
    def v_idx(si):       return si
    def y_idx(si):       return num_shifts + si
    def b_idx(si, p):    return num_shifts * 2 + si * num_periods + p

    total_vars = num_shifts * 2 + num_shifts * num_periods

    # =====================================================
    # OBJECTIVE  (minimize workers; push last shift down hard)
    # =====================================================
    c = np.zeros(total_vars)
    for si in range(num_shifts):
        c[v_idx(si)] = 1

    last_shift_si = num_shifts - 1
    c[v_idx(last_shift_si)] += last_shift_weight

    # =====================================================
    # BOUNDS
    # =====================================================
    lb = np.zeros(total_vars)
    ub = np.full(total_vars, np.inf)

    # y binary
    for si in range(num_shifts):
        ub[y_idx(si)] = 1

    # break windows: b = 0 outside i+3 .. i+5
    for si, i in enumerate(active):
        for p in range(num_periods):
            k = p - i
            if not (break_start <= k <= break_end):
                ub[b_idx(si, p)] = 0

    # =====================================================
    # INTEGER VARIABLES (v and y integer; b continuous = faster)
    # =====================================================
    integrality = np.zeros(total_vars)
    for si in range(num_shifts):
        integrality[v_idx(si)] = 1
        integrality[y_idx(si)] = 1

    # =====================================================
    # CONSTRAINT STORAGE
    # =====================================================
    A_rows, b_lo, b_hi = [], [], []

    # =====================================================
    # v == 0 OR >= min_shift
    # =====================================================
    BIG_M = 10000
    for si in range(num_shifts):
        # v >= min_shift * y
        row = np.zeros(total_vars)
        row[v_idx(si)] = 1
        row[y_idx(si)] = -min_shift
        A_rows.append(row); b_lo.append(0); b_hi.append(np.inf)

        # v <= M * y
        row = np.zeros(total_vars)
        row[v_idx(si)] = 1
        row[y_idx(si)] = -BIG_M
        A_rows.append(row); b_lo.append(-np.inf); b_hi.append(0)

    # =====================================================
    # COVERAGE  (zeros skipped automatically)
    # =====================================================
    for j in range(num_periods):
        if r1[j] <= 0:
            continue
        row = np.zeros(total_vars)
        for si, i in enumerate(active):
            if i <= j < i + window_size:
                row[v_idx(si)] = 1
        A_rows.append(row); b_lo.append(int(r1[j])); b_hi.append(np.inf)

    # =====================================================
    # BREAK ASSIGNMENT  (sum of breaks = v[si])
    # =====================================================
    for si, i in enumerate(active):
        row = np.zeros(total_vars)
        row[v_idx(si)] = 1
        for p in range(num_periods):
            k = p - i
            if break_start <= k <= break_end:
                row[b_idx(si, p)] = -1
        A_rows.append(row); b_lo.append(0); b_hi.append(0)

    # =====================================================
    # BREAK CAPACITY  (breaks at j <= surplus at j)
    # =====================================================
    for j in range(num_periods):
        if r1[j] <= 0:
            continue
        row = np.zeros(total_vars)
        has_breaker = False
        for si, i in enumerate(active):
            k = j - i
            if break_start <= k <= break_end:
                row[b_idx(si, j)] += 1
                has_breaker = True
            if i <= j < i + window_size:
                row[v_idx(si)] -= 1
        if not has_breaker:
            continue
        A_rows.append(row); b_lo.append(-np.inf); b_hi.append(-int(r1[j]))

    # =====================================================
    # LIMIT LAST SHIFT
    # cap it at the demand of the last period it actually covers
    # (NOT r1[-1], which is a padding zero)
    # =====================================================
    last_period_covered = active[-1] + window_size - 1
    if last_period_covered >= num_periods:
        last_period_covered = num_periods - 1
    last_limit = int(r1[last_period_covered])

    row = np.zeros(total_vars)
    row[v_idx(last_shift_si)] = 1
    A_rows.append(row); b_lo.append(0); b_hi.append(last_limit)

    # =====================================================
    # SOLVE
    # =====================================================
    res = milp(
        c           = c,
        constraints = LinearConstraint(np.array(A_rows), np.array(b_lo), np.array(b_hi)),
        integrality = integrality,
        bounds      = Bounds(lb, ub)
    )

    if not res.success:
        return jsonify({ "feasible": False, "message": str(res.message) })

    # =====================================================
    # EXTRACT
    # =====================================================
    x = res.x
    v = [int(round(x[v_idx(si)])) for si in range(num_shifts)]

    # BUILD r2 / r3
    r2 = np.zeros(num_periods)
    for j in range(num_periods):
        for si, i in enumerate(active):
            if i <= j < i + window_size:
                r2[j] += v[si]
    r3 = r2 - r1

    return jsonify({
        "feasible":       True,
        "result":         int(sum(v)),
        "v":              v,
        "r1":             r1.astype(int).tolist(),
        "r2":             r2.astype(int).tolist(),
        "r3":             r3.astype(int).tolist(),
        "lastShiftValue": int(v[-1]),
        "lastShiftLimit": last_limit
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)