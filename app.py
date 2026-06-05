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

    r1 = np.array(data["r1"])              # trimmed demand (no leading/trailing zeros)

    window_size       = data.get("windowSize", 9)
    break_start       = data.get("breakStart", 3)
    break_end         = data.get("breakEnd", 5)
    min_shift         = data.get("minShift", 17)

    num_periods = len(r1)
    num_shifts  = num_periods - window_size + 1   # contiguous shifts, like Colab

    if num_shifts <= 0:
        return jsonify({
            "feasible": False,
            "message": "Demand stretch shorter than one shift window."
        })

    # =====================================================
    # VARIABLE LAYOUT
    #   v[i] : i
    #   b[i,p] : num_shifts + i*num_periods + p
    #   y[i] : num_shifts + num_shifts*num_periods + i
    # =====================================================
    def v_idx(i):    return i
    def b_idx(i, p): return num_shifts + i * num_periods + p
    def y_idx(i):    return num_shifts + num_shifts * num_periods + i

    total_vars = num_shifts * 2 + num_shifts * num_periods

    # shift i covers period j ?
    def active_shifts(period):
        return range(max(0, period - window_size + 1), min(period + 1, num_shifts))

    # =====================================================
    # OBJECTIVE  (minimize workers; push last shift down)
    # =====================================================
    c = np.zeros(total_vars)
    for i in range(num_shifts):
        c[v_idx(i)] = 1

    last_shift = num_shifts - 1
    

    # =====================================================
    # BOUNDS
    # =====================================================
    lb = np.zeros(total_vars)
    ub = np.full(total_vars, np.inf)

    # y binary
    for i in range(num_shifts):
        ub[y_idx(i)] = 1

    # break windows: b = 0 outside i+break_start .. i+break_end
    for i in range(num_shifts):
        for p in range(num_periods):
            k = p - i
            if not (break_start <= k <= break_end):
                ub[b_idx(i, p)] = 0

    # =====================================================
    # INTEGER VARIABLES (v, y, b integer;)
    # =====================================================
    integrality = np.ones(total_vars)    # everything integer (1)

    # =====================================================
    # CONSTRAINTS
    # =====================================================
    A, b_l, b_u = [], [], []

    # 1. v == 0 OR >= min_shift
    BIG_M = 10000
    for i in range(num_shifts):
        # v >= min_shift * y
        row = np.zeros(total_vars)
        row[v_idx(i)] = 1
        row[y_idx(i)] = -min_shift
        A.append(row); b_l.append(0); b_u.append(np.inf)
        # v <= M * y
        row = np.zeros(total_vars)
        row[v_idx(i)] = 1
        row[y_idx(i)] = -BIG_M
        A.append(row); b_l.append(-np.inf); b_u.append(0)

    # 2. COVERAGE  r2[j] >= r1[j]
    for j in range(num_periods):
        row = np.zeros(total_vars)
        for i in active_shifts(j):
            row[v_idx(i)] = 1
        A.append(row); b_l.append(int(r1[j])); b_u.append(np.inf)

    # 3. BREAK ASSIGNMENT  sum_p b[i,p] = v[i]
    for i in range(num_shifts):
        row = np.zeros(total_vars)
        row[v_idx(i)] = 1
        for p in range(num_periods):
            k = p - i
            if break_start <= k <= break_end:
                row[b_idx(i, p)] = -1
        A.append(row); b_l.append(0); b_u.append(0)

    # 4. BREAK CAPACITY  sum_i b[i,j] <= r2[j] - r1[j]
    for j in range(num_periods):
        row = np.zeros(total_vars)
        for i in range(num_shifts):
            row[b_idx(i, j)] = 1
        for i in active_shifts(j):
            row[v_idx(i)] -= 1
        A.append(row); b_l.append(-np.inf); b_u.append(-int(r1[j]))

    # 5. FIX LAST SHIFT to last period's demand
    row = np.zeros(total_vars)
    row[v_idx(last_shift)] = 1
    A.append(row); b_l.append(int(r1[-1])); b_u.append(int(r1[-1]))

    # 6. FIRST HALF: MAX 2 CONSECUTIVE ACTIVE SHIFTS
    half = num_shifts // 2
    for i in range(half - 2):
        row = np.zeros(total_vars)
        row[y_idx(i)]     = 1
        row[y_idx(i + 1)] = 1
        row[y_idx(i + 2)] = 1
        A.append(row); b_l.append(-np.inf); b_u.append(2)

    # 7. SHIFT UPPER BOUNDS
    for i in range(num_shifts):
        if i == 0:
            limit = max(100, int(r1[0]))
        elif i == num_shifts - 1:
            limit = max(100, int(r1[-1]))
        else:
            limit = 100
        row = np.zeros(total_vars)
        row[v_idx(i)] = 1
        A.append(row); b_l.append(-np.inf); b_u.append(limit)

    # =====================================================
    # SOLVE
    # =====================================================
    res = milp(
        c           = c,
        constraints = LinearConstraint(np.array(A), np.array(b_l), np.array(b_u)),
        integrality = integrality,
        bounds      = Bounds(lb, ub)
    )

    if not res.success:
        return jsonify({ "feasible": False, "message": str(res.message) })

    # =====================================================
    # EXTRACT
    # =====================================================
    x = res.x
    v = [int(round(x[v_idx(i)])) for i in range(num_shifts)]

    r2 = np.zeros(num_periods, dtype=int)
    for j in range(num_periods):
        for i in active_shifts(j):
            r2[j] += v[i]
    r3 = r2 - r1

    return jsonify({
        "feasible":       True,
        "result":         int(sum(v)),
        "v":              v,
        "r1":             r1.astype(int).tolist(),
        "r2":             r2.astype(int).tolist(),
        "r3":             r3.astype(int).tolist(),
        "lastShiftValue": int(v[-1])
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)