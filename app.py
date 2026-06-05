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

    # INPUT
    r1     = np.array(data["r1"])        # full demand row (still contains the 0s)
    active = data["activeShifts"]        # timeline indices where a shift may start (0s cut)

    window_size = data.get("windowSize", 9)
    break_start = data.get("breakStart", 3)
    break_end   = data.get("breakEnd", 5)
    min_shift   = data.get("minShift", 17)

    num_periods = len(r1)
    num_shifts  = len(active)

    # =====================================================
    # VARIABLE LAYOUT
    #   v[si]    : si
    #   b[si,p]  : num_shifts + si*num_periods + p
    #   y[si]    : num_shifts + num_shifts*num_periods + si
    # =====================================================
    def v_idx(si):    return si
    def b_idx(si, p): return num_shifts + si * num_periods + p
    def y_idx(si):    return num_shifts + num_shifts * num_periods + si

    V_SIZE = num_shifts
    B_SIZE = num_shifts * num_periods
    Y_SIZE = num_shifts
    TOTAL_VARS = V_SIZE + B_SIZE + Y_SIZE

    # =====================================================
    # OBJECTIVE — minimize total workers
    # =====================================================
    c = np.zeros(TOTAL_VARS)
    for si in range(num_shifts):
        c[v_idx(si)] = 1

    # =====================================================
    # BOUNDS
    # =====================================================
    lb = np.zeros(TOTAL_VARS)
    ub = np.full(TOTAL_VARS, np.inf)

    # y is binary
    for si in range(num_shifts):
        ub[y_idx(si)] = 1

    # BREAK WINDOW RESTRICTION (constraint 4 done via bounds):
    # shift si may only break at active[si]+3, +4, +5 → all other b forced to 0
    for si, i in enumerate(active):
        for p in range(num_periods):
            k = p - i
            if not (break_start <= k <= break_end):
                ub[b_idx(si, p)] = 0

    integrality = np.ones(TOTAL_VARS)  # v, b, y all integer (matches standalone)

    # =====================================================
    # CONSTRAINT STORAGE
    # =====================================================
    A, b_l, b_u = [], [], []

    # which shifts (si) cover period j?
    def covering(j):
        return [si for si, i in enumerate(active) if i <= j < i + window_size]

    # =====================================================
    # 1. COVERAGE  r2[j] >= r1[j]
    # =====================================================
    for j in range(num_periods):
        row = np.zeros(TOTAL_VARS)
        for si in covering(j):
            row[v_idx(si)] = 1
        A.append(row); b_l.append(int(r1[j])); b_u.append(np.inf)

    # =====================================================
    # 2. EVERY EMPLOYEE TAKES EXACTLY ONE BREAK
    #    sum_p b[si,p] = v[si]
    # =====================================================
    for si in range(num_shifts):
        row = np.zeros(TOTAL_VARS)
        row[v_idx(si)] = 1
        for p in range(num_periods):
            row[b_idx(si, p)] = -1
        A.append(row); b_l.append(0); b_u.append(0)

    # =====================================================
    # 3. BREAK CAPACITY
    #    sum_si b[si,j] <= r2[j] - r1[j]
    # =====================================================
    for j in range(num_periods):
        row = np.zeros(TOTAL_VARS)
        for si in range(num_shifts):
            row[b_idx(si, j)] = 1
        for si in covering(j):
            row[v_idx(si)] -= 1
        A.append(row); b_l.append(-np.inf); b_u.append(-int(r1[j]))

    # =====================================================
    # 5. v[si] == 0 OR >= MIN_SHIFT
    # =====================================================
    BIG_M = 10000
    for si in range(num_shifts):
        # v <= M*y
        row = np.zeros(TOTAL_VARS)
        row[v_idx(si)] = 1
        row[y_idx(si)] = -BIG_M
        A.append(row); b_l.append(-np.inf); b_u.append(0)
        # v >= min_shift*y
        row = np.zeros(TOTAL_VARS)
        row[v_idx(si)] = 1
        row[y_idx(si)] = -min_shift
        A.append(row); b_l.append(0); b_u.append(np.inf)

    # =====================================================
    # 6. FIX LAST SHIFT to r1[-1]
    # =====================================================
    last_shift = num_shifts - 1
    fixed_val  = int(r1[-1])
    row = np.zeros(TOTAL_VARS)
    row[v_idx(last_shift)] = 1
    A.append(row); b_l.append(fixed_val); b_u.append(fixed_val)

    # =====================================================
    # 7. FIRST HALF: MAX 2 CONSECUTIVE ACTIVE SHIFTS
    # =====================================================
    half = num_shifts // 2
    for si in range(half - 2):
        row = np.zeros(TOTAL_VARS)
        row[y_idx(si)]     = 1
        row[y_idx(si + 1)] = 1
        row[y_idx(si + 2)] = 1
        A.append(row); b_l.append(-np.inf); b_u.append(2)

    # =====================================================
    # SHIFT UPPER BOUNDS
    #   middle shifts <= 100
    #   first & last  <= max(100, their endpoint demand)
    # =====================================================
    for si in range(num_shifts):
        if si == 0:
            limit = max(100, int(r1[0]))
        elif si == num_shifts - 1:
            limit = max(100, int(r1[-1]))
        else:
            limit = 100
        row = np.zeros(TOTAL_VARS)
        row[v_idx(si)] = 1
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
    x = np.round(res.x).astype(int)
    v = x[:num_shifts]
    b = x[num_shifts : num_shifts + B_SIZE].reshape(num_shifts, num_periods)

    # BUILD r2 / r3
    r2 = np.zeros(num_periods, dtype=int)
    for j in range(num_periods):
        for si in covering(j):
            r2[j] += v[si]
    r3 = r2 - r1

    return jsonify({
        "feasible":       True,
        "result":         int(np.sum(v)),
        "v":              v.tolist(),
        "r1":             r1.astype(int).tolist(),
        "r2":             r2.astype(int).tolist(),
        "r3":             r3.astype(int).tolist(),
        "lastShiftValue": int(v[-1])
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)