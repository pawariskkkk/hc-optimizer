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
    r1 = np.array(data["r1"])
    active = data["activeShifts"]

    window_size = data.get("windowSize", 9)
    break_start = data.get("breakStart", 3)
    break_end = data.get("breakEnd", 5)
    min_shift = data.get("minShift", 17)

    num_periods = len(r1)
    num_shifts = len(active)

    # =====================================================
    # VARIABLE INDEX
    # =====================================================

    def v_idx(si):
        return si

    def b_idx(si, period):
        return num_shifts + si * num_periods + period

    def y_idx(si):
        return num_shifts + num_shifts * num_periods + si

    V_SIZE = num_shifts
    B_SIZE = num_shifts * num_periods
    Y_SIZE = num_shifts
    TOTAL_VARS = V_SIZE + B_SIZE + Y_SIZE

    # =====================================================
    # OBJECTIVE
    # minimize total workers
    # =====================================================

    c = np.zeros(TOTAL_VARS)
    for si in range(num_shifts):
        c[v_idx(si)] = 1

    # =====================================================
    # BOUNDS
    # =====================================================

    lb = np.zeros(TOTAL_VARS)
    ub = np.full(TOTAL_VARS, np.inf)

    # binary y
    for si in range(num_shifts):
        ub[y_idx(si)] = 1

    # break windows
    for si, i in enumerate(active):
        for p in range(num_periods):
            k = p - i
            if not (break_start <= k <= break_end):
                ub[b_idx(si, p)] = 0

    # =====================================================
    # INTEGER VARIABLES
    # =====================================================

    integrality = np.ones(TOTAL_VARS)

    # =====================================================
    # CONSTRAINT STORAGE
    # =====================================================

    A_rows = []
    b_lo = []
    b_hi = []

    # =====================================================
    # 1. v[i] == 0 OR >= MIN_SHIFT
    # =====================================================

    BIG_M = 10000

    for si in range(num_shifts):
        # v >= min_shift * y
        row = np.zeros(TOTAL_VARS)
        row[v_idx(si)] = 1
        row[y_idx(si)] = -min_shift
        A_rows.append(row)
        b_lo.append(0)
        b_hi.append(np.inf)

        # v <= M * y
        row = np.zeros(TOTAL_VARS)
        row[v_idx(si)] = 1
        row[y_idx(si)] = -BIG_M
        A_rows.append(row)
        b_lo.append(-np.inf)
        b_hi.append(0)

    # =====================================================
    # 2. COVERAGE
    # =====================================================

    def active_shifts(period):
        return range(
            max(0, period - window_size + 1),
            min(period + 1, num_shifts)
        )

    for j in range(num_periods):
        if r1[j] <= 0:
            continue

        row = np.zeros(TOTAL_VARS)
        for si, i in enumerate(active):
            if i <= j < i + window_size:
                row[v_idx(si)] = 1

        A_rows.append(row)
        b_lo.append(r1[j])
        b_hi.append(np.inf)

    # =====================================================
    # 3. BREAK ASSIGNMENT
    # sum_j b[i,j] = v[i]
    # =====================================================

    for si, i in enumerate(active):
        row = np.zeros(TOTAL_VARS)
        row[v_idx(si)] = 1

        for p in range(num_periods):
            k = p - i
            if break_start <= k <= break_end:
                row[b_idx(si, p)] = -1

        A_rows.append(row)
        b_lo.append(0)
        b_hi.append(0)

    # =====================================================
    # 4. BREAK CAPACITY
    # sum_i b[i,j] <= r2[j] - r1[j]
    # =====================================================

    for j in range(num_periods):
        if r1[j] <= 0:
            continue

        row = np.zeros(TOTAL_VARS)
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

        A_rows.append(row)
        b_lo.append(-np.inf)
        b_hi.append(-r1[j])

    # =====================================================
    # 5. FIX LAST SHIFT
    # =====================================================

    last_shift = num_shifts - 1
    fixed_val = int(r1[-1])

    row = np.zeros(TOTAL_VARS)
    row[v_idx(last_shift)] = 1
    A_rows.append(row)
    b_lo.append(fixed_val)
    b_hi.append(fixed_val)

    # =====================================================
    # 6. FIRST HALF: MAX 2 CONSECUTIVE ACTIVE SHIFTS
    # =====================================================

    half = num_shifts // 2

    for si in range(half - 2):
        row = np.zeros(TOTAL_VARS)
        row[y_idx(si)] = 1
        row[y_idx(si + 1)] = 1
        row[y_idx(si + 2)] = 1

        A_rows.append(row)
        b_lo.append(-np.inf)
        b_hi.append(2)

    # =====================================================
    # 7. SHIFT UPPER BOUNDS
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

        A_rows.append(row)
        b_lo.append(-np.inf)
        b_hi.append(limit)

    # =====================================================
    # SOLVE
    # =====================================================

    res = milp(
        c=c,
        constraints=LinearConstraint(
            np.array(A_rows),
            np.array(b_lo),
            np.array(b_hi)
        ),
        integrality=integrality,
        bounds=Bounds(lb, ub)
    )

    if not res.success:
        return jsonify({
            "feasible": False,
            "message": str(res.message)
        })

    # =====================================================
    # EXTRACT
    # =====================================================

    x = np.round(res.x).astype(int)

    v = x[V_SIZE + 0 - V_SIZE : V_SIZE]  # same as x[0:V_SIZE]
    v = x[:num_shifts]

    b = x[
        num_shifts : num_shifts + B_SIZE
    ].reshape(num_shifts, num_periods)

    # =====================================================
    # BUILD r2
    # =====================================================

    r2 = np.zeros(num_periods, dtype=int)

    for j in range(num_periods):
        for si, i in enumerate(active):
            if i <= j < i + window_size:
                r2[j] += v[si]

    r3 = r2 - r1

    return jsonify({
        "feasible": True,
        "result": int(np.sum(v)),
        "v": v.tolist(),
        "r1": r1.astype(int).tolist(),
        "r2": r2.astype(int).tolist(),
        "r3": r3.astype(int).tolist(),
        "lastShiftValue": int(v[-1])
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)