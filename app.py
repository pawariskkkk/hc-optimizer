from flask import Flask, request, jsonify

import numpy as np

from scipy.optimize import (
    milp,
    LinearConstraint,
    Bounds
)

app = Flask(__name__)

@app.route("/")
def home():
    return "HC Optimizer API Running"

@app.route("/solve", methods=["POST"])
def solve():

    data = request.get_json()

    r1 = np.array(data["r1"])

    active = data["activeShifts"]

    window_size = data.get("windowSize", 9)

    break_start = data.get("breakStart", 3)

    break_end = data.get("breakEnd", 5)

    min_shift = data.get("minShift", 17)

    last_shift_weight = data.get(
        "lastShiftWeight",
        100000
    )

    num_periods = len(r1)

    num_shifts = len(active)

    # =====================================================
    # VARIABLE INDEX
    # =====================================================

    def v_idx(si):
        return si

    def y_idx(si):
        return num_shifts + si

    def b_idx(si, period):

        return (
            num_shifts * 2
            + si * num_periods
            + period
        )

    total_vars = (
        num_shifts * 2
        + num_shifts * num_periods
    )

    # =====================================================
    # OBJECTIVE
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

    for si in range(num_shifts):
        ub[y_idx(si)] = 1

    # break windows

    for si, i in enumerate(active):

        for p in range(num_periods):

            k = p - i

            if not (
                break_start <= k <= break_end
            ):

                ub[b_idx(si, p)] = 0

    # =====================================================
    # INTEGER VARIABLES
    # =====================================================

    integrality = np.zeros(total_vars)

    for si in range(num_shifts):

        integrality[v_idx(si)] = 1

        integrality[y_idx(si)] = 1

    # =====================================================
    # CONSTRAINT STORAGE
    # =====================================================

    A_rows = []

    b_lo = []

    b_hi = []

    # =====================================================
    # v == 0 OR >= min_shift
    # =====================================================

    BIG_M = 10000

    for si in range(num_shifts):

        # v >= min_shift * y

        row = np.zeros(total_vars)

        row[v_idx(si)] = 1

        row[y_idx(si)] = -min_shift

        A_rows.append(row)

        b_lo.append(0)

        b_hi.append(np.inf)

        # v <= M*y

        row = np.zeros(total_vars)

        row[v_idx(si)] = 1

        row[y_idx(si)] = -BIG_M

        A_rows.append(row)

        b_lo.append(-np.inf)

        b_hi.append(0)

    # =====================================================
    # COVERAGE
    # =====================================================

    for j in range(num_periods):

        if r1[j] <= 0:
            continue

        row = np.zeros(total_vars)

        for si, i in enumerate(active):

            if i <= j < i + window_size:

                row[v_idx(si)] = 1

        A_rows.append(row)

        b_lo.append(r1[j])

        b_hi.append(np.inf)

    # =====================================================
    # BREAK ASSIGNMENT
    # =====================================================

    for si, i in enumerate(active):

        row = np.zeros(total_vars)

        row[v_idx(si)] = 1

        for p in range(num_periods):

            k = p - i

            if break_start <= k <= break_end:

                row[b_idx(si, p)] = -1

        A_rows.append(row)

        b_lo.append(0)

        b_hi.append(0)

    # =====================================================
    # BREAK CAPACITY
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

        A_rows.append(row)

        b_lo.append(-np.inf)

        b_hi.append(-r1[j])

    # =====================================================
    # LIMIT LAST SHIFT
    # =====================================================

    last_limit = int(max(r1[-3:]))

    row = np.zeros(total_vars)

    row[v_idx(last_shift_si)] = 1

    A_rows.append(row)

    b_lo.append(0)

    b_hi.append(last_limit)

    # =====================================================
    # SOLVE
    # =====================================================

    res = milp(

        c = c,

        constraints = LinearConstraint(
            np.array(A_rows),
            np.array(b_lo),
            np.array(b_hi)
        ),

        integrality = integrality,

        bounds = Bounds(lb, ub)
    )

    if not res.success:

        return jsonify({

            "feasible": False,

            "message": res.message
        })

    # =====================================================
    # EXTRACT
    # =====================================================

    x = res.x

    v = [

        int(round(x[v_idx(si)]))

        for si in range(num_shifts)
    ]

    # =====================================================
    # BUILD r2
    # =====================================================

    r2 = np.zeros(num_periods)

    for j in range(num_periods):

        for si, i in enumerate(active):

            if i <= j < i + window_size:

                r2[j] += v[si]

    r3 = r2 - r1

    return jsonify({

        "feasible": True,

        "result": int(sum(v)),

        "v": v,

        "r1": r1.astype(int).tolist(),

        "r2": r2.astype(int).tolist(),

        "r3": r3.astype(int).tolist(),

        "lastShiftValue": int(v[-1]),

        "lastShiftLimit": last_limit
    })

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )