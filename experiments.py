"""
Parameter-sweep harness for the Avellaneda-Stoikov strategy in main.py.

Edit CONFIG below, then:  python3 experiments.py
Plots are saved as PNGs in experiment_plots/, and also popped up on screen.

This reuses main.py's own dates()/data()/updating() -- it does not
reimplement the pricing model. main.py's updating() now takes an optional
`risk=` argument (default: main.py's own module-level risk, so nothing
about main.py's own behavior changes) and an optional `record_trace=True`
to get back full intraday arrays for one day instead of just the
end-of-day summary.
"""

from datetime import date
import os

import numpy as np
import matplotlib.pyplot as plt

import main


# ============================================================
# CONFIG -- everything you'd want to tweak to test lives here
#
# COST WARNING: main.py's updating() replays every tick of every day in
# Python, one row at a time. On this dataset that's ~60-90 seconds per
# (day, risk, k) combination for a busy day (e.g. ~560k quote updates on
# 2025-03-03). Each extra value you add to k_values/risk_values multiplies
# runtime by (that many extra combos) x (trading days in range) x ~60-90s.
# The defaults below are kept small on purpose so a first run finishes in
# a few minutes -- widen the lists once you've confirmed it works.
# ============================================================
CONFIG = {
    # date range used for the k-sweep (graph 1) and risk-sweep (graph 2)
    'start_date': date(2025, 3, 1),
    'end_date': date(2025, 3, 5),

    # --- graph 1: k sweep ---------------------------------------------
    # total X, total fill count, and average end-of-day q vs k, at one
    # fixed risk value, summed/averaged over start_date..end_date
    'k_values': [1, 5, 10],
    'risk_for_k_sweep': 0.01,

    # --- graph 2: risk sweep (panel data) -------------------------------
    # daily X / q / count, one line per risk value, at one fixed k
    'risk_values': [0.005, 0.01, 0.02],
    'k_for_risk_sweep': 10,

    # --- graph 3: intraday trace ----------------------------------------
    # mid / pA / pB over the day, with q underneath, for specific day(s)
    'trace_dates': [date(2025, 3, 3)],
    'risk_for_trace': 0.01,
    'k_for_trace': 10,

    # output behaviour
    'show_plots': True,
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'experiment_plots')


# ============================================================
# Graph 1: X / count / avg end-of-day q vs k, at a fixed risk
# ============================================================
def graph_k_sweep(data_cache, k_values, risk):
    k_list, X_list, count_list, avg_q_list = [], [], [], []
    for k in k_values:
        PL = main.updating(data_cache, k, risk=risk)
        k_list.append(k)
        X_list.append(sum(entry['X'] for entry in PL))
        count_list.append(sum(entry['count'] for entry in PL))
        avg_q_list.append(np.mean([entry['q'] for entry in PL]) if PL else 0)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(k_list, X_list, marker='o', color='tab:blue')
    axes[0].set_ylabel('Total P/L (X)')
    axes[0].set_title(f'P/L vs k  (risk={risk})')
    axes[0].grid(True)

    axes[1].plot(k_list, count_list, marker='o', color='tab:green')
    axes[1].set_ylabel('Fill count')
    axes[1].set_title('Fill count vs k')
    axes[1].grid(True)

    axes[2].bar(k_list, avg_q_list, color='tab:orange')
    axes[2].set_ylabel('Avg end-of-day q')
    axes[2].set_xlabel('k')
    axes[2].set_title('Average end-of-day inventory vs k')
    axes[2].grid(True)

    fig.tight_layout()
    return fig


# ============================================================
# Graph 2: daily X / q / count, one line per risk value (panel data),
# at a fixed k -- three separate figures
# ============================================================
def graph_risk_panels(data_cache, risk_values, k):
    per_risk = {risk: main.updating(data_cache, k, risk=risk) for risk in risk_values}

    figs = []
    for metric, ylabel, title in [
        ('X', 'P/L (X)', 'Daily P/L by risk'),
        ('q', 'End-of-day q', 'Daily end-of-day inventory by risk'),
        ('count', 'Fill count', 'Daily fill count by risk'),
    ]:
        fig, ax = plt.subplots(figsize=(11, 5))
        for risk, PL in per_risk.items():
            PL_sorted = sorted(PL, key=lambda e: e['date'])
            dates_ = [entry['date'] for entry in PL_sorted]
            vals = [entry[metric] for entry in PL_sorted]
            ax.plot(dates_, vals, marker='o', label=f'risk={risk}')
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Date')
        ax.set_title(f'{title}  (k={k})')
        ax.legend()
        ax.grid(True)
        fig.autofmt_xdate()
        fig.tight_layout()
        figs.append(fig)
    return figs  # [fig_X, fig_q, fig_count]


# ============================================================
# Graph 3: intraday mid / pA / pB, with q underneath, for one day
# ============================================================
def graph_day_trace(data_cache, trace_date, risk, k):
    day_df = data_cache.get(trace_date)
    if day_df is None or day_df.empty:
        print(f'  no data for {trace_date}, skipping trace plot')
        return None

    PL = main.updating({trace_date: day_df}, k, risk=risk, record_trace=True)
    trace = PL[0]['trace']

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    axes[0].plot(trace['t'], trace['mid'], label='mid', color='black', linewidth=1)
    axes[0].plot(trace['t'], trace['pA'], label='pA (ask quote)', color='tab:red', linewidth=0.8)
    axes[0].plot(trace['t'], trace['pB'], label='pB (bid quote)', color='tab:blue', linewidth=0.8)
    axes[0].set_ylabel('Price')
    axes[0].set_title(f'{trace_date}  --  mid / pA / pB  (risk={risk}, k={k})')
    axes[0].legend()
    axes[0].grid(True)

    # step + fill instead of a plain bar() -- a day has tens of thousands of
    # ticks, and a literal bar per tick is both slow to render and unreadable;
    # this reads the same way (a filled q profile) without either problem
    axes[1].fill_between(trace['t'], trace['q'], step='post', color='tab:purple', alpha=0.6)
    axes[1].axhline(0, color='black', linewidth=0.6)
    axes[1].set_ylabel('q')
    axes[1].set_xlabel('Time')
    axes[1].set_title('Inventory (q) over the day')
    axes[1].grid(True)

    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# ============================================================
# Orchestration
# ============================================================
def main_run():
    cfg = CONFIG

    fnames = set(main.dates(cfg['start_date'], cfg['end_date']))
    for td in cfg['trace_dates']:
        fnames |= set(main.dates(td, td))

    print(f'Loading {len(fnames)} trading day(s)...')
    data_cache = main.data(list(fnames))

    sweep_cache = {d: df for d, df in data_cache.items() if cfg['start_date'] <= d <= cfg['end_date']}

    os.makedirs(OUT_DIR, exist_ok=True)
    figs = []

    print('Running k sweep (graph 1)...')
    fig1 = graph_k_sweep(sweep_cache, cfg['k_values'], cfg['risk_for_k_sweep'])
    fig1.savefig(os.path.join(OUT_DIR, '1_k_sweep.png'), dpi=130)
    figs.append(fig1)

    print('Running risk sweep / panel data (graph 2)...')
    fig_X, fig_q, fig_count = graph_risk_panels(sweep_cache, cfg['risk_values'], cfg['k_for_risk_sweep'])
    fig_X.savefig(os.path.join(OUT_DIR, '2a_risk_panel_X.png'), dpi=130)
    fig_q.savefig(os.path.join(OUT_DIR, '2b_risk_panel_q.png'), dpi=130)
    fig_count.savefig(os.path.join(OUT_DIR, '2c_risk_panel_count.png'), dpi=130)
    figs.extend([fig_X, fig_q, fig_count])

    print('Running intraday trace(s) (graph 3)...')
    for trace_date in cfg['trace_dates']:
        fig3 = graph_day_trace(data_cache, trace_date, cfg['risk_for_trace'], cfg['k_for_trace'])
        if fig3 is not None:
            fig3.savefig(os.path.join(OUT_DIR, f'3_trace_{trace_date}.png'), dpi=130)
            figs.append(fig3)

    print(f'Saved {len(figs)} plot(s) to {OUT_DIR}')
    if cfg.get('show_plots', True):
        plt.show()


if __name__ == '__main__':
    main_run()
