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
import hashlib
import inspect
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
    'end_date': date(2025, 3, 1),

    # --- graph 1: k sweep ---------------------------------------------
    # total X, wealth, total fill count, and average end-of-day q vs k, at one
    # fixed risk value, summed/averaged over start_date..end_date
    'run_k_sweep': False,
    'k_values': [20, 30, 40, 50, 60, 70],
    'risk_for_k_sweep': 0.0001,

    # --- graph 2: risk sweep (panel data) -------------------------------
    # daily X / q / count, one line per risk value, at one fixed k
    'run_risk_sweep': False,
    'risk_values': [0.001, 0.005, 0.0001],
    'k_for_risk_sweep': 30,

    # --- graphs 3/4/5: intraday trace + distributions --------------------
    # mid / pA / pB over the day, with q underneath -- one PNG is printed
    # per trading day found in [trace_start_date, trace_end_date] (set both
    # to the same day to only run one day, same as before). Graphs 4 and 5
    # are built from the same replay, so this one switch covers all three.
    'run_trace': True,
    'trace_start_date': date(2025, 3, 1),
    'trace_end_date': date(2026, 3, 1),
    'risk_for_trace': 0.0004,
    'k_for_trace': 45,

    # --- graph 5: y-axis cap ---------------------------------------------
    # q spends most of its ticks near 0, so those bins tower over the tails and
    # squash everything else flat. This caps the y-axis at a tick count: taller
    # bars are drawn cut off at the cap (with their true height labelled) and
    # every value still counts towards the bins, the mean and the sd. None =
    # no cap, autoscale as before.
    'q_dist_ymax': 500000,

    # --- graph 3z: zoomable trace ---------------------------------------
    # graph 3 draws a whole 6.5h session into 12 inches, so ~180k ticks land
    # roughly 1800 to the inch and the five quote lines sit on top of each
    # other. Zooming the PDF does not help: the viewer scales stroke width
    # with the geometry, so they stay just as overlapped at 800%. This writes
    # one extra multi-page PDF per traced day, each page a short window
    # replotted at full width and rescaled on y, which is what actually pulls
    # the lines apart. Smaller window = clearer lines = more pages
    # (a 6.5h session gives ~78 pages at 5 minutes, ~390 at 1 minute).
    'run_trace_zoom': False,
    'trace_zoom_minutes': 5,

    # --- result cache ----------------------------------------------------
    # replayed days are stored in experiment_cache/ keyed by (day, risk, k) and
    # by the (T-t) decay main.spread() currently uses, so re-running the same
    # combination is instant instead of 60-90s per day, and switching the decay
    # to a constant and back does not throw the other one's results away.
    # Entries are invalidated automatically when main.py's model changes.
    # Set False to force a fresh replay (it still refreshes what it computes).
    'use_cache': True,

    # output behaviour
    'show_plots': False,
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'experiment_plots')

# every graph goes out as vector PDF so it stays sharp at any zoom. Graph 3's
# whole-session figure lands around 1.9 MB. Note that vector buys sharpness
# only -- for reading individual quote lines see graph 3z, since zooming cannot
# undo the point density that makes them overlap.

# one fixed colour per strategy, reused on every panel of every figure, so the
# same colour always means the same book no matter which chart you're reading
C_ASYM = 'tab:blue'    # inventory-aware quotes: spread() skews with q
C_SYM = 'tab:orange'   # symmetric quotes: spread_symmetric(), no q skew

# floors on the bin count for graphs 4/5. Graph 5's q is integer-valued, so once
# Q_BINS_MIN exceeds the observed span in shares the bins are narrower than one
# share and the ones that no integer lands in come out empty -- the histogram
# turns into a comb. Lower it to the span (printed at run time) to avoid that.
WEALTH_BINS_MIN = 150
Q_BINS_MIN = 1000


# ============================================================
# Result cache: (day, risk, k) -> the numbers updating() would have returned
#
# A replay costs 60-90s per (day, risk, k) and the result only ever depends on
# that triple plus the model itself, so it is worth keeping on disk. Summaries
# go in one JSON; the intraday traces graphs 3/3z/4/5 need are far too big for
# that, so each gets its own compressed .npz beside it.
#
# Every entry carries a fingerprint of the model that produced it -- main.vars
# plus the source of the functions that decide the numbers. Edit any of them
# and the fingerprint changes, so stale entries are ignored rather than served.
# That is what stops the cache handing back pre-flags-fix results.
#
# The (T-t) decay main.spread() is running is not left to the fingerprint: it
# is probed and written into the key as well (see TDECAY below), so constant-t
# and (1-t) results coexist on disk instead of evicting each other.
# ============================================================
USE_CACHE = True   # CONFIG['use_cache'] overrides this at run time
CACHE_DIR = os.path.join(BASE_DIR, 'experiment_cache')
SUMMARY_PATH = os.path.join(CACHE_DIR, 'summary.json')
SUMMARY_FIELDS = ('X', 'q', 'count', 'wealth', 'sX', 'sq', 'scount', 'swealth')
_TRACE_FLOAT = ('mid', 'pA', 'pB', 'spA', 'spB', 'X', 'sX')
_TRACE_INT = ('q', 'sq')
TRACE_TZ = 'America/New_York'   # main.data() converts ts_event to this


def _fingerprint():
    parts = [str(main.vars)]
    for fn in (main.spread, main.spread_symmetric, main.prices, main.updating, main.data):
        parts.append(inspect.getsource(fn))
    return hashlib.sha256(''.join(parts).encode()).hexdigest()[:12]


FINGERPRINT = _fingerprint()
_summary = None


# ------------------------------------------------------------
# (T-t) tag: which time decay main.py is currently running
#
# main.spread() multiplies the inventory term by a decay factor D(t) -- (1-t)
# for the real Avellaneda-Stoikov horizon, or a constant while the end-of-day
# inventory blow-up is parked. That choice moves every number in a day, but it
# is not an argument to updating(), so it cannot go into the key as one. It is
# read back out of main.spread() instead, by calling it rather than by reading
# its source, so any way of writing the same decay lands on the same tag and
# main.py needs no edit to report it.
#
# The tag joins (day, risk, k) in the key, so a constant run and a (1-t) run
# are stored side by side instead of overwriting each other -- flip main.spread
# back and the earlier sweep is still on disk. FINGERPRINT above still guards
# everything else about the model.
# ------------------------------------------------------------
def _decay(t):
    """The factor main.spread() puts on the inventory term at time t.

    Recovered by differencing, so the rest of the function cannot confuse it:
    the term is (0.5 - q) * risk * var * D(t), so q=0 minus q=1 leaves
    risk * var * D(t) and the log(1 + risk/k) part cancels out.
    """
    risk, var, k = 1e-4, 25.0, 45.0
    return (main.spread(risk, 0, var, t, k)[0]
            - main.spread(risk, 1, var, t, k)[0]) / (risk * var)


def _decay_symmetric(t):
    """Same for main.spread_symmetric(), which has no q to difference over.

    Its term is 0.5 * risk * var * D(t), so two var values difference down to
    0.5 * risk * (v1 - v2) * D(t) and the log part drops out again.
    """
    risk, k = 1e-4, 45.0
    v1, v2 = 25.0, 50.0
    return 2 * (main.spread_symmetric(risk, v1, t, k)[0]
                - main.spread_symmetric(risk, v2, t, k)[0]) / (risk * (v1 - v2))


def _decay_tag(decay):
    """'c<value>' if the decay ignores t, 'v<t=0>-<t=0.9>' if it moves with it.

    The value itself goes in, not just constant-or-not, so swapping the 0.5 for
    0.3 becomes its own cache entry instead of silently replacing the last one.
    """
    d0, d1 = decay(0.0), decay(1.0)
    if abs(d0 - d1) <= 1e-9 * max(1.0, abs(d0)):
        return f'c{d0:.6g}'
    return f'v{d0:.6g}-{decay(0.9):.6g}'


TDECAY = f'{_decay_tag(_decay)}+{_decay_tag(_decay_symmetric)}'   # spread+spread_symmetric


def _key(file_date, risk, k):
    return f'{file_date}|risk={float(risk):.12g}|k={float(k):.12g}|T={TDECAY}'


def _trace_path(file_date, risk, k):
    return os.path.join(
        CACHE_DIR,
        f'trace_{file_date}_r{float(risk):.12g}_k{float(k):.12g}_T{TDECAY}.npz')


def _load_summary():
    global _summary
    if _summary is None:
        try:
            with open(SUMMARY_PATH) as f:
                _summary = json.load(f)
        except (OSError, ValueError):   # missing, or half-written by a killed run
            _summary = {}
    return _summary


def _save_summary():
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = SUMMARY_PATH + '.tmp'      # write-then-rename: a killed run cannot
    with open(tmp, 'w') as f:        # leave a truncated summary.json behind
        json.dump(_summary, f, indent=1, sort_keys=True)
    os.replace(tmp, SUMMARY_PATH)


def _save_trace(path, trace):
    arrays = {'t': pd.DatetimeIndex(trace['t']).asi8}   # ns since epoch, UTC
    arrays.update({key: np.asarray(trace[key], dtype=np.float64) for key in _TRACE_FLOAT})
    arrays.update({key: np.asarray(trace[key], dtype=np.int32) for key in _TRACE_INT})
    np.savez_compressed(path, **arrays)


def _load_trace(path):
    with np.load(path) as z:
        trace = {'t': pd.to_datetime(z['t']).tz_localize('UTC').tz_convert(TRACE_TZ)}
        for key in _TRACE_FLOAT + _TRACE_INT:
            trace[key] = z[key]
    return trace


def cached_updating(data_cache, k, risk, record_trace=False, use_cache=None):
    """main.updating(), but days already on disk for this (risk, k) are reused.

    Same return shape as main.updating(). Days that miss are replayed and then
    written back, so a run only ever pays for what it has not seen before.
    """
    if use_cache is None:
        use_cache = USE_CACHE
    if not use_cache:
        return main.updating(data_cache, k, risk=risk, record_trace=record_trace)

    summary = _load_summary()
    hits, misses = [], {}
    for file_date, df in data_cache.items():
        row = summary.get(_key(file_date, risk, k))
        if row is None or row.get('fp') != FINGERPRINT:
            misses[file_date] = df
            continue
        entry = {field: row[field] for field in SUMMARY_FIELDS}
        entry['date'] = file_date
        if record_trace:
            path = _trace_path(file_date, risk, k)
            if not os.path.exists(path):   # cached summary-only, trace needed now
                misses[file_date] = df
                continue
            entry['trace'] = _load_trace(path)
        hits.append(entry)

    if misses:
        print(f'  cache: reused {len(hits)} day(s), replaying {len(misses)} '
              f'(risk={risk}, k={k})')
        fresh = main.updating(misses, k, risk=risk, record_trace=record_trace)
        os.makedirs(CACHE_DIR, exist_ok=True)
        for entry in fresh:
            row = {field: entry[field] for field in SUMMARY_FIELDS}
            row['fp'] = FINGERPRINT
            summary[_key(entry['date'], risk, k)] = row
            if record_trace and 'trace' in entry:
                _save_trace(_trace_path(entry['date'], risk, k), entry['trace'])
        _save_summary()
        hits.extend(fresh)
    else:
        print(f'  cache: reused all {len(hits)} day(s) (risk={risk}, k={k})')

    return sorted(hits, key=lambda e: e['date'])


def trading_days(s, e):
    """Every trading day main.py has data for in [s, e], as date objects."""
    return sorted(date(int(f[10:14]), int(f[14:16]), int(f[16:18])) for f in main.dates(s, e))


# ============================================================
# Graph 1: X / count / avg end-of-day q vs k, at a fixed risk
# ============================================================
def graph_k_sweep(data_cache, k_values, risk):
    k_list, X_list, wealth_list, count_list, avg_q_list = [], [], [], [], []
    for k in k_values:
        PL = cached_updating(data_cache, k, risk)
        k_list.append(k)
        X_list.append(sum(entry['X'] for entry in PL))
        wealth_list.append(sum(entry['wealth'] for entry in PL))
        count_list.append(sum(entry['count'] for entry in PL))
        avg_q_list.append(np.mean([entry['q'] for entry in PL]) if PL else 0)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(k_list, X_list, marker='o', color='tab:blue', label='X (cash)')
    axes[0].plot(k_list, wealth_list, marker='o', color='tab:red', label='Wealth (X + mid*q)')
    axes[0].set_ylabel('Total P/L')
    axes[0].set_title(f'P/L vs k  (risk={risk})')
    axes[0].legend()
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
# Graph 2: daily X / wealth / q / count, one line per risk value
# (panel data), at a fixed k -- four separate figures
# ============================================================
def graph_risk_panels(data_cache, risk_values, k):
    per_risk = {risk: cached_updating(data_cache, k, risk) for risk in risk_values}

    figs = []
    for metric, ylabel, title in [
        ('X', 'P/L (X)', 'Daily P/L by risk'),
        ('wealth', 'Wealth (X + mid*q)', 'Daily end-of-day wealth by risk'),
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
    return figs  # [fig_X, fig_wealth, fig_q, fig_count]


# ============================================================
# Graph 3: intraday mid / pA / pB, with q underneath, for one day
# ============================================================
def graph_day_trace(data_cache, trace_date, risk, k):
    """One day's trace figure, plus that day's stats for the interval-level plots.

    Returns (fig, stats); stats carries both books' end-of-day wealth and the
    full intraday q path, not just the closing q.
    """
    day_df = data_cache.get(trace_date)
    if day_df is None or day_df.empty:
        print(f'  no data for {trace_date}, skipping trace plot')
        return None, None

    PL = cached_updating({trace_date: day_df}, k, risk, record_trace=True)
    entry = PL[0]
    trace = entry['trace']

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, gridspec_kw={'height_ratios': [3, 1.5, 1.5]})

    # ask sits above mid and bid below it, so side is already unambiguous from
    # position -- colour is spent on the thing you actually want to compare here
    # over a whole session these five lines sit within a few cents of each other,
    # so draw order decides what you can see: mid as the backdrop, then symmetric,
    # then the inventory-aware book on top since that's the one being studied
    axes[0].plot(trace['t'], trace['mid'], label='mid', color='black', linewidth=1)
    axes[0].plot(trace['t'], trace['spA'], label='symmetric spA/spB', color=C_SYM, linewidth=0.8, linestyle='--')
    axes[0].plot(trace['t'], trace['spB'], color=C_SYM, linewidth=0.8, linestyle='--')
    axes[0].plot(trace['t'], trace['pA'], label='A-S pA/pB', color=C_ASYM, linewidth=0.8)
    axes[0].plot(trace['t'], trace['pB'], color=C_ASYM, linewidth=0.8)
    axes[0].set_ylabel('Price')
    axes[0].set_title(f'{trace_date}  --  quotes vs mid  (risk={risk}, k={k})')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(trace['t'], trace['X'], label='A-S', color=C_ASYM, linewidth=1)
    axes[1].plot(trace['t'], trace['sX'], label='symmetric', color=C_SYM, linewidth=1, linestyle='--')
    axes[1].axhline(0, color='black', linewidth=0.6)
    axes[1].set_ylabel('X')
    axes[1].set_title('Running X over the day')
    axes[1].legend()
    axes[1].grid(True)

    # step + fill instead of a plain bar() -- a day has tens of thousands of
    # ticks, and a literal bar per tick is both slow to render and unreadable;
    # this reads the same way (a filled q profile) without either problem
    axes[2].fill_between(trace['t'], trace['q'], step='post', color=C_ASYM, alpha=0.45,
                         label='A-S')
    axes[2].plot(trace['t'], trace['sq'], color=C_SYM, linewidth=1, linestyle='--',
                 label='symmetric')
    axes[2].axhline(0, color='black', linewidth=0.6)
    axes[2].set_ylabel('q')
    axes[2].set_xlabel('Time')
    axes[2].set_title('Inventory (q) over the day')
    axes[2].legend()
    axes[2].grid(True)

    fig.autofmt_xdate()
    fig.tight_layout()

    fig.subplots_adjust(right=0.86)
    fig.text(0.995, 0.5,
             f'Wealth (EOD)\nX + mid*q\n\nA-S\n\\${entry["wealth"]:,.2f}\n\nsymmetric\n\\${entry["swealth"]:,.2f}',
             ha='right', va='center', fontsize=10,
             bbox={'boxstyle': 'round', 'facecolor': 'whitesmoke', 'edgecolor': 'gray'})

    stats = {
        'date': trace_date,
        'wealth': entry['wealth'],
        'swealth': entry['swealth'],
        'q': list(trace['q']),    # every tick's q, not just the close
        'sq': list(trace['sq']),
    }
    return fig, stats


# ============================================================
# Graph 3z: the same day as graph 3, cut into short windows, one per PDF page
# ============================================================
def graph_trace_zoom_pdf(data_cache, trace_date, risk, k, window_minutes, out_path):
    """Write a multi-page PDF for one day: each page is a `window_minutes` slice.

    Graph 3's lines overlap because of point density, not resolution -- a whole
    session is ~180k ticks drawn into 12 inches, and a PDF viewer scales stroke
    width along with the geometry when you zoom, so the overlap never resolves.
    Each page here replots one short window at full width and lets matplotlib
    rescale y to just that window, which is what actually separates the lines.
    """
    day_df = data_cache.get(trace_date)
    if day_df is None or day_df.empty:
        print(f'  no data for {trace_date}, skipping zoom pdf')
        return 0

    trace = cached_updating({trace_date: day_df}, k, risk, record_trace=True)[0]['trace']
    t = pd.DatetimeIndex(trace['t'])
    if len(t) < 2:
        return 0
    col = {key: np.asarray(trace[key], dtype=float)
           for key in ('mid', 'pA', 'pB', 'spA', 'spB', 'X', 'sX', 'q', 'sq')}

    step = pd.Timedelta(minutes=window_minutes)
    start = t[0].floor(f'{window_minutes}min')
    pages = 0
    with PdfPages(out_path) as pdf:
        while start <= t[-1]:
            stop = start + step
            m = (t >= start) & (t < stop)
            if m.sum() < 2:   # gaps and the tail of the session
                start = stop
                continue
            tw = t[m]

            fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                                     gridspec_kw={'height_ratios': [3, 1.5, 1.5]})
            # thicker than graph 3: with only a few hundred points per page
            # there is room, and it reads better than hairlines
            axes[0].plot(tw, col['mid'][m], label='mid', color='black', linewidth=1.2)
            axes[0].plot(tw, col['spA'][m], label='symmetric spA/spB', color=C_SYM, linewidth=1, linestyle='--')
            axes[0].plot(tw, col['spB'][m], color=C_SYM, linewidth=1, linestyle='--')
            axes[0].plot(tw, col['pA'][m], label='A-S pA/pB', color=C_ASYM, linewidth=1)
            axes[0].plot(tw, col['pB'][m], color=C_ASYM, linewidth=1)
            axes[0].set_ylabel('Price')
            axes[0].set_title(f'{trace_date}  {start:%H:%M}-{stop:%H:%M}  --  quotes vs mid  '
                              f'(risk={risk}, k={k})')
            axes[0].legend(loc='upper left')
            axes[0].grid(True)

            axes[1].plot(tw, col['X'][m], label='A-S', color=C_ASYM, linewidth=1.2)
            axes[1].plot(tw, col['sX'][m], label='symmetric', color=C_SYM, linewidth=1.2, linestyle='--')
            axes[1].axhline(0, color='black', linewidth=0.6)
            axes[1].set_ylabel('X')
            axes[1].legend(loc='upper left')
            axes[1].grid(True)

            axes[2].step(tw, col['q'][m], where='post', color=C_ASYM, linewidth=1.2, label='A-S')
            axes[2].step(tw, col['sq'][m], where='post', color=C_SYM, linewidth=1.2, linestyle='--',
                         label='symmetric')
            axes[2].axhline(0, color='black', linewidth=0.6)
            axes[2].set_ylabel('q')
            axes[2].set_xlabel('Time')
            axes[2].legend(loc='upper left')
            axes[2].grid(True)

            fig.autofmt_xdate()
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            pages += 1
            start = stop
    return pages


# ============================================================
# Graph 4: frequency distribution of end-of-day wealth across the traced days
# ============================================================
def graph_wealth_distribution(daily, risk, k):
    wealth = np.array([d['wealth'] for d in daily])
    swealth = np.array([d['swealth'] for d in daily])

    lo = min(wealth.min(), swealth.min())
    hi = max(wealth.max(), swealth.max())
    if hi == lo:  # a single traced day leaves no range to bin over
        lo, hi = lo - 1, hi + 1
    bins = np.linspace(lo, hi, max(WEALTH_BINS_MIN, 2 * len(daily)) + 1)

    fig, ax = plt.subplots(figsize=(11, 5))
    # \$ throughout: an unescaped pair of $ on one line is mathtext to matplotlib,
    # which swallows the signs and italicises whatever sits between them
    ax.hist(wealth, bins=bins, histtype='stepfilled', alpha=0.5, linewidth=1.5,
            color=C_ASYM, edgecolor=C_ASYM,
            label=f'A-S  (total \\${wealth.sum():,.2f}, sd \\${wealth.std():,.2f})')
    ax.hist(swealth, bins=bins, histtype='stepfilled', alpha=0.5, linewidth=1.5,
            linestyle='--', color=C_SYM, edgecolor=C_SYM,
            label=f'symmetric  (total \\${swealth.sum():,.2f}, sd \\${swealth.std():,.2f})')
    ax.axvline(wealth.mean(), color=C_ASYM, linewidth=2,
               label=f'A-S mean  \\${wealth.mean():,.2f}')
    ax.axvline(swealth.mean(), color=C_SYM, linewidth=2, linestyle='--',
               label=f'symmetric mean  \\${swealth.mean():,.2f}')
    ax.axvline(0, color='black', linewidth=0.6)
    ax.set_xlabel('End-of-day wealth (X + mid*q)')
    ax.set_ylabel('Days')
    ax.set_title(f'End-of-day wealth distribution over {len(daily)} day(s)  (risk={risk}, k={k})')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


# ============================================================
# Graph 5: frequency distribution of q over every tick of every traced day
# ============================================================
def graph_q_distribution(daily, risk, k, ymax=None):
    q_all = np.concatenate([d['q'] for d in daily])
    sq_all = np.concatenate([d['sq'] for d in daily])

    lo = min(q_all.min(), sq_all.min())
    hi = max(q_all.max(), sq_all.max())
    # at least Q_BINS_MIN bins, but never coarser than one bin per share: when
    # the span exceeds the floor each bin widens to cover several shares instead
    span = int(hi - lo) + 1
    nbins = max(Q_BINS_MIN, span)
    bins = np.linspace(lo - 0.5, hi + 0.5, nbins + 1)
    if nbins > span:
        print(f'  graph 5: {nbins} bins over a {span}-share span '
              f'({nbins / span:.1f} bins per share -- expect empty bins between integers)')

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(q_all, bins=bins, histtype='stepfilled', alpha=0.5, linewidth=1.5,
            color=C_ASYM, edgecolor=C_ASYM,
            label=f'A-S  (mean {q_all.mean():.2f}, sd {q_all.std():.2f})')
    ax.hist(sq_all, bins=bins, histtype='stepfilled', alpha=0.5, linewidth=1.5,
            linestyle='--', color=C_SYM, edgecolor=C_SYM,
            label=f'symmetric  (mean {sq_all.mean():.2f}, sd {sq_all.std():.2f})')
    ax.axvline(0, color='black', linewidth=0.6)
    ax.set_xlabel('q (shares)')
    ax.set_ylabel('Ticks spent at this q')  # tick-weighted, not time-weighted
    title = f'Inventory distribution over {len(daily)} day(s)  (risk={risk}, k={k})'

    if ymax is not None:
        # clip the view, not the data: every tick still lands in its bin and in
        # the mean/sd above, the bars that run past the cap are just drawn cut off
        n_clipped = sum(int((np.histogram(x, bins=bins)[0] > ymax).sum())
                        for x in (q_all, sq_all))
        ax.set_ylim(0, ymax)
        if n_clipped:
            title += f'  --  y capped at {ymax:,}, {n_clipped} bar(s) cut off'

    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


# ============================================================
# Orchestration
# ============================================================
def main_run():
    global USE_CACHE
    cfg = CONFIG
    USE_CACHE = cfg['use_cache']

    run_sweeps = cfg['run_k_sweep'] or cfg['run_risk_sweep']
    trace_dates = trading_days(cfg['trace_start_date'], cfg['trace_end_date']) if cfg['run_trace'] else []

    # only pay for the days the enabled graphs actually need
    fnames = set(main.dates(cfg['start_date'], cfg['end_date'])) if run_sweeps else set()
    if cfg['run_trace']:
        fnames |= set(main.dates(cfg['trace_start_date'], cfg['trace_end_date']))

    if not fnames:
        print('Every graph is switched off in CONFIG -- nothing to do.')
        return

    print(f'Model: T-t decay {TDECAY} (fingerprint {FINGERPRINT})')
    print(f'Loading {len(fnames)} trading day(s)...')
    data_cache = main.data(list(fnames))

    sweep_cache = {d: df for d, df in data_cache.items() if cfg['start_date'] <= d <= cfg['end_date']}

    os.makedirs(OUT_DIR, exist_ok=True)
    figs = []
    saved = 0

    if cfg['run_k_sweep']:
        print('Running k sweep (graph 1)...')
        fig1 = graph_k_sweep(sweep_cache, cfg['k_values'], cfg['risk_for_k_sweep'])
        fig1.savefig(os.path.join(OUT_DIR, '1_k_sweep.pdf'))
        figs.append(fig1)
        saved += 1

    if cfg['run_risk_sweep']:
        print('Running risk sweep / panel data (graph 2)...')
        fig_X, fig_wealth, fig_q, fig_count = graph_risk_panels(sweep_cache, cfg['risk_values'], cfg['k_for_risk_sweep'])
        fig_X.savefig(os.path.join(OUT_DIR, '2a_risk_panel_X.pdf'))
        fig_wealth.savefig(os.path.join(OUT_DIR, '2b_risk_panel_wealth.pdf'))
        fig_q.savefig(os.path.join(OUT_DIR, '2c_risk_panel_q.pdf'))
        fig_count.savefig(os.path.join(OUT_DIR, '2d_risk_panel_count.pdf'))
        figs.extend([fig_X, fig_wealth, fig_q, fig_count])
        saved += 4

    if cfg['run_trace']:
        print(f'Running intraday trace (graph 3) for {len(trace_dates)} day(s) '
              f'in [{cfg["trace_start_date"]}, {cfg["trace_end_date"]}]...')
        daily = []
        for trace_date in trace_dates:
            fig3, stats = graph_day_trace(data_cache, trace_date, cfg['risk_for_trace'], cfg['k_for_trace'])
            if fig3 is None:
                continue
            out_path = os.path.join(OUT_DIR, f'3_trace_{trace_date}.pdf')
            fig3.savefig(out_path)
            print(f'  saved {os.path.basename(out_path)}   wealth: '
                  f'A-S ${stats["wealth"]:,.2f}   symmetric ${stats["swealth"]:,.2f}')

            if cfg['run_trace_zoom']:
                zoom_path = os.path.join(OUT_DIR, f'3z_trace_zoom_{trace_date}.pdf')
                pages = graph_trace_zoom_pdf(data_cache, trace_date, cfg['risk_for_trace'],
                                             cfg['k_for_trace'], cfg['trace_zoom_minutes'], zoom_path)
                if pages:
                    print(f'  saved {os.path.basename(zoom_path)}   '
                          f'{pages} page(s) at {cfg["trace_zoom_minutes"]} min/page')
                    saved += 1
            # the per-day traces are read from the PNGs, so drop them rather than
            # pile a window per trading day onto plt.show()
            plt.close(fig3)
            saved += 1
            daily.append(stats)

        if daily:
            fig4 = graph_wealth_distribution(daily, cfg['risk_for_trace'], cfg['k_for_trace'])
            fig4.savefig(os.path.join(OUT_DIR, '4_wealth_distribution.pdf'))
            figs.append(fig4)

            fig5 = graph_q_distribution(daily, cfg['risk_for_trace'], cfg['k_for_trace'],
                                        cfg.get('q_dist_ymax'))
            fig5.savefig(os.path.join(OUT_DIR, '5_q_distribution.pdf'))
            figs.append(fig5)
            saved += 2

    print(f'Saved {saved} plot(s) to {OUT_DIR}')
    if cfg.get('show_plots', True) and figs:
        plt.show()


if __name__ == '__main__':
    main_run()
