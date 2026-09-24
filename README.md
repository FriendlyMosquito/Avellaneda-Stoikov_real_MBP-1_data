# Avellaneda–Stoikov, rebuilt on real MBP-1 data

A from-scratch implementation of the Avellaneda–Stoikov (2008) optimal market-making
model, backtested tick-by-tick on MBP-1 (top-of-book) equity data, with an
inventory-neutral symmetric quoter running side by side on the same ticks as a control.

The point of the project wasn't to find a profitable strategy. The project takes
closed-form quotes from the paper, feeds them real order flow, and outputs what the
inventory skew actually does to the inventory distribution `q` and the P&L `wealth` distribution
compared to quoting symmetrically around the mid-price.

As will be seen in the distributions and plots later the model does work, although it rarely makes a positive P&L but if the symmetrical strategy experiences losses, A-S loses in that case are significantly smaller. 
The `q` sits on average around zero and doesn't deviate by a lot. While the symmetrical case `q` can go all over the place. This is what the Avellaneda-Stoikov model was created for, managing `q`, and the code reproduces it.
Also I haven't estimated `k` and the appropriate `constant` in place of `T-t`, and **$\sigma^2$** also is a single constant, estimated only once, which also isn't really accurate because the interval was chosen naively. Given these limitations of my implementation the model still has a chance of making a positive P&L while keeping `q` under control.

---

## The model

Both books quote around the mid-price $s$. The Avellaneda–Stoikov book skews with
inventory $q$; the symmetric book does not.

**Reservation price of the market maker**:

$$r(s, q, t) = s - q \gamma \sigma^2 (T - t)$$

**Total spread:**

$$\delta^a + \delta^b = \gamma \sigma^2 (T-t) + \frac{2}{\gamma} \ln\left(1 + \frac{\gamma}{k}\right)$$

**Per side**, (reservation price included):

$$\delta^a = \left(\tfrac{1}{2} - q\right)\gamma\sigma^2(T-t) + \frac{1}{\gamma}\ln\left(1 + \frac{\gamma}{k}\right)$$

$$\delta^b = \left(\tfrac{1}{2} + q\right)\gamma\sigma^2(T-t) + \frac{1}{\gamma}\ln\left(1 + \frac{\gamma}{k}\right)$$

and the quotes are $p^a = s + \delta^a$, $p^b = s - \delta^b$.

For the symmetric control I drop the $q$ term, so both half-spreads are
$\tfrac{1}{2}\gamma\sigma^2(T-t) + \tfrac{1}{\gamma}\ln(1 + \gamma/k)$ and the book
sits centred on the mid-price no matter how much stock it is carrying. Everything else
about the two books is identical, so any difference in the results comes from the
skew alone.

**Parameters**

| Symbol     | Code                | Meaning                                                 | How it is set here                                                        |
| ---------- | ------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------- |
| $\gamma$   | `risk`              | Inventory risk aversion                                 | Free parameter, can be tested in `experiments.py`                         |
| $\sigma^2$ | `vars`              | Price variance over one session (day), in dollars²      | Estimated from the data by `main.var()`                                   |
| $k$        | `k`                 | Order-arrival decay, $\lambda(\delta) = A e^{-k\delta}$ | Free parameter, can be swept in `experiments.py`                          |
| $T - t$    | `1-t` or `constant` | Fraction of the session remaining                       | Can be frozen for an infinite horizon, unfrozen for finite horizon models |

---

### `main.py`

The code for the model itself. Other code only calls on main.py

- `var(s, interval)` — realized variance of the mid over the first `s` days,
  sampled no more often than `interval`, regular trading hours only. Returns the
  average per-session variance in dollars.
- `dates(s, e)` — the filenames in the manifest whose date falls in `[s, e]`.
- `data(dates)` — loads and cleans those days into `{date: DataFrame}`.
- `spread()` / `spread_symmetric()` / `prices()` — the quotes
- `updating(data_cache, k, risk, record_trace)` — the tick replay. Runs the whole day and returns the summary row per day: cash `X`, inventory `q`, fill `count`, `wealth = X+q*last_mid_price`, and the same four for the symmetric book (`sX`, `sq`, `scount`,
  `swealth`). 
- With `record_trace=True` `updating` also returns every tick's mid, both
  books' quotes, and both inventory paths.

### `experiments.py`

The main document for running the code, all variables are changed in config.

**Output:**

| Graph | Output                      | What it shows                                                                                                     |
| ----- | --------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| 1     | `1_k_sweep.pdf`             | Total X, wealth, fill count and average end-of-day q against different $k$ choices, at fixed $\gamma$             |
| 2     | `2a`–`2d_risk_panel_*.pdf`  | Daily X / wealth / q / fills, one line per $\gamma$, at fixed $k$                                                 |
| 3     | `3_trace_<date>.pdf`        | For `[s-e]` sessions. Fixed `k` and $\gamma$. both books' quotes against the mid, running X, running inventory q. |
| 3z    | `3z_trace_zoom_<date>.pdf`  | One day/session cut into short windows, one page each.                                                            |
| 4     | `4_wealth_distribution.pdf` | End-of-day wealth across every traced day, both books compared.                                                   |
| 5     | `5_q_distribution.pdf`      | Inventory across every tick of every traced day, both books compared.                                             |

**The cache.** A replay costs roughly 60–90 seconds per (day, $\gamma$, $k$), and the result depends only on these three variables plus the model, so results
are stored in `experiment_cache/`. Entries carry a fingerprint built from `T-t` equal to paper definition or `constant` and the source of `spread`, `spread_symmetric`, `prices`, `updating` and `data`. Edit any of those and the fingerprint changes. Set `'use_cache': False` to force a replay.

### `interval.py`

Answers "how often should I sample the mid when estimating $\sigma^2$?" Realized
variance of a noisy price is biased upward at short sampling intervals, because
bid–ask bounce gets counted as real price movement. Sampling too coarsely throws
away real variance. This sweeps the interval for the first `s` days of the sample, plot realized volatility against it, and to choose the interval I read off where the curve flattens.

It outputs `rv_vs_interval.png`.

After running it for the first 30 trading days:




I picked the 5 minute interval as it is where in the
regular-session the curve plateaus. While pre- and post-market keeps falling as the interval grows, which makes sense with low trade intensity
resampling the same price over and over. In the end I dropped pre- and post-market and focused only on the regular-session.

### `datacheck.py`

Data quality work, kept in the repo because the conclusions feed directly into the
filters in `main.data()`.

- Counts `flags` values across a day.
- Finds every crossed book (`bid_px_00 > ask_px_00`) across the whole dataset and
  attributes it by `publisher_id`.
- what I found: every crossed print came from **publisher_id 95**, which reads as a
  reporting artefact rather than a real market state, so those rows were dropped.
- Also flags absurd ask prices (4000+ on a stock trading nowhere near that), which
  is where the max-spread filter comes in.
Across the whole dataset there was 5620 crossed ticks, and all of them came from `publisher_id` 95:
	`Counter({95: 5620})`

---

## Data

- **Vendor / dataset:** Databento, schema `mbp-1`
- **Symbol:** MSFT
- **Date range:**
	- **Start:** 2025-01-01 00:00:00 UTC 
	- **End:** 2026-09-04 00:00:00 UTC
	- 420 trading days

The data is **not** in this repo. To run anything you need your own copy with the same layout.

Layout:

**Assumptions the code makes about the layout**

1. `manifest.json` has a `files` list whose **first two entries are `condition.json`
   and `metadata.json`**. Every daily CSV comes after them, which is why the code
   indexes `manifest['files'][2:]`.
2. Daily filenames follow the Databento convention, with the date at characters
   10–17: `xnas-itch-YYYYMMDD.mbp-1.csv`. `dates()` slices the date straight out of
   the filename.
3. Columns used: `ts_event`, `action`, `side`, `price`, `bid_px_00`, `ask_px_00`,
   `publisher_id`, `flags`.
4. `ts_event` is converted to `America/New_York`, so the session boundaries follow
   the exchange through daylight-saving changes.

**Cleaning applied in `main.data()`**

| Filter                  | Reason                                                              |
| ----------------------- | ------------------------------------------------------------------- |
| `action != 'R'`         | Book-clear events are not tradeable states                          |
| `spread < 2`            | Removes the 4000+ ask prints and other quotes that would never fill |
| `bid_px_00 < ask_px_00` | Drops crossed books, all traced to publisher_id 95                  |
| Session `09:30`–`16:00` | Regular trading hours only, applied inside `updating()`             |

---

## Setup

```bash
git clone <your-repo-url>
cd <repo>
python3 -m venv .venv && source .venv/bin/activate
pip install pandas numpy matplotlib
```

 

## Running it

**1. Check the data first.**

```bash
python3 datacheck.py
```

Confirm the crossed-book finding holds on your dataset before trusting the filters.

**2. Pick a sampling interval and estimate variance.**

```bash
python3 interval.py
```

Read the plateau off `rv_vs_interval.png`, then estimate the variance at that
interval and hard-code the result. In `main.py`:

```python
# vars = var(30, interval='5min')   # slow, run once
vars = 24.779591501879146           # dollars per session
```

> The value above is for MSFT over 30 days at a 5-minute sampling interval.
> $\sqrt{24.78} \approx \$4.98$, roughly a $5 standard
> deviation of session price movement. Divide by your price level and confirm the
> implied daily percentage move is plausible.

**3. Run the experiments.**

Edit `CONFIG` at the top of `experiments.py`, then run. Every graph has its own on/off switch. Start with one day for the trace graph (3) only. After confirming that the model works (q is centered around zero) run other tests. Each extra value in `k_values` or `risk_values` multiplies runtime by
(combinations × trading days × ~60–90 s), so widen the lists only once you have
confirmed the setup works.

---

## Results

**Configuration**

|                     |                          |
| ------------------- | ------------------------ |
| Symbol              | MSFT                     |
| Period              | 2025-05-01 __ 2025-06-01 |
| Trading days        | 21                       |
| $\gamma$ (`risk`)   | 0.0004                   |
| $k$                 | 45                       |
| $\sigma^2$ (`vars`) | 24.779591501879146       |
| T-t                 | Constant                 |

|                     |                          |
| ------------------- | ------------------------ |
| Symbol              | MSFT                     |
| Period              | 2025-05-01 __ 2025-06-01 |
| Trading days        | 21                       |
| $\gamma$ (`risk`)   | 0.0004                   |
| $k$                 | 45                       |
| $\sigma^2$ (`vars`) | 24.779591501879146       |
| T-t                 | Paper definition         |

---

|                     |                          |
| ------------------- | ------------------------ |
| Symbol              | MSFT                     |
| Period              | 2025-09-01 __ 2025-10-01 |
| Trading days        | 22                       |
| $\gamma$ (`risk`)   | 0.0004                   |
| $k$                 | 45                       |
| $\sigma^2$ (`vars`) | 24.779591501879146       |
| T-t                 | Constant                 |

|                     |                          |
| ------------------- | ------------------------ |
| Symbol              | MSFT                     |
| Period              | 2025-09-01 __ 2025-10-01 |
| Trading days        | 22                       |
| $\gamma$ (`risk`)   | 0.0004                   |
| $k$                 | 45                       |
| $\sigma^2$ (`vars`) | 24.779591501879146       |
| T-t                 | Paper definition         |

---

|                     |                          |
| ------------------- | ------------------------ |
| Symbol              | MSFT                     |
| Period              | 2026-05-01 __ 2026-06-01 |
| Trading days        | 21                       |
| $\gamma$ (`risk`)   | 0.0004                   |
| $k$                 | 45                       |
| $\sigma^2$ (`vars`) | 24.779591501879146       |
| T-t                 | Constant                 |

|                     |                          |
| ------------------- | ------------------------ |
| Symbol              | MSFT                     |
| Period              | 2026-05-01 __ 2026-06-01 |
| Trading days        | 21                       |
| $\gamma$ (`risk`)   | 0.0004                   |
| $k$                 | 45                       |
| $\sigma^2$ (`vars`) | 24.779591501879146       |
| T-t                 | Paper definition         |

---

|                     |                          |
| ------------------- | ------------------------ |
| Symbol              | MSFT                     |
| Period              | 2026-07-01 __ 2026-08-01 |
| Trading days        | 22                       |
| $\gamma$ (`risk`)   | 0.0004                   |
| $k$                 | 45                       |
| $\sigma^2$ (`vars`) | 24.779591501879146       |
| T-t                 | Constant                 |
|                     |                          |

|                     |                          |     |
| ------------------- | ------------------------ | --- |
| Symbol              | MSFT                     |     |
| Period              | 2026-07-01 __ 2026-08-01 |     |
| Trading days        | 22                       |     |
| $\gamma$ (`risk`)   | 0.0004                   |     |
| $k$                 | 45                       |     |
| $\sigma^2$ (`vars`) | 24.779591501879146       |     |
| T-t                 | Paper definition         |     |

---
## 1 year run:

|                     |                          |
| ------------------- | ------------------------ |
| Symbol              | MSFT                     |
| Period              | 2025-03-01 __ 2026-03-01 |
| Trading days        | 250                      |
| $\gamma$ (`risk`)   | 0.0004                   |
| $k$                 | 45                       |
| $\sigma^2$ (`vars`) | 24.779591501879146       |
| T-t                 | Constant                 |
|                     |                          |

|                     |                          |     |
| ------------------- | ------------------------ | --- |
| Symbol              | MSFT                     |     |
| Period              | 2025-03-01 __ 2026-03-01 |     |
| Trading days        | 250                      |     |
| $\gamma$ (`risk`)   | 0.0004                   |     |
| $k$                 | 45                       |     |
| $\sigma^2$ (`vars`) | 24.779591501879146       |     |
| T-t                 | Paper definition         |     |


---

## Limitations and deviations from the paper

**Model:**

- **$(T-t)$ is frozen at `constant = 0.5`.** In the paper the inventory penalty and
  the spread shrink as the session closes, which is what forces the book flat into
  the bell. In the last commit both terms are constant all day, so the model has no reason to
  unwind. `updating()` already computes the normalised session time `t` it can be plugged in the calculations of spread if wanted.
- **$k$ and $A$ are not estimated.** The paper fits the order-arrival intensity
  $\lambda(\delta) = Ae^{-k\delta}$ to the flow. Here $k$ is a free parameter swept
  over a range and $A$ never appears because we use real data, so $k$ is a knob rather than a measurement.
- **$\sigma^2$ is a single constant** for the entire backtest period, estimated once
  and hard-coded.

**Simplification:**

- **No queue position.** Any trade that crosses the quoted price counts as a fill.
  In reality the order joins the back of a queue at that price. This is the least realistic assumption, but I am limited by my data.
- **No latency.** Quotes are recomputed from the current mid on every single tick
  and are assumed live instantly.
- **Unit size.** Every fill moves $q$ by exactly one share, regardless of the size
  actually printed.
- **No fees, rebates, or borrow cost.** Rebates and fees would change the models P&L.
- **No position limits and no short constraint.** $q$ is free to run in either
  direction.
- **Fill direction follows the vendor's aggressor convention:** a trade with
  `side == 'B'` is treated as a buyer lifting the offer (so our ask is hit), and
  `side == 'A'` as a seller hitting the bid.

**Accounting**

- End-of-day wealth is marked at the session's last mid, `X + last_mid * q`. Any
  inventory left at the close is valued rather than liquidated. The exit q is simulated to be liquidated (not checked on ask/bid prices and quantity traded)
- `X` alone is cash and is not a P/L figure on its own; compare `wealth`.

## References

- Avellaneda, M. and Stoikov, S. (2008). *High-frequency trading in a limit order
  book.* Quantitative Finance, 8(3), 217–224.


---

## Licence

MIT

Data is not covered by it
> and is not redistributed here.

## Author

> **Fill in:** Nedas Virbickas
