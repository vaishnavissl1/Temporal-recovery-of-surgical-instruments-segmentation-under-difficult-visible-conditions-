# Handoff — Stages 1–2 → Stages 3–5

**Stages 1 and 2 are complete and frozen. Do not edit them.** Everything you
need is behind one module: `sitr/stage34_interface.py`.

---

## Setup on your machine (10 minutes, no training)

1. Clone the repo, create the environment, and install `requirements.txt`.
2. Download the release zip from the shared link and unzip it anywhere. It
   contains the trained models, the per-frame caches **and** the prepared
   images (`release/prepared/`), so you do not need the raw dataset.
3. Confirm it arrived intact:

   ```
   python -m tests.test_pipeline
   python -m tools.verify_release --release <path-to>/release --sequences 4
   ```

   You should see `30 passed` and `RELEASE VERIFIED`, with the image data
   reported as *inside the release: portable*.
4. Load it in your own code with `ReleaseBundle.load("<path-to>/release")`.

Please don't edit files under `sitr/` that Stages 1–2 own. Put your work in new
modules and work on a branch; see "Suggested first commit" at the end.

## The boundary in one equation

Stages 1–2 produce the left-hand side of the report's central decision (eq. 26):

```
S^final_t = S_t       if r_t >= tau     <-  Stage 2 already decides this
          = S-hat_t   if r_t <  tau     <-  YOUR JOB produces this
```

`SelectiveInference.step()` already implements **Algorithm 1 lines 1–6 and
20–23**: run `f_seg`, compute the four indicators, score reliability, take the
threshold decision, and maintain the bounded reliable memory `B_t`. It delegates
**lines 11–19** to you through three Protocols.

Run it today with `recovery=None` and you get the honest Stage 1+2 system:
everything reliable is accepted, everything unreliable is flagged, nothing is
recovered. That is a working baseline you can diff against from your first
commit.

---

## What you implement, in order

### 1. `TemporalAligner` — eq. (16)

```python
def __call__(self, mask_s, gray_s=None, gray_t=None) -> np.ndarray
```

Two implementations already exist in `sitr/align.py`: `IdentityAlign` (the
default — see the caveat below) and `FarnebackAlign` (the ablation of
Sec. V-J.4). **The report's stated default is region-level correspondence via
the appearance + centroid criteria of eq. (24), which is not written yet.**
That is your first task, and it shares code with item 3.

> Why identity is the current default, not laziness: Sec. IV-E argues that at
> the 1 Hz release rate, instrument displacement over ~1 s routinely
> exceeds the range where dense flow stays accurate. Do not quietly switch the
> default to flow without measuring it — the report treats flow as an ablation
> precisely so this is testable.

### 2. `RecoveryModule` — eqs. (17)–(23) — *the paper's main claim*

```python
def __call__(self, state: FrameState) -> RecoveryResult
```

Called **only** when `state.needs_recovery` is True. `state.memory` is `B_t`,
already pruned to recent-and-reliable per eq. (15). You implement:

| eq. | what |
|---|---|
| (18) | weighted aggregation `w_s = r_s · γ^(t−s)` over the memory |
| (19) | convex fusion `P'_t = α_t·P_t + (1−α_t)·P^prop_t` — **on probability maps, not class labels** |
| (20) | `S-hat_t = argmax_c P'_t` |
| (21) | scalar weight `α_t = min(1, r_t/τ)` |
| (22) | per-pixel visibility `ν_t(x,y)` from `state.features` vs the stored embedding |
| (23) | spatially varying `α_t(x,y) = 1 − (1−α_t)·ν_t(x,y)` |

Eqs. (22)–(23) are the part that distinguishes *recoverable* from
*unrecoverable* failure, and the report is explicit that it rests on the
weakest assumption in the design — that backbone features stay discriminative
under exactly the degradations that made the base model fail. **Report the AUC
separating ν_t on recoverable vs unrecoverable events as a primary diagnostic**,
not a footnote. If that separation is weak, Sec. IV-F tells you the fix: a small
dedicated visibility head trained on the recoverable/unrecoverable labels, which
is a change of estimator, not of architecture.

### 3. `IdentityVerifier` — eqs. (24)–(25)

```python
def __call__(self, mask, state: FrameState) -> IdentityResult
```

Hungarian assignment (`scipy.optimize.linear_sum_assignment`) of current
instrument regions to identities held in memory, with cost eq. (24) combining
mask IoU, appearance cosine and normalised centroid distance
(`λ_m + λ_a + λ_d = 1`). **Accept a match only when `C_ij ≤ c_max`;** mark the
region uncertain otherwise. Forcing every region onto an existing identity is
exactly the silent failure of Gap 3. Set `IdentityResult.switches` — Stage 7's
identity continuity metric (eq. 37) reads it.

---

## Contracts you must not break

1. **`SelectiveInference.reset()` between sequences.** Memory crossing a
   procedure boundary is a silent, catastrophic leak. `run_sequence` in
   `sitr/reliability.py` already resets per sequence; match that.
2. **Recovery runs only on `needs_recovery` frames.** If you run it everywhere
   you have built Baseline B, not the proposed method — the comparison the whole
   paper rests on.
3. **Never let ground truth reach inference.** `r*_t` is a training target only
   (eq. 13). Stages 3–5 must never read `cache.r_star`.
4. **Only reliable frames enter memory** (eq. 15, Algorithm 1 line 20–22) —
   and *reliable for this purpose* means `r_s >= tau_memory`, **not**
   `r_s >= tau`. These are two different thresholds answering two different
   questions, and conflating them is a real bug that was caught by
   `tools/verify_release.py` on the first release:

   | | symbol | what it decides | value here |
   |---|---|---|---|
   | routing | `tau`, eq. (2) | is THIS frame sent to recovery? | 0.76 |
   | admission | `tau_memory`, eq. (15) | is this frame good enough to build history from? | 0.50 |

   Stage 2 trains and evaluates with `tau_memory`. If you admit at `tau`
   instead, memory is stricter at inference than it was in training: fewer
   frames enter `B_t` → more cold starts → `T/A/F` fall to zero → `r_t` drops →
   still more frames are routed. A compounding loop, not a small offset — it
   moved the observed routing rate from 0.339 to 0.546. `SelectiveInference`
   and `ReleaseBundle` now keep them separate; don't merge them back.
5. **Masks are class-id maps.** Any warp or resize uses nearest neighbour.
   Interpolating class indices produces classes that do not exist.
6. **τ must be re-swept once recovery exists.** The τ in `stage2_report.json` is
   a documented placeholder chosen by a Youden-J proxy, because the report's
   real criterion (maximise event-level recovery success without degrading
   reliable frames) is not computable from Stages 1–2. Re-sweep it with
   `sitr.reliability.sweep_threshold` replaced by your event-level criterion.

---

## One measured warning before you write a line of Stage 3

Stage 2 fits a small MLP on observable indicators and reports how much each one
contributes. Across every configuration run so far, on this dataset, the result
is consistent and it concerns *you* more than it concerns Stage 2:

| indicator | what it measures | contribution |
|---|---|---|
| `q_t` (and the entropy/margin variants) | posterior confidence, eq. (6) | **carries nearly all the signal** |
| `T_t` temporal consistency, eq. (7) | agreement with the aligned previous mask | weak |
| `A_t` area stability, eq. (8) | frame-to-frame foreground area change | weak, occasionally negative |
| `F_t` feature consistency, eq. (9) | embedding drift | weak |

The gate works, but it works almost entirely on **single-frame confidence**.
The *temporal* indicators — the ones that assume consecutive frames are
similar — add little.

This is not a bug, and it is not a criticism of the report. It is what the
dataset is: EndoVis 2018 is released at **1 Hz** (official challenge paper, arXiv:2001.11190) **with visually similar frames
deliberately removed**. Consecutive frames are far apart in content, so
frame-to-frame temporal agreement is a weak signal here by construction.

**Why it matters to Stages 3–5.** Temporal recovery rests on the same
assumption these indicators are testing: that a nearby past frame is a good
prior for the current one. The indicators say that assumption is weak at this
frame rate. So:

* Expect warping (eq. 16) to be the bottleneck, not the aggregation (eq. 18).
  Budget your effort accordingly.
* Report the **cold-start and unrecoverable rates** prominently. If recovery
  frequently cannot fire, that is a finding about the data, not a failure of
  your code.
* Run the identity check (eqs. 24–25) as a *guard*, not a formality. With weak
  temporal coupling, a propagated mask can plausibly land on the wrong
  instrument.
* If recovery underperforms, test it on a higher-frame-rate source before
  concluding the method fails. The report's falsification criterion 3 is about
  exactly this, and the honest answer may be "the method needs denser frames
  than this release provides".

Say this in the write-up. A measured negative result about the data is worth
more than an unexamined positive one.

---

## What Stages 1–2 give you, per frame

`FrameState` (see the dataclass for exact shapes): the image, `P_t`, `S_t`,
dense features `Φ_t` for eq. (22), the embedding `E_t`, the four indicators,
`r_t`, `reliable`, `a_t`, and the `degenerate` / `cold_start` flags. Plus
`state.memory` = `B_t` as a list of `MemoryEntry` (eq. 14).

Two flags matter to you specifically:

* **`cold_start`** — `B_t` is empty (sequence start, or a run of unreliable
  frames longer than K). Propagation is impossible; the report says flag the
  frame *unrecoverable* rather than fabricating a mask from an empty memory, and
  report the rate. Do not try to recover these.
* **`degenerate`** — `Ω^fg_t` is empty, so eq. (10) has already driven `r_t`
  toward 0. Note the `IoU(∅,∅)` caveat in `NOVELTY.md` §3.5 before treating
  every degenerate frame as a failure: some of them are correct.

## Stages 6–7 need no new model code

Event construction (eq. 32) and the metrics — RSR (34), recovery latency (35),
FRR (36), identity continuity (37), phase-wise IoU (38) — all consume sequences
of `FrameState` plus ground truth. Build them on the cache, not on new forward
passes.

Two things to get right there, because they are where this kind of paper usually
gets shredded:

* **Report RSR and FRR together, always.** RSR alone rewards hallucinating
  instrument pixels into genuinely occluded regions. The report says this
  explicitly; honour it.
* **The event inventory is small.** A single EndoVis 2017 fold yields ~10–20
  events, which is not enough to separate methods. Sec. V-H1 says pool events
  across the full annotated corpus and report bootstrap confidence intervals
  with denominators. Do that from the start rather than retrofitting it.

---

## Suggested first commit

```python
from sitr.stage34_interface import SelectiveInference, RecoveryResult

class PassthroughRecovery:
    """Returns the base prediction untouched. Identical to recovery=None,
    but proves the Protocol wiring before any real logic exists."""
    def __call__(self, state):
        return RecoveryResult(mask=state.mask, source_frames=())
```

Wire it in, confirm the routed-frame count in `SelectiveInference.stats` matches
the routing rate ρ that Stage 2 reported, then replace the body with eq. (18).
If those two numbers disagree, the wiring is wrong and nothing downstream is
worth debugging yet.
