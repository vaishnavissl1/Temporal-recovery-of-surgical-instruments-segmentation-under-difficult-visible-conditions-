# Run the batch demo on your own computer

The Analyse page uploads a folder of consecutive stereo frames to a small server on your computer. The server runs the released pipeline over them in order and the page shows every result. Nothing leaves your machine: the server listens on `127.0.0.1` only, and it deletes the uploaded files as soon as the run ends.

This is a research prototype on public EndoVis 2018 data. It is not a medical device.

## What you need (once)

- Windows, macOS or Linux, no GPU required (the demo runs on the CPU).
- The `sitr` conda environment (Python with PyTorch, OpenCV, scikit-learn): `conda activate sitr`.
- Node.js 18 or newer, for the website. In `web/`, run `npm install` once.
- The trained models and caches in `work_v2/` (the release folder `work_v2/release`, the out-of-fold action data `work_v2/actions`) and `results/task2/stereo_check.json`. They are not in the git repository because they are large; they are on the machine that produced the results.
- The demo folder `demo_subset/` in the repository root (24 frames, 76 MB).

## The five steps, in order

**1. Start the analysis server** (terminal 1, from the repository root):

```
conda activate sitr
python -m tools.serve_analyse --device cpu
```

Wait for the line `ready in ... s on http://127.0.0.1:8765  device=cpu`. Loading and fitting the selector takes about 10 to 15 seconds. Leave this terminal open.

**2. Start the website** (terminal 2):

```
cd web
npm run dev
```

Wait for `Ready`. The file `web/.env.development` already points the page at `http://127.0.0.1:8765`; you change nothing.

**3. Open this address in a browser:**

```
http://localhost:3000/analyse/
```

The first tab, "Analyse a folder (full pipeline)", is selected.

**4. Choose the folder.** Under "1. Choose the folder", click the **Folder** control and select `demo_subset` (the folder in the repository root, the one that contains `left_frames`, `right_frames`, `labels` and `camera_calibration.txt`). A zip of that folder works too, through "or a zip". The page then shows what you selected, for example `Selected: 73 files, 75.1 MB (24 left frames, 24 right frames, 24 label files, calibration found)`. Nothing runs yet.

**5. Click Analyse.** A progress bar and the current step appear while it runs. When it finishes, the results appear below: the per-frame table, the summary numbers, the reliability-against-IoU chart, the per-class IoU, and a **Download results CSV** button.

## How long it takes

For the 24 frames of `demo_subset`, **about 1 to 3 minutes** on a laptop CPU. We measured 60 s on a quiet machine and 130 to 150 s on a busy one; on the quiet run that is 2.5 s per frame. Stereo matching and segmentation take most of it. (If 24 frames had taken longer than 4 minutes, the default subset would have been cut to 12; they did not.)

## What you should see for `demo_subset`

| | value |
|---|---|
| frames | 24 (official test seq_1, frames 171 to 194) |
| flagged by the gate (reliability below 0.64) | 6 of 24 (25.0%) |
| corrected | 3 of 24 (12.5%): temporal on 2, refine on 1 |
| official mIoU, before and after | 0.5953 and 0.5954 |
| mcIoU, before and after | 0.6004 and 0.6005 |
| corrections that helped, hurt, left unchanged | 0, 0, 3 |
| harmful-correction rate | 0.0% of corrected, 0.0% of all frames |
| needle IoU | 0.000 (the paper's known weakness) |

These match the paper's own stored masks for the same 24 frames: the same 24 flag decisions, the same 24 action choices, and mIoU and mcIoU equal to four decimals. The reliability scores agree to a mean absolute difference of 0.003 over the 24 frames, almost all of it frame 171 (0.062): the memory is empty at the first uploaded frame, where the paper's run had already seen frames 0 to 170.

## What each part of the page is

- **Per frame:** frame name, reliability score, flagged yes or no, the action chosen (none, A2 temporal, A3 refine), instrument IoU before and after, and whether the change helped, hurt or did nothing (a change within 0.01 counts as nothing).
- **Summary:** official mIoU and mcIoU over the uploaded frames (official rule, full resolution), activation rate, harmful-correction rate with both denominators (of corrected frames and of all frames), mean latency.
- **Identity verification:** the shipped Stage 5 returns the mask unchanged by design, so the page reports how many regions it matched and how many it left uncertain, and that it changed 0 masks.
- **No `labels/` folder?** Every stage still runs, and the page says nothing was scored instead of showing empty metrics.

## Things to know

- The frames must be consecutive and from one sequence. The reliability gate compares each frame with the last three reliable ones, so on such a folder it runs exactly as in the paper. (The separate single-frame tool cannot, so it reports confidence only.)
- Two stated differences from the paper's run: the memory starts empty, and the stereo self-calibration uses the first few frames of your folder instead of the first 36 of the whole video.
- Latency is measured on your CPU and is not comparable with the paper's GPU numbers.
- To use a GPU instead, start the server with `--device cuda`.
- The public website shows a short note instead of the folder tool, because the models are not published; this tool is demonstrated live.

## If something goes wrong

| symptom | what to do |
|---|---|
| page says it "could not reach the local analysis server" | the server in terminal 1 is not running, or is not on port 8765; start it (step 1) |
| the Analyse button is greyed out | the selection is missing `left_frames/`, `right_frames/` or `camera_calibration.txt`; the line under the picker says which |
| `Address already in use` when starting the server | another copy is running; stop it, then start again |
| the page shows no folder tool, only a note | the site was started with `npm run build` instead of `npm run dev`; use `npm run dev` |
