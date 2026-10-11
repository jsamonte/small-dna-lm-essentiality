# paper/

## What to edit

`manuscript.md` is **generated** — do not edit it directly, your changes will
be overwritten. Edit these three instead:

| File | What it is |
|---|---|
| `sections/abstract.md` | Abstract |
| `sections/introduction.md` | Introduction |
| `sections/discussion.md` | Discussion |

Everything else in the manuscript (Methods, Results, Limitations, References)
is built from `results/` so that no number is ever typed by hand.

## After editing

```
.venv\Scripts\python scripts\build_manuscript.py
```

That rebuilds `manuscript.md` with your prose and the current numbers. Run it
again after any re-analysis.

## Reading the draft

Open `manuscript.md` in VS Code and press Ctrl+Shift+V for a rendered preview.
