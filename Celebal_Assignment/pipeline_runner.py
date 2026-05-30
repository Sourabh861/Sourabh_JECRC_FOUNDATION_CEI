# PIPELINE RUNNER — Master Orchestrator
# Runs all 7 steps in order.

import importlib, time, traceback, argparse, os, sys

# Add current folder to path so Python can find the step modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STEPS = {
    1: ("Data Loader",           "data_loader"),
    2: ("Preprocessing",         "preprocessing"),
    3: ("EDA",                   "eda"),
    4: ("Feature Engineering",   "feature_engineering"),
    5: ("Regression Modeling",   "regression_modeling"),
    6: ("Hyperparameter Tuning", "hyperparameter_tuning"),
    7: ("Time Series Forecast",  "time_series_forecasting"),
}

def run(step_num, name, module):
    print(f"\n{'='*50}")
    print(f"  STEP {step_num}/7 — {name.upper()}")
    print(f"{'='*50}")
    t0 = time.time()
    try:
        importlib.import_module(module).main()
        elapsed = time.time() - t0
        print(f"  ✓ Done in {elapsed:.1f}s")
        return True, elapsed
    except Exception:
        elapsed = time.time() - t0
        print(f"  ✗ FAILED after {elapsed:.1f}s:")
        traceback.print_exc()
        return False, elapsed

def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--from", dest="from_step", type=int, default=1)
    g.add_argument("--only", dest="only_step", type=int)
    p.add_argument("--skip", dest="skip", nargs="+", type=int, default=[])
    args = p.parse_args()

    to_run = [args.only_step] if args.only_step else \
             [s for s in STEPS if s >= args.from_step and s not in args.skip]

    print("\n" + "="*50)
    print("  TESLA DELIVERIES — ML PIPELINE")
    print("="*50)
    print(f"  Steps to run: {to_run}")

    results, t_start = {}, time.time()
    for s in to_run:
        name, module = STEPS[s]
        ok, elapsed  = run(s, name, module)
        results[s]   = (name, elapsed, ok)
        if not ok:
            print(f"\n  Pipeline stopped at Step {s}.")
            print(f"  Fix the error, then re-run: python pipeline_runner.py --from {s}")
            break

    # Summary
    print(f"\n{'='*50}  SUMMARY  {'='*50}")
    all_ok = True
    for s, (name, elapsed, ok) in sorted(results.items()):
        status = "✓" if ok else "✗ FAILED"
        print(f"  Step {s}: {name:<28} {status}  ({elapsed:.1f}s)")
        if not ok: all_ok = False
    print(f"\n  Total time : {time.time()-t_start:.1f}s")
    print(f"  Result     : {'ALL STEPS PASSED ✓' if all_ok else 'SOME STEPS FAILED ✗'}")

    for folder in ["outputs", "plots"]:
        if os.path.isdir(folder):
            files = os.listdir(folder)
            print(f"\n  {folder}/ ({len(files)} files): {', '.join(sorted(files))}")

if __name__ == "__main__":
    main()