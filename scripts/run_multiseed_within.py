#!/usr/bin/env python
"""재현성용 multi-seed within-domain LOSO 러너 (서버 2-GPU).

신경망 보정기 5종(tabm·excelformer·smoothnet·tcn·diffusion)을 16 within-domain 셀 ×
여러 시드로 돌려 mean±std 를 만들 재료(persubject CSV)를 생성.
트리(ExtraTrees)·MLP·평활기는 이미 결정적이라 제외.

- GPU 0/1 두 워커가 작업 큐에서 병렬 소비.
- 이미 결과(persubject_*.csv)가 있으면 스킵 → 중단 후 재실행하면 이어서.
- 기본: 모든 보정기 × 16셀 × seed 0,1,2. 필요시 --correctors/--seeds/--cells 로 축소.

사용(서버):
  ../.venv-nlf/bin/python run_multiseed_within.py --gpus 0 1 --seeds 0 1 2
집계:
  ../.venv-nlf/bin/python aggregate_multiseed.py
"""
import argparse, os, queue, subprocess, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FEAT = HERE.parent / "experiments" / "paper"
OUT = FEAT / "multiseed_out"
PY = str(HERE.parent / ".venv-nlf" / "bin" / "python")

# 16 within-domain 셀: (cell_name, feature_file)
MP_PREFIX = {"aihub": "aihub", "ex6": "ex6", "fit3d": "fit3d", "sume": "sumedipose"}
DOMAINS = ["aihub", "ex6", "fit3d", "sume"]
ESTS = ["offstd", "nlf", "rtmw3d", "motionbert"]

def cells():
    out = []
    for dom in DOMAINS:
        for est in ESTS:
            feat = f"features_{MP_PREFIX[dom]}_offstd.csv" if est == "offstd" else f"features_{dom}_{est}.csv"
            out.append((f"{dom}_{est}", feat))
    return out

# 보정기: name -> (script, persubject_basename)
CORRECTORS = {
    "tabm":        ("tabm_corrector.py",        "persubject_tabm.csv"),
    "excelformer": ("excelformer_corrector.py", "persubject_excelformer.csv"),
    "smoothnet":   ("smoothnet_corrector.py",   "persubject_smoothnet.csv"),
    "tcn":         ("tcn_corrector.py",         "persubject_tcn.csv"),
    "diffusion":   ("diffusion_corrector.py",   "persubject_diffusion.csv"),
}

def build_jobs(correctors, seeds, only_cells):
    jobs = []
    for corr in correctors:
        script, base = CORRECTORS[corr]
        for cell, feat in cells():
            if only_cells and cell not in only_cells:
                continue
            fp = FEAT / feat
            if not fp.exists():
                print(f"[skip] 피처 없음: {feat}"); continue
            for s in seeds:
                odir = OUT / corr / cell / f"seed{s}"
                if (odir / base).exists():
                    continue  # 이미 완료 → 재개 스킵
                jobs.append((corr, script, str(fp), s, odir, base))
    return jobs

def worker(gpu, q, lock):
    while True:
        try:
            corr, script, feat, seed, odir, base = q.get_nowait()
        except queue.Empty:
            return
        odir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        cmd = [PY, script, "--features", feat, "--output", str(odir),
               "--device", "cuda", "--seed", str(seed)]
        t0 = time.time()
        with lock:
            print(f"[GPU{gpu}] START {corr}/{odir.parent.name}/seed{seed}", flush=True)
        log = odir / "run.log"
        with open(log, "w") as lf:
            rc = subprocess.run(cmd, cwd=str(HERE), env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        ok = "OK" if rc == 0 and (odir / base).exists() else f"FAIL(rc={rc})"
        with lock:
            print(f"[GPU{gpu}] {ok} {corr}/{odir.parent.name}/seed{seed} ({time.time()-t0:.0f}s)", flush=True)
        q.task_done()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--correctors", nargs="+", default=list(CORRECTORS))
    ap.add_argument("--cells", nargs="+", default=None, help="특정 셀만 (예: fit3d_rtmw3d ex6_nlf)")
    a = ap.parse_args()

    jobs = build_jobs(a.correctors, a.seeds, a.cells)
    print(f"총 작업 {len(jobs)}개 | GPU {a.gpus} | seeds {a.seeds} | correctors {a.correctors}")
    if not jobs:
        print("할 일 없음(전부 완료됐거나 필터 결과 0)."); return
    q = queue.Queue()
    for j in jobs:
        q.put(j)
    lock = threading.Lock()
    threads = [threading.Thread(target=worker, args=(g, q, lock), daemon=True) for g in a.gpus]
    for t in threads: t.start()
    for t in threads: t.join()
    print("### MULTISEED WITHIN DONE ###")

if __name__ == "__main__":
    main()
