"""
experiments/run_sia_attack.py
Run SIA on the subset. Same CSV output format as run_vanilla_subset_generation.py.

Usage (from repo root):
  python experiments/run_sia_attack.py \
      --pairs      docs/subset_input_pairs.csv \
      --faces_dir  dataset_extractedfaces \
      --thresholds core/verification_thresholds.json \
      --output_dir experiments/results_new/
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.sia_attack import sia_attack, cosine_sim_np

ATTACKERS = ["Facenet512", "ArcFace", "VGG-Face"]
VICTIMS   = ["Facenet", "Facenet512", "ArcFace", "VGG-Face"]
SKIP = {
    "Facenet512":   {"Facenet512", "Facenet"},
    "ArcFace":      {"ArcFace"},
    "VGG-Face":     {"VGG-Face"},
}

def load_img(p):
    return np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0

def get_emb(model_name, img_np):
    from deepface import DeepFace
    u8 = (img_np * 255).clip(0, 255).astype(np.uint8)
    r = DeepFace.represent(img_path=u8, model_name=model_name,
                           enforce_detection=False, detector_backend="skip")
    return np.array(r[0]["embedding"], dtype=np.float32)

def fix_path(rel, faces_dir):
    """Strip any absolute prefix, join with faces_dir."""
    p = str(rel).strip()
    for prefix in ["/content/face_module/", "/content/", "dataset_extractedfaces/"]:
        if p.startswith(prefix):
            p = p[len(prefix):]
    return os.path.join(faces_dir, p.lstrip("/\\"))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs",       required=True)
    p.add_argument("--faces_dir",   required=True)
    p.add_argument("--thresholds",  required=True)
    p.add_argument("--output_dir",  default="experiments/results_new")
    p.add_argument("--epsilon",     type=float, default=0.05)
    p.add_argument("--steps",       type=int,   default=10)
    p.add_argument("--copies",      type=int,   default=20)
    p.add_argument("--grid",        type=int,   default=4)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    pairs = pd.read_csv(args.pairs)
    with open(args.thresholds) as f:
        thresh = json.load(f)
    print(f"Loaded {len(pairs)} pairs | output -> {args.output_dir}")

    rows = []
    for idx, row in pairs.iterrows():
        if "img1" in row and "img2" in row:
            i1 = str(row["img1"])
            i2 = str(row["img2"])
        else:
            i1 = str(row.iloc[0]); i2 = str(row.iloc[1])
        goal = str(row.get("attack_type", row.get("goal", "impersonation"))).strip().lower()
        if goal.endswith("_attack"):
            goal = goal[: goal.rfind("_attack")]
        if goal == "":
            goal = "impersonation"
        dataset = str(row.get("dataset", "digiface")).strip()
        p1   = fix_path(i1, args.faces_dir)
        p2   = fix_path(i2, args.faces_dir)
        if not os.path.exists(p1) or not os.path.exists(p2):
            print(f"[{idx+1}] SKIP missing: {p1}"); continue
        img1 = load_img(p1); img2 = load_img(p2)
        print(f"\n[{idx+1}/{len(pairs)}] goal={goal} dataset={dataset}")

        for atk in ATTACKERS:
            at  = "impersonation_attack" if goal=="impersonation" else "dodging_attack"
            try:
                tgt = get_emb(atk, img2 if goal=="impersonation" else img1)
            except Exception as e:
                print(f"  [{atk}] emb error: {e}"); continue

            t0 = time.time()
            try:
                adv = sia_attack(atk, img1, tgt, attack_type=at,
                                 eps=args.epsilon, steps=args.steps,
                                 num_copies=args.copies, grid=args.grid)
            except Exception as e:
                print(f"  [{atk}] attack error: {e}"); continue
            print(f"  {atk}: {time.time()-t0:.1f}s")

            for vic in VICTIMS:
                if vic in SKIP.get(atk, set()): continue
                try:
                    ae = get_emb(vic, adv)
                    re = get_emb(vic, img2)
                    oe = get_emb(vic, img1)
                    rows.append({
                        "pair_id": idx, "img1": i1, "img2": i2, "goal": goal,
                        "attack": "SIA", "attacker_model": atk, "victim_model": vic,
                        "dataset": dataset,
                        "sim_adv_vs_target":  cosine_sim_np(ae, re),
                        "sim_orig_vs_target": cosine_sim_np(oe, re),
                    })
                except Exception as e:
                    print(f"    [{vic}] error: {e}")

    if not rows:
        print("No results — check --faces_dir path"); return

    df = pd.DataFrame(rows)
    df.to_csv(f"{args.output_dir}/SIA_raw_similarities_long.csv", index=False)

    def breach(r):
        th = thresh.get(r["victim_model"])
        if not th:
            return 0
        ds = r.get("dataset", "digiface")
        if isinstance(th, dict):
            th = th.get(ds) or th.get("digiface") or next(iter(th.values()), {})
        if isinstance(th, dict):
            th = th.get("threshold")
        if not isinstance(th, (int, float)):
            return 0
        return int(r["sim_adv_vs_target"] >= th) if r["goal"] == "impersonation" \
               else int(r["sim_adv_vs_target"] < th)

    df["breach"] = df.apply(breach, axis=1)
    df.to_csv(f"{args.output_dir}/SIA_attack_eval_long.csv", index=False)

    s = (df.groupby("attack")["breach"]
         .agg(total="count", breaches="sum")
         .assign(breach_rate=lambda d:(d["breaches"]/d["total"]*100).round(2))
         .reset_index())
    s.to_csv(f"{args.output_dir}/SIA_attack_summary.csv", index=False)

    bg = (df.groupby(["attack","goal"])["breach"]
          .agg(total="count", breaches="sum")
          .assign(breach_rate=lambda d:(d["breaches"]/d["total"]*100).round(2))
          .reset_index())
    bg.to_csv(f"{args.output_dir}/SIA_attack_summary_by_goal.csv", index=False)

    av = (df.groupby(["attack","attacker_model","victim_model"])["breach"]
          .agg(total="count", breaches="sum")
          .assign(breach_rate=lambda d:(d["breaches"]/d["total"]*100).round(2))
          .reset_index())
    av.to_csv(f"{args.output_dir}/SIA_attacker_victim_summary.csv", index=False)

    df["emb_impact"] = df["sim_adv_vs_target"] - df["sim_orig_vs_target"]
    impact = df.groupby("attack")["emb_impact"].mean().reset_index()
    impact.to_csv(f"{args.output_dir}/SIA_embedding_impact.csv", index=False)

    print(f"\n{'='*40}\n{s.to_string(index=False)}")
    print(f"Mean embedding impact: {df['emb_impact'].mean():.4f}")

if __name__ == "__main__":
    main()