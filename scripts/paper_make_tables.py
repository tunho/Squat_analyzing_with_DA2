from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def write_markdown_table(path: Path, title: str, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [str(col) for col in df.columns]
    rows = []
    for _, row in df.iterrows():
        rows.append([str(row[col]) for col in df.columns])

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(row) + " |\n")
        f.write("\n")


def add_summary_row(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number")
    row = {col: "" for col in df.columns}
    row["subject_id"] = "mean" if "subject_id" in row else "mean"
    for col in numeric.columns:
        row[col] = float(df[col].mean())
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)


def save_bar_plot(df: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    plot_df = df[df["subject_id"] != "mean"].copy()
    if plot_df.empty:
        return

    ax = plot_df.plot(
        x="subject_id",
        y=["mae_raw", "mae_corrected"],
        kind="bar",
        figsize=(10, 5),
        rot=35,
    )
    ax.set_ylabel("MAE (degrees)")
    ax.set_xlabel("Subject")
    ax.set_title("LOSO Raw vs Corrected MAE")
    ax.legend(["Raw MediaPipe", "Corrected"])
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close()


def save_scatter(predictions: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    sample = predictions.dropna(subset=["gt_angle", "mp_knee_angle", "corrected_angle"])
    if sample.empty:
        return
    if len(sample) > 8000:
        sample = sample.sample(8000, random_state=42)

    plt.figure(figsize=(6, 6))
    plt.scatter(sample["gt_angle"], sample["mp_knee_angle"], s=5, alpha=0.25, label="Raw")
    plt.scatter(sample["gt_angle"], sample["corrected_angle"], s=5, alpha=0.25, label="Corrected")
    lo = min(sample["gt_angle"].min(), sample["mp_knee_angle"].min(), sample["corrected_angle"].min())
    hi = max(sample["gt_angle"].max(), sample["mp_knee_angle"].max(), sample["corrected_angle"].max())
    plt.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    plt.xlabel("GT angle")
    plt.ylabel("Predicted angle")
    plt.legend()
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper tables and figures from LOSO outputs.")
    parser.add_argument("--loso", required=True, type=Path, help="LOSO metrics CSV.")
    parser.add_argument("--predictions", type=Path, default=None, help="Optional LOSO predictions CSV.")
    parser.add_argument("--summary", type=Path, default=None, help="Optional summary_metrics.json.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    loso_df = pd.read_csv(args.loso)
    table_df = add_summary_row(loso_df)
    table_df.to_csv(args.output / "table_loso_metrics.csv", index=False)
    write_markdown_table(args.output / "table_loso_metrics.md", "LOSO Metrics", table_df)

    if args.summary and args.summary.exists():
        with open(args.summary, "r", encoding="utf-8") as f:
            summary = json.load(f)
        pd.DataFrame([summary]).to_json(args.output / "summary_metrics_flat.json", orient="records", indent=2)

    try:
        save_bar_plot(table_df, args.output / "fig_loso_mae_bar.png")
        if args.predictions and args.predictions.exists():
            save_scatter(pd.read_csv(args.predictions), args.output / "fig_gt_scatter.png")
    except ImportError as exc:
        print(f"Skipping figures because a plotting dependency is missing: {exc}")

    print(f"Saved tables: {args.output}")


if __name__ == "__main__":
    main()
