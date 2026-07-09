"""학습 loss 곡선 렌더 — outputs/loss/**/events.out.tfevents.* → PNG.

6종 QLoRA run(Qwen3-1.7B/4B, Kanana-2.1B × v1/v2)의 train/eval loss를 한 장에 겹쳐
리포트용 그림을 만든다. TensorBoard 이벤트만 읽으므로 개인정보·문자 원문 없음(집계지표만).

실행:  python -m src.plot_loss
산출:  outputs/loss/loss_curves.png  (+ loss_summary.csv)
"""
from __future__ import annotations

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tbparse import SummaryReader

# 한글 라벨 깨짐 방지 — Windows 기본 맑은 고딕, 없으면 무시(영문만 정상).
for _f in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
    try:
        plt.rcParams["font.family"] = _f
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

LOSS_DIR = os.path.join("outputs", "loss")

# run 디렉터리명 → (표시라벨, 색, 선스타일). v1=실선, v2=파선.
RUNS = [
    ("pc-1/qwen3-1.7b-v1", "Qwen3-1.7B v1 (주력)", "#1f77b4", "-"),
    ("pc-1/qwen3-1.7b-v2", "Qwen3-1.7B v2",        "#1f77b4", "--"),
    ("pc-2/kanana2.1b-v1", "Kanana-2.1B v1",       "#2ca02c", "-"),
    ("pc-2/kanana2.1b-v2", "Kanana-2.1B v2",       "#2ca02c", "--"),
    ("pc-2/qwen3-4b-v1",   "Qwen3-4B v1",          "#d62728", "-"),
    ("pc-2/qwen3-4b-v2",   "Qwen3-4B v2",          "#d62728", "--"),
]


def _read_scalars(run_dir: str):
    """run 디렉터리의 tfevents를 DataFrame(step,tag,value)으로."""
    path = os.path.join(LOSS_DIR, run_dir)
    df = SummaryReader(path, extra_columns={"dir_name"}).scalars
    return df


def main() -> None:
    fig, (ax_tr, ax_ev) = plt.subplots(1, 2, figsize=(13, 5))
    summary_rows = []

    for run_dir, label, color, ls in RUNS:
        df = _read_scalars(run_dir)
        if df.empty:
            print(f"[skip] 로그 없음: {run_dir}")
            continue

        train = df[df.tag == "train/loss"].sort_values("step")
        # HF Trainer는 eval loss를 'eval/loss'로 기록
        ev = df[df.tag == "eval/loss"].sort_values("step")

        if not train.empty:
            ax_tr.plot(train.step, train.value, color=color, ls=ls, lw=1.4, label=label)
        if not ev.empty:
            ax_ev.plot(ev.step, ev.value, color=color, ls=ls, lw=1.4,
                       marker="o", ms=3, label=label)

        summary_rows.append({
            "run": run_dir,
            "steps": int(train.step.max()) if not train.empty else 0,
            "train_loss_last": round(float(train.value.iloc[-1]), 4) if not train.empty else None,
            "eval_loss_min": round(float(ev.value.min()), 4) if not ev.empty else None,
            "eval_loss_last": round(float(ev.value.iloc[-1]), 4) if not ev.empty else None,
        })
        print(f"[ok] {label:22} steps={summary_rows[-1]['steps']:4} "
              f"train_last={summary_rows[-1]['train_loss_last']} "
              f"eval_min={summary_rows[-1]['eval_loss_min']}")

    for ax, title in ((ax_tr, "Training loss"), (ax_ev, "Eval loss")):
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("annakkyeo — QLoRA 학습 loss (6종, 실선=v1 / 파선=v2)", fontsize=12)
    fig.tight_layout()

    # 리포트용 그림은 커밋 가능한 docs/assets/ 에 저장(집계지표만 → 데이터정책 OK).
    assets_dir = os.path.join("docs", "assets")
    os.makedirs(assets_dir, exist_ok=True)
    out_png = os.path.join(assets_dir, "loss_curves.png")
    fig.savefig(out_png, dpi=130)
    print(f"\n저장: {out_png}")

    # 요약 CSV (집계지표만)
    import csv
    out_csv = os.path.join(LOSS_DIR, "loss_summary.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"저장: {out_csv}")


if __name__ == "__main__":
    main()
