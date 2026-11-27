"""Unit tests tích hợp (Smoke + Normal) cho Training, Resume, Eval, 4-Step Error Analysis, Inference & Notebook Lab 2 v2."""

import json
from pathlib import Path

import nbformat
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.vibe_scripts.lab2_v2.data import OxfordPetsBundle
from src.vibe_scripts.lab2_v2.evaluator import (
    evaluate_final_test,
    plot_error_taxonomy_breakdown,
    run_deep_error_analysis,
)
from src.vibe_scripts.lab2_v2.inference import PetPredictor
from src.vibe_scripts.lab2_v2.models import build_model
from src.vibe_scripts.lab2_v2.trainer import (
    _persist_ablation_summary,
    run_sanity_checks,
    set_global_seed,
    train_single_run,
)
from src.vibe_scripts.lab2_v2.transforms import ImageNetNormStats


def _make_synthetic_bundle(num_classes: int = 4, n_train: int = 16, n_val: int = 8, n_test: int = 8) -> OxfordPetsBundle:
    set_global_seed(248)
    class_names = tuple(f"Breed_{i}" for i in range(num_classes))
    species = {name: ("Cat" if i % 2 == 0 else "Dog") for i, name in enumerate(class_names)}

    def _ds(n: int) -> TensorDataset:
        y = torch.tensor([i % num_classes for i in range(n)], dtype=torch.long)
        x = torch.randn((n, 3, 224, 224), dtype=torch.float32) * 0.25
        # Tín hiệu nhẹ theo lớp để overfit 1 batch nhanh
        for idx in range(n):
            c = int(y[idx].item())
            x[idx, c % 3, :16, :16] += 1.5
        return TensorDataset(x, y)

    return OxfordPetsBundle(
        train_loader=DataLoader(_ds(n_train), batch_size=8, shuffle=True),
        val_loader=DataLoader(_ds(n_val), batch_size=8, shuffle=False),
        test_loader=DataLoader(_ds(n_test), batch_size=8, shuffle=False),
        class_names=class_names,
        species_by_class=species,
        norm_stats=ImageNetNormStats(),
        split_sizes={"train": n_train, "val": n_val, "test": n_test},
        image_paths={"train": [], "val": [], "test": []},
        labels={"train": [], "val": [], "test": []},
    )


@pytest.mark.smoke
def test_single_run_per_run_eval_and_inference_smoke(tmp_path: Path) -> None:
    """Smoke: chạy 1 epoch trên synthetic data, tạo historys/ + eval/ + inference Top-K pass."""
    bundle = _make_synthetic_bundle()
    spec = {
        "run_id": "smoke_r18",
        "group": "Smoke",
        "axis": "smoke",
        "arch": "resnet18",
        "mode": "transfer",
        "pretrained": False,
        "lr": 1e-3,
        "lr_backbone": 0.0,
        "weight_decay": 1e-4,
        "batch_size": 8,
        "epochs": 1,
        "is_baseline": False,
    }
    res = train_single_run(
        run_spec=spec,
        bundle=bundle,
        runs_dir=tmp_path / "runs",
        device="cpu",
        seed=248,
        use_amp=False,
        tb_logs_dir=tmp_path / "tb_logs",
        enable_tensorboard=True,  # Phải fallback mượt nếu chưa cài tensorboard
        show_progress=False,
        eval_after_train=True,
        save_eval_plots=True,
    )
    run_dir = Path(res["run_dir"])
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "last.pt").exists()
    assert (run_dir / "historys" / "history.json").exists()
    assert (run_dir / "historys" / "history.csv").exists()
    assert (run_dir / "eval" / "eval_metrics.json").exists()
    assert (run_dir / "eval" / "loss_curve.png").exists()
    assert res["save_load_passed"] is True

    predictor = PetPredictor(run_dir / "best.pt", device="cpu")
    pred = predictor.predict_tensor(torch.randn((3, 224, 224)), top_k=3)[0]
    assert len(pred["top_k"]) == 3
    assert pred["requires_grad"] is False
    assert pred["species"] in ("Cat", "Dog")


def test_sanity_checks_and_resume_training_preserves_history(tmp_path: Path) -> None:
    """Normal: kiểm chứng Sanity Checks & cơ chế --resume nối liên tục epochs + khôi phục optimizer_state_dict."""
    bundle = _make_synthetic_bundle(num_classes=4, n_train=8)
    model = build_model("resnet18", mode="scratch", num_classes=4, pretrained=False)
    sanity = run_sanity_checks(model, bundle.train_loader, device="cpu", lr=2e-3, overfit_steps=45)
    assert sanity["init_loss_passed"] is True
    assert sanity["overfit_passed"] is True

    spec = {
        "run_id": "base_r18",
        "group": "Test",
        "axis": "test",
        "arch": "resnet18",
        "mode": "transfer",
        "pretrained": False,
        "lr": 1e-3,
        "lr_backbone": 0.0,
        "weight_decay": 1e-4,
        "batch_size": 8,
        "epochs": 1,
        "is_baseline": True,
    }
    runs_dir = tmp_path / "runs"
    r1 = train_single_run(
        run_spec=spec,
        bundle=bundle,
        runs_dir=runs_dir,
        device="cpu",
        use_amp=False,
        enable_tensorboard=False,
        show_progress=False,
        save_eval_plots=False,
    )
    r2 = train_single_run(
        run_spec=spec,
        bundle=bundle,
        runs_dir=runs_dir,
        device="cpu",
        use_amp=False,
        enable_tensorboard=False,
        show_progress=False,
        save_eval_plots=True,
        resume_from=Path(r1["run_dir"]) / "last.pt",
        extra_epochs=2,
        output_run_id="base_r18_ep3",
    )
    hist = json.loads((Path(r2["run_dir"]) / "historys" / "history.json").read_text(encoding="utf-8"))
    assert [e["epoch"] for e in hist["epochs"]] == [1, 2, 3]
    assert hist["resumed"]["optimizer_state_restored"] is True


def test_final_test_no_duplicate_curves_and_deep_error_analysis(tmp_path: Path) -> None:
    """Normal: test_eval không vẽ lặp curves, model selection ưu tiên val_acc, và 4-step Error Analysis chạy sạch."""
    bundle = _make_synthetic_bundle(num_classes=4, n_train=16, n_val=8, n_test=12)
    runs_dir = tmp_path / "runs"
    base_spec = {
        "run_id": "resnet18_scratch",
        "group": "Baseline",
        "axis": "baseline",
        "arch": "resnet18",
        "mode": "scratch",
        "pretrained": False,
        "lr": 1e-3,
        "lr_backbone": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 8,
        "epochs": 1,
        "is_baseline": True,
    }
    best_spec = {
        **base_spec,
        "run_id": "resnet18_transfer",
        "group": "Transfer",
        "mode": "transfer",
        "is_baseline": False,
    }
    r_base = train_single_run(base_spec, bundle, runs_dir, device="cpu", use_amp=False, enable_tensorboard=False, show_progress=False, save_eval_plots=False)
    r_best = train_single_run(best_spec, bundle, runs_dir, device="cpu", use_amp=False, enable_tensorboard=False, show_progress=False, save_eval_plots=False)

    # Kiểm chứng quy tắc chọn ưu tiên val_acc -> val_macro_f1 -> -val_loss
    fake_low_acc_low_loss = {**r_best, "run_id": "cand_low_loss", "val_acc": 0.80, "val_macro_f1": 0.80, "val_loss": 0.10}
    fake_hi_acc = {**r_best, "run_id": "cand_hi_acc", "val_acc": 0.90, "val_macro_f1": 0.89, "val_loss": 0.30}
    summary = _persist_ablation_summary(tmp_path / "ablation", [r_base, fake_low_acc_low_loss, fake_hi_acc], total_planned=3)
    assert summary["winning_model_run_id"] == "cand_hi_acc"

    final_out = evaluate_final_test(
        runs_dir=runs_dir,
        baseline_run_id="resnet18_scratch",
        best_run_id="resnet18_transfer",
        test_loader=bundle.test_loader,
        device="cpu",
        evaluation_dir=tmp_path / "evaluation",
        figures_dir=tmp_path / "figures",
        class_names=bundle.class_names,
        species_by_class=bundle.species_by_class,
        ablation_summary_path=tmp_path / "ablation" / "ablation_summary.json",
    )
    assert (tmp_path / "evaluation" / "test_comparison.json").exists()
    assert (tmp_path / "evaluation" / "species_analysis.json").exists()
    # Không vẽ lặp learning curves vào test_eval/
    assert not (runs_dir / "resnet18_transfer" / "test_eval" / "loss_curve.png").exists()
    assert not (runs_dir / "resnet18_transfer" / "test_eval" / "accuracy_curve.png").exists()
    assert final_out["best_run_id"] == "resnet18_transfer"

    ea = run_deep_error_analysis(
        run_dir=runs_dir / "resnet18_transfer",
        eval_loader=bundle.test_loader,
        device="cpu",
        evaluation_dir=tmp_path / "evaluation",
        figures_dir=tmp_path / "figures",
        class_names=bundle.class_names,
        species_by_class=bundle.species_by_class,
        gallery_size=4,
        audit_target_total=6,
    )
    assert (tmp_path / "figures" / "confusion_matrix_best_normalized.png").exists()
    assert (tmp_path / "figures" / "error_slice_analysis.png").exists()
    assert (tmp_path / "evaluation" / "slice_analysis.json").exists()

    tax = plot_error_taxonomy_breakdown(
        [
            {"cause": "model_limitation", "conf_group": "high_conf"},
            {"cause": "ambiguity", "conf_group": "low_conf"},
            {"cause": "label_noise", "conf_group": "high_conf"},
        ],
        output_path=tmp_path / "figures" / "error_taxonomy_breakdown.png",
    )
    assert tax["total_audited"] == 3 and Path(tax["figure"]).exists()


def test_experiment_notebook_schema_valid() -> None:
    """Normal: xác thực schema JSON của labs/lab2_v2/experiment.ipynb bằng nbformat."""
    nb_path = Path("labs/lab2_v2/experiment.ipynb")
    nb = nbformat.read(nb_path, as_version=4)
    nbformat.validate(nb)
    assert len(nb.cells) >= 7
