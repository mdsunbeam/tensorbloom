"""End-to-end tests for training, checkpoints, export weights, and custom blocks."""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(__file__))

import torch
import tensorbloom_sidecar as sc

tmp = tempfile.mkdtemp(prefix="tensorbloom-test-")
os.environ["TENSORBLOOM_DATA_DIR"] = tmp

notifications = []
sc.send_notification = lambda method, params=None: notifications.append((method, params))

failures = []

def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def graph(out_features=3, custom=False):
    hidden = {"id": "fc", "type": "call_module", "moduleType": "Linear", "label": "fc",
              "params": {"in_features": 8, "out_features": out_features}}
    if custom:
        hidden = {"id": "fc", "type": "call_module", "moduleType": "MyBlock", "label": "MyBlock",
                  "params": {"width": out_features},
                  "customCode": (
                      "import torch\nimport torch.nn as nn\n\n"
                      "class MyBlock(nn.Module):\n"
                      "    def __init__(self, width=3):\n"
                      "        super().__init__()\n"
                      "        self.lin = nn.Linear(8, width)\n"
                      "    def forward(self, x):\n"
                      "        return torch.relu(self.lin(x))\n")}
    return {
        "nodes": [
            {"id": "inp", "type": "placeholder", "moduleType": "Input", "label": "Input", "params": {"shape": "[1, 8]"}},
            hidden,
            {"id": "out", "type": "output", "moduleType": "Output", "label": "Output", "params": {}},
        ],
        "edges": [
            {"id": "e1", "source": "inp", "target": "fc", "sourceHandle": "out", "targetHandle": "in"},
            {"id": "e2", "source": "fc", "target": "out", "sourceHandle": "out", "targetHandle": "in"},
        ],
    }


# --- Custom blocks build from their source code ---
print("Custom blocks")
model = sc._build_model_from_graph(graph(custom=True))
y = model(torch.randn(4, 8))
check("custom block builds and runs", tuple(y.shape) == (4, 3), str(y.shape))
try:
    bad = graph()
    bad["nodes"][1]["moduleType"] = "NotARealLayer"
    sc._build_model_from_graph(bad)
    check("unknown layer gives a clear error", False)
except ValueError as e:
    check("unknown layer gives a clear error", "NotARealLayer" in str(e), str(e))

# --- OneCycleLR ---
print("Schedulers")
opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
sched = sc._build_scheduler({"type": "OneCycleLR"}, opt, epochs=2, steps_per_epoch=5)
check("OneCycleLR is built", isinstance(sched, torch.optim.lr_scheduler.OneCycleLR))
cos = sc._build_scheduler({"type": "CosineAnnealingLR"}, opt, epochs=7)
check("CosineAnnealingLR T_max defaults to epochs", cos.T_max == 7, str(cos.T_max))

# --- Export before any training ---
print("Export")
try:
    sc.export_torchscript({"graph": graph(), "input_shapes": {"x": [1, 8]}, "output_path": os.path.join(tmp, "m.pt")})
    check("export with no trained weights is refused", False)
except ValueError as e:
    check("export with no trained weights is refused", "No trained weights" in str(e), str(e))

# --- Train end to end on a tensor file ---
print("Training")
data_path = os.path.join(tmp, "data.pt")
torch.manual_seed(0)
torch.save({"x": torch.randn(64, 8), "y": torch.randint(0, 3, (64,))}, data_path)
config = {
    "optimizer": {"type": "Adam", "lr": 0.01},
    "scheduler": {"type": "OneCycleLR"},
    "loss": "CrossEntropyLoss",
    "epochs": 5,
    "batch_size": 16,
    "dataset": {"type": "TensorFile", "root": data_path, "features_key": "x", "labels_key": "y", "val_split": 0.25},
    "device": "cpu",
    "seed": 0,
}
sc._train_stop_event.clear()
sc._train_loop(graph(), config)

progress = [p for m, p in notifications if m == "train.progress"]
logs = [p["text"] for m, p in notifications if m == "train.log"]
errors = [p for p in progress if p.get("status") == "error"]
check("training completes without error", not errors and any(p.get("status") == "complete" for p in progress),
      str(errors) + "\n" + "\n".join(logs[-10:]))
ckpts = [p["path"] for p in progress if p.get("status") == "checkpoint"]
check("checkpoint written under the data dir", bool(ckpts) and ckpts[-1].startswith(tmp) and os.path.exists(ckpts[-1]), str(ckpts))
lrs = [p["lr"] for p in progress if "lr" in p and p.get("lr") is not None]
check("OneCycleLR changes the learning rate", len(set(round(lr, 8) for lr in lrs)) > 2, str(lrs))
check("trained weights are kept for export", sc._last_trained_state is not None)

# --- Export uses trained weights ---
reference = sc._build_model_from_graph(graph())
reference.load_state_dict(sc._last_trained_state)
reference.eval()
x = torch.randn(2, 8)

res = sc.export_torchscript({"graph": graph(), "input_shapes": {"x": [1, 8]}, "output_path": os.path.join(tmp, "trained.pt")})
loaded = torch.jit.load(res["path"])
check("TorchScript export uses trained weights", res["weights"] == "trained" and torch.allclose(loaded(x), reference(x)))

res = sc.export_torchscript({"graph": graph(), "input_shapes": {"x": [1, 8]}, "output_path": os.path.join(tmp, "ckpt.pt"),
                             "weights": "checkpoint", "checkpoint_path": ckpts[-1]})
loaded = torch.jit.load(res["path"])
check("export from checkpoint file", res["weights"] == "checkpoint" and torch.allclose(loaded(x), reference(x)))

res = sc.export_torchscript({"graph": graph(), "input_shapes": {"x": [1, 8]}, "output_path": os.path.join(tmp, "rand.pt"),
                             "weights": "random"})
check("untrained export is labelled", res["weights"] == "random")

try:
    sc.export_torchscript({"graph": graph(out_features=5), "input_shapes": {"x": [1, 8]}, "output_path": os.path.join(tmp, "m.pt")})
    check("changed architecture is reported", False)
except ValueError as e:
    check("changed architecture is reported", "don't match" in str(e), str(e))

try:
    import onnx  # noqa: F401
    res = sc.export_onnx({"graph": graph(), "input_shapes": {"x": [1, 8]}, "output_path": os.path.join(tmp, "trained.onnx")})
    check("ONNX export uses trained weights", res["weights"] == "trained" and os.path.exists(res["path"]))
except ImportError:
    print("  SKIP  ONNX export (onnx not installed)")

print()
if failures:
    print(f"{len(failures)} failed: {', '.join(failures)}")
    sys.exit(1)
print("All sidecar tests passed")
