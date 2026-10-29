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
steps = [p for p in progress if p.get("total_steps")]
check("batch progress says how many batches an epoch has", bool(steps) and all(p["total_steps"] >= 1 for p in steps), str(progress[:3]))
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

# --- Trying the trained model ---
print("Trying the model")
# Record the validation split each dataset load makes, to check predictions use the one training held out
splits = []
_real_load = sc._load_dataset
def _recording_load(*args, **kwargs):
    train_loader, test_loader = _real_load(*args, **kwargs)
    splits.append(list(test_loader.dataset.indices) if test_loader is not None else None)
    return train_loader, test_loader
sc._load_dataset = _recording_load

sc._train_stop_event.clear()
sc._train_loop(graph(), config)
res = sc.model_predict({"graph": graph(), "config": config, "count": 5, "sample_seed": 1})
check("samples come from the validation split", res["split"] == "validation" and res["dataset_size"] == 16, str(res["split"]))
check("the validation split is the one training held out", len(splits) == 2 and splits[0] == splits[1], str(splits))
check("asked-for number of samples", len(res["samples"]) == 5, str(len(res["samples"])))
s0 = res["samples"][0]
check("classifier gives its top classes with probabilities",
      s0["output"]["kind"] == "classes" and len(s0["output"]["top"]) == 3
      and s0["output"]["top"][0]["prob"] >= s0["output"]["top"][1]["prob"], str(s0["output"]))
check("each sample says whether the model got it right",
      all(isinstance(x["correct"], bool) and x["target"]["kind"] == "class" for x in res["samples"]))
check("tabular inputs are shown as values", s0["input"]["kind"] == "values" and s0["input"]["total"] == 8, str(s0["input"]))
check("a model without images doesn't offer image files", res["accepts_images"] is False)
mistakes = sc.model_predict({"graph": graph(), "config": config, "count": 4, "mistakes_only": True})
check("mistakes only returns only wrong answers",
      all(x["correct"] is False for x in mistakes["samples"]) and mistakes["scanned"] >= len(mistakes["samples"]),
      str([x["correct"] for x in mistakes["samples"]]))
same = sc.model_predict({"graph": graph(), "config": config, "count": 5, "sample_seed": 1})
check("same seed, same samples", [x["index"] for x in same["samples"]] == [x["index"] for x in res["samples"]])

# Image inputs: shown as PNGs, and an image file can be tried
img_path = os.path.join(tmp, "images.pt")
torch.save({"x": torch.rand(40, 1, 8, 8), "y": torch.randint(0, 3, (40,))}, img_path)
img_graph = graph()
img_graph["nodes"][0]["params"] = {"shape": "[1, 1, 8, 8]"}
img_graph["nodes"].insert(1, {"id": "flat", "type": "call_module", "moduleType": "Flatten", "label": "flat", "params": {}})
img_graph["nodes"][2]["params"] = {"in_features": 64, "out_features": 3}
img_graph["edges"][0] = {"id": "e0", "source": "inp", "target": "flat", "sourceHandle": "out", "targetHandle": "in"}
img_graph["edges"].append({"id": "e3", "source": "flat", "target": "fc", "sourceHandle": "out", "targetHandle": "in"})
img_config = dict(config, epochs=1, dataset={"type": "TensorFile", "root": img_path, "features_key": "x", "labels_key": "y", "val_split": 0.25})
sc._train_stop_event.clear()
sc._train_loop(img_graph, img_config)
res = sc.model_predict({"graph": img_graph, "config": img_config, "count": 3})
s0 = res["samples"][0]
check("image inputs come back as PNGs", s0["input"]["kind"] == "image" and s0["input"]["png"] and s0["input"]["width"] == 8, str(s0["input"])[:120])
check("an image model offers image files", res["accepts_images"] is True)
from PIL import Image
png_path = os.path.join(tmp, "mine.png")
Image.new("RGB", (40, 30), (200, 30, 30)).save(png_path)
own = sc.model_predict({"graph": img_graph, "config": img_config, "image_path": png_path})
check("an image file is resized and predicted", len(own["samples"]) == 1 and own["samples"][0]["output"]["kind"] == "classes"
      and own["samples"][0]["target"] is None, str(own["samples"])[:200])
try:
    sc.model_predict({"graph": graph(), "config": config, "image_path": png_path, "weights": "checkpoint", "checkpoint_path": ckpts[-1]})
    check("an image file is refused for a model without images", False)
except ValueError as e:
    check("an image file is refused for a model without images", "doesn't take images" in str(e), str(e))

sc._last_trained_state = None
try:
    sc.model_predict({"graph": graph(), "config": config})
    check("trying with no trained weights is refused", False)
except ValueError as e:
    check("trying with no trained weights is refused", "No trained weights" in str(e), str(e))
sc._load_dataset = _real_load

print()
if failures:
    print(f"{len(failures)} failed: {', '.join(failures)}")
    sys.exit(1)
print("All sidecar tests passed")
