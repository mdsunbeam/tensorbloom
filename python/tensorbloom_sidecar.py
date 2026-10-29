"""
TensorBloom Python Sidecar
JSON-RPC server over stdin/stdout for PyTorch operations.

Handles:
- compile: PyTorch code -> visual graph JSON
- decompile: visual graph JSON -> PyTorch code
- infer_shapes: propagate tensor shapes through graph
- train.start / train.stop: training lifecycle
- validate: check graph for errors
"""

import sys
import io
import json
import traceback
import importlib
import platform
from typing import Any

import torch
import torch.nn as nn
import torch.fx

# Preserve real stdout before any redirection — all JSON-RPC goes through this
_real_stdout = sys.stdout


# ---------------------------------------------------------------------------
# JSON-RPC transport
# ---------------------------------------------------------------------------

def send_response(id: int | None, result: Any = None, error: Any = None):
    msg: dict[str, Any] = {"jsonrpc": "2.0"}
    if error is not None:
        msg["error"] = error
    else:
        msg["result"] = result
    if id is not None:
        msg["id"] = id
    _real_stdout.write(json.dumps(msg) + "\n")
    _real_stdout.flush()


def send_notification(method: str, params: Any = None):
    msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    _real_stdout.write(json.dumps(msg) + "\n")
    _real_stdout.flush()


class _LogCapture(io.TextIOBase):
    """Captures write() calls and routes them as train.log JSON-RPC notifications."""
    def __init__(self, stream_name: str):
        self._stream = stream_name
        self._buf = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                send_notification("train.log", {"text": line, "stream": self._stream})
        return len(s)

    def flush(self):
        if self._buf.strip():
            send_notification("train.log", {"text": self._buf, "stream": self._stream})
            self._buf = ""

    def isatty(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Module builder: JSON node config -> nn.Module instance
# ---------------------------------------------------------------------------

# Modules that take non-trivial constructor args need special handling
class _Permute(nn.Module):
    """nn.Module wrapper for tensor.permute() — not built into PyTorch."""
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        return x.permute(*self.dims)

class _LSTMExtract(nn.Module):
    """Extract just the output tensor from LSTM (discards hidden states)."""
    def __init__(self, lstm):
        super().__init__()
        self.lstm = lstm
    def forward(self, x):
        out, _ = self.lstm(x)
        return out

class _TransformerDecoderSelfAttn(nn.Module):
    """Wraps TransformerDecoderLayer for self-attention (uses input as both tgt and memory)."""
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    def forward(self, x):
        return self.layer(x, x)

SPECIAL_CONSTRUCTORS = {
    "AdaptiveAvgPool2d": lambda p: nn.AdaptiveAvgPool2d(
        output_size=tuple(p["output_size"]) if isinstance(p.get("output_size"), list) else p.get("output_size", (1, 1))
    ),
    "AdaptiveAvgPool1d": lambda p: nn.AdaptiveAvgPool1d(
        output_size=p.get("output_size", 1)
    ),
    "LayerNorm": lambda p: nn.LayerNorm(
        normalized_shape=json.loads(p["normalized_shape"]) if isinstance(p.get("normalized_shape"), str) else p.get("normalized_shape", [512]),
        eps=p.get("eps", 1e-5),
        elementwise_affine=p.get("elementwise_affine", True),
    ),
    "RMSNorm": lambda p: nn.RMSNorm(
        normalized_shape=json.loads(p["normalized_shape"]) if isinstance(p.get("normalized_shape"), str) else p.get("normalized_shape", [512]),
        eps=p.get("eps", 1e-5),
    ),
    "Permute": lambda p: _Permute(
        dims=json.loads(p["dims"]) if isinstance(p.get("dims"), str) else p.get("dims", [0, 1, 2])
    ),
    "Unflatten": lambda p: nn.Unflatten(
        dim=p.get("dim", 1),
        unflattened_size=tuple(json.loads(p["unflattened_size"])) if isinstance(p.get("unflattened_size"), str) else tuple(p.get("unflattened_size", [1]))
    ),
    "LSTM": lambda p: _LSTMExtract(nn.LSTM(**{k: v for k, v in p.items() if v is not None})),
    "GRU": lambda p: _LSTMExtract(nn.GRU(**{k: v for k, v in p.items() if v is not None})),
    "TransformerDecoderLayer": lambda p: _TransformerDecoderSelfAttn(
        nn.TransformerDecoderLayer(**{k: v for k, v in p.items() if v is not None})
    ),
}


def build_module(module_type: str, params: dict) -> nn.Module:
    """Build an nn.Module from type name and parameter dict."""
    if module_type in SPECIAL_CONSTRUCTORS:
        return SPECIAL_CONSTRUCTORS[module_type](params)

    cls = getattr(nn, module_type, None)
    if cls is None:
        raise ValueError(f"Unknown module type: {module_type}")

    # Filter out None values (optional params) and build
    clean = {k: v for k, v in params.items() if v is not None}
    return cls(**clean)


# ---------------------------------------------------------------------------
# Compiler: PyTorch code -> graph JSON
# ---------------------------------------------------------------------------

def compile_code(code: str) -> dict:
    """Execute PyTorch code, find the model, trace it, return graph JSON."""
    namespace: dict[str, Any] = {
        "torch": torch,
        "nn": nn,
        "F": torch.nn.functional,
    }
    exec(code, namespace)

    # Find the model: look for nn.Module subclass instances or the last class defined
    model = None
    for name, obj in namespace.items():
        if name.startswith("_"):
            continue
        if isinstance(obj, nn.Module):
            model = obj
            break

    if model is None:
        # Try to find a class and instantiate it
        for name, obj in namespace.items():
            if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module:
                try:
                    model = obj()
                except Exception:
                    pass
                break

    if model is None:
        raise ValueError("No nn.Module found in code. Define a model instance or class.")

    # Trace
    traced = torch.fx.symbolic_trace(model)

    nodes = []
    edges = []
    y = 0

    for node in traced.graph.nodes:
        visual: dict[str, Any] = {
            "id": node.name,
            "type": node.op,
            "label": node.name,
            "position": {"x": 200, "y": y},
            "params": {},
        }

        if node.op == "call_module":
            submod = traced.get_submodule(node.target)
            visual["moduleType"] = type(submod).__name__
            visual["params"] = extract_module_params(submod)

        elif node.op == "call_function":
            visual["target"] = _qualified_name(node.target)

        elif node.op == "call_method":
            visual["target"] = node.target

        elif node.op == "placeholder":
            visual["moduleType"] = "Input"

        elif node.op == "output":
            visual["moduleType"] = "Output"

        nodes.append(visual)
        y += 150

        for arg_node in node.all_input_nodes:
            edges.append({
                "id": f"e_{arg_node.name}_{node.name}",
                "source": arg_node.name,
                "target": node.name,
                "sourceHandle": "out",
                "targetHandle": "in",
            })

    return {"nodes": nodes, "edges": edges}


def extract_module_params(module: nn.Module) -> dict:
    """Extract constructor-relevant parameters from a module instance."""
    params: dict[str, Any] = {}
    cls_name = type(module).__name__

    if isinstance(module, nn.Linear):
        params = {"in_features": module.in_features, "out_features": module.out_features, "bias": module.bias is not None}
    elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        params = {
            "in_channels": module.in_channels, "out_channels": module.out_channels,
            "kernel_size": module.kernel_size[0] if isinstance(module.kernel_size, tuple) and len(set(module.kernel_size)) == 1 else module.kernel_size,
            "stride": module.stride[0] if isinstance(module.stride, tuple) and len(set(module.stride)) == 1 else module.stride,
            "padding": module.padding[0] if isinstance(module.padding, tuple) and len(set(module.padding)) == 1 else module.padding,
            "dilation": module.dilation[0] if isinstance(module.dilation, tuple) and len(set(module.dilation)) == 1 else module.dilation,
            "groups": module.groups, "bias": module.bias is not None,
        }
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        params = {"num_features": module.num_features, "eps": module.eps, "momentum": module.momentum, "affine": module.affine}
    elif isinstance(module, nn.LayerNorm):
        params = {"normalized_shape": list(module.normalized_shape), "eps": module.eps, "elementwise_affine": module.elementwise_affine}
    elif isinstance(module, nn.Dropout):
        params = {"p": module.p, "inplace": module.inplace}
    elif isinstance(module, nn.Embedding):
        params = {"num_embeddings": module.num_embeddings, "embedding_dim": module.embedding_dim}
    elif isinstance(module, (nn.LSTM, nn.GRU)):
        params = {
            "input_size": module.input_size, "hidden_size": module.hidden_size,
            "num_layers": module.num_layers, "bias": module.bias,
            "batch_first": module.batch_first, "dropout": module.dropout,
            "bidirectional": module.bidirectional,
        }
    elif isinstance(module, nn.MultiheadAttention):
        params = {"embed_dim": module.embed_dim, "num_heads": module.num_heads, "dropout": module.dropout, "bias": module.in_proj_bias is not None}
    elif isinstance(module, nn.ReLU):
        params = {"inplace": module.inplace}
    elif isinstance(module, nn.LeakyReLU):
        params = {"negative_slope": module.negative_slope, "inplace": module.inplace}
    elif isinstance(module, nn.MaxPool2d):
        params = {"kernel_size": module.kernel_size, "stride": module.stride, "padding": module.padding}
    elif isinstance(module, nn.AdaptiveAvgPool2d):
        params = {"output_size": list(module.output_size) if isinstance(module.output_size, tuple) else module.output_size}
    elif isinstance(module, nn.Flatten):
        params = {"start_dim": module.start_dim, "end_dim": module.end_dim}

    return params


def _qualified_name(fn) -> str:
    """Get qualified name of a function."""
    if hasattr(fn, "__module__") and hasattr(fn, "__qualname__"):
        return f"{fn.__module__}.{fn.__qualname__}"
    return str(fn)


# ---------------------------------------------------------------------------
# Decompiler: graph JSON -> PyTorch code
# ---------------------------------------------------------------------------

def decompile_graph(graph_json: dict) -> str:
    """Convert visual graph JSON to clean PyTorch code."""
    nodes = graph_json["nodes"]
    edges = graph_json["edges"]

    # Build handle-aware adjacency
    adj: dict[str, list[str]] = {}
    adj_detailed: dict[str, list[dict]] = {}
    for e in edges:
        adj.setdefault(e["target"], []).append(e["source"])
        adj_detailed.setdefault(e["target"], []).append({
            "source": e["source"],
            "sourceHandle": e.get("sourceHandle", "out"),
            "targetHandle": e.get("targetHandle", "in"),
        })

    # Topological sort via the node order (assumed topological from editor)
    module_nodes = [n for n in nodes if n.get("moduleType") and n["type"] not in ("placeholder", "output")]
    input_nodes = [n for n in nodes if n["type"] == "placeholder" or n.get("moduleType") == "Input"]
    output_nodes = [n for n in nodes if n["type"] == "output" or n.get("moduleType") == "Output"]

    # Build unique names from labels (fall back to node ID)
    import re
    used_names: set[str] = set()
    name_map: dict[str, str] = {}
    def _sanitize(s: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]", "_", s)
    for n in module_nodes:
        base = _sanitize(n.get("label") or n["id"])
        name = base
        i = 2
        while name in used_names:
            name = f"{base}_{i}"; i += 1
        used_names.add(name)
        name_map[n["id"]] = name

    # Identify code block nodes and emit helper classes
    helper_classes: list[str] = []
    code_block_ids: set[str] = set()
    for n in module_nodes:
        cb = n.get("codeBlock")
        if cb:
            code_block_ids.add(n["id"])
            name = name_map[n["id"]]
            class_name = f"_{name}_Block"
            input_handles = cb.get("inputHandles", [{"id": "x"}])
            sig = ", ".join(h["id"] for h in input_handles)
            init_body = cb.get("initBody", "pass")
            forward_body = cb.get("forwardBody", "return x")
            init_lines_cb = "\n".join(f"        {line}" for line in init_body.strip().split("\n"))
            fwd_lines_cb = "\n".join(f"        {line}" for line in forward_body.strip().split("\n"))
            helper_classes.extend([
                f"class {class_name}(nn.Module):",
                f"    def __init__(self):",
                f"        super().__init__()",
                init_lines_cb,
                f"",
                f"    def forward(self, {sig}):",
                fwd_lines_cb,
                f"",
            ])

    # Build __init__ lines
    init_lines = []
    for n in module_nodes:
        name = name_map[n["id"]]
        if n["id"] in code_block_ids:
            class_name = f"_{name}_Block"
            init_lines.append(f"        self.{name} = {class_name}()")
        elif n["moduleType"] in ("Add", "Concatenate", "MatMul"):
            continue
        else:
            mt = n["moduleType"]
            params = n.get("params", {})
            param_str = format_constructor_args(mt, params)
            init_lines.append(f"        self.{name} = nn.{mt}({param_str})")

    # Build forward lines with handle-aware variable resolution
    forward_lines = []
    node_var: dict[str, str] = {}
    # Multi-output: node_id -> {handle_id -> var_name}
    output_var_map: dict[str, dict[str, str]] = {}

    # Assign unique variable names to each input
    if len(input_nodes) == 1:
        input_var_names = ["x"]
    else:
        input_var_names = [_sanitize(n.get("label") or f"x{i+1}") or f"x{i+1}" for i, n in enumerate(input_nodes)]
    for n, var in zip(input_nodes, input_var_names):
        node_var[n["id"]] = var

    def _resolve_var(src_id: str, src_handle: str) -> str:
        """Resolve upstream variable, considering multi-output handles."""
        hmap = output_var_map.get(src_id)
        if hmap and src_handle in hmap:
            return hmap[src_handle]
        return node_var.get(src_id, "x")

    for n in module_nodes:
        var_name = name_map[n["id"]]
        node_var[n["id"]] = var_name
        detail = adj_detailed.get(n["id"], [])
        mt = n["moduleType"]

        if n["id"] in code_block_ids:
            cb = n["codeBlock"]
            input_handles = cb.get("inputHandles", [])
            output_handles = cb.get("outputHandles", [{"id": "out_"}])

            # Map each input handle to the upstream variable
            arg_vars = []
            for h in input_handles:
                edge = next((d for d in detail if d["targetHandle"] == h["id"]), None)
                if edge:
                    arg_vars.append(_resolve_var(edge["source"], edge["sourceHandle"]))
                else:
                    arg_vars.append(h["id"])

            if len(output_handles) > 1:
                out_vars = [f"{var_name}_{h['id']}" for h in output_handles]
                forward_lines.append(f"        {', '.join(out_vars)} = self.{var_name}({', '.join(arg_vars)})")
                hmap = {}
                for h, v in zip(output_handles, out_vars):
                    hmap[h["id"]] = v
                output_var_map[n["id"]] = hmap
            else:
                forward_lines.append(f"        {var_name} = self.{var_name}({', '.join(arg_vars)})")
        elif mt in ("Add",):
            input_vars = [_resolve_var(d["source"], d["sourceHandle"]) for d in detail]
            if len(input_vars) >= 2:
                forward_lines.append(f"        {var_name} = {input_vars[0]} + {input_vars[1]}")
            else:
                forward_lines.append(f"        {var_name} = {', '.join(input_vars) if input_vars else 'x'}")
        elif mt in ("Concatenate",):
            input_vars = [_resolve_var(d["source"], d["sourceHandle"]) for d in detail]
            dim = n.get("params", {}).get("dim", 1)
            forward_lines.append(f"        {var_name} = torch.cat([{', '.join(input_vars)}], dim={dim})")
        elif mt in ("MatMul",):
            input_vars = [_resolve_var(d["source"], d["sourceHandle"]) for d in detail]
            if len(input_vars) >= 2:
                forward_lines.append(f"        {var_name} = torch.matmul({input_vars[0]}, {input_vars[1]})")
            else:
                forward_lines.append(f"        {var_name} = {', '.join(input_vars) if input_vars else 'x'}")
        else:
            input_vars = [_resolve_var(d["source"], d["sourceHandle"]) for d in detail]
            input_str = ", ".join(input_vars) if input_vars else "x"
            forward_lines.append(f"        {var_name} = self.{var_name}({input_str})")

    # Return statement
    last_var = "x"
    for n in output_nodes:
        detail = adj_detailed.get(n["id"], [])
        if detail:
            last_var = _resolve_var(detail[0]["source"], detail[0]["sourceHandle"])

    code_lines = [
        "import torch",
        "import torch.nn as nn",
        "",
    ]

    if helper_classes:
        code_lines.append("")
        code_lines.extend(helper_classes)

    code_lines.extend([
        "",
        "class Model(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
    ])
    code_lines.extend(init_lines)
    code_lines.append("")
    forward_params = ", ".join(input_var_names) if input_var_names else "x"
    code_lines.append(f"    def forward(self, {forward_params}):")
    code_lines.extend(forward_lines)
    code_lines.append(f"        return {last_var}")
    code_lines.append("")

    return "\n".join(code_lines)


def format_constructor_args(module_type: str, params: dict) -> str:
    """Format params dict as constructor argument string."""
    if not params:
        return ""

    # Get default values from registry knowledge to skip defaults
    parts = []
    for k, v in params.items():
        if v is None:
            continue
        if isinstance(v, bool):
            parts.append(f"{k}={v}")
        elif isinstance(v, str):
            # Could be a JSON list like "[512]"
            try:
                parsed = json.loads(v)
                parts.append(f"{k}={parsed}")
            except (json.JSONDecodeError, TypeError):
                parts.append(f"{k}='{v}'")
        elif isinstance(v, list):
            parts.append(f"{k}={tuple(v) if len(v) > 1 else v[0]}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Shape inference
# ---------------------------------------------------------------------------

def infer_shapes(graph_json: dict, input_shapes: dict[str, list[int]]) -> dict:
    """Build model from graph, run shape propagation, return shapes per node."""
    nodes = graph_json["nodes"]
    edges = graph_json["edges"]

    # Build the module
    root = nn.Module()
    for n in nodes:
        mt = n.get("moduleType")
        if mt and mt not in ("Input", "Output", "Add", "Concatenate", "MatMul"):
            try:
                submod = build_module(mt, n.get("params", {}))
                root.add_module(n["id"], submod)
            except Exception:
                pass

    # Build fx graph
    graph = torch.fx.Graph()
    node_map: dict[str, torch.fx.Node] = {}
    adj: dict[str, list[str]] = {}
    for e in edges:
        adj.setdefault(e["target"], []).append(e["source"])

    for n in nodes:
        nid = n["id"]
        inputs = [node_map[s] for s in adj.get(nid, []) if s in node_map]

        if n["type"] == "placeholder" or n.get("moduleType") == "Input":
            fx_node = graph.placeholder(nid)
        elif n["type"] == "output" or n.get("moduleType") == "Output":
            fx_node = graph.output(inputs[0] if inputs else None)
        elif n.get("moduleType") in ("Add",):
            if len(inputs) >= 2:
                fx_node = graph.call_function(torch.add, args=(inputs[0], inputs[1]))
            else:
                fx_node = graph.call_function(torch.add, args=tuple(inputs))
        elif n.get("moduleType") in ("Concatenate",):
            dim = n.get("params", {}).get("dim", 1)
            fx_node = graph.call_function(torch.cat, args=(inputs,), kwargs={"dim": dim})
        else:
            fx_node = graph.call_module(nid, args=tuple(inputs) if inputs else (list(node_map.values())[-1] if node_map else (),))

        node_map[nid] = fx_node

    try:
        gm = torch.fx.GraphModule(root, graph)
    except Exception as e:
        return {"error": str(e), "shapes": {}}

    # Create fake inputs
    fake_inputs = {}
    for name, shape in input_shapes.items():
        fake_inputs[name] = torch.randn(*shape)

    # Run shape propagation
    try:
        from torch.fx.passes.shape_prop import ShapeProp
        ShapeProp(gm).propagate(*fake_inputs.values())
    except Exception as e:
        return {"error": str(e), "shapes": {}}

    shapes: dict[str, Any] = {}
    for node in gm.graph.nodes:
        meta = node.meta.get("tensor_meta")
        if meta is not None:
            shapes[node.name] = {
                "shape": list(meta.shape),
                "dtype": str(meta.dtype),
            }

    return {"shapes": shapes}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_graph(graph_json: dict) -> dict:
    """Check graph for common errors."""
    errors = []
    warnings = []
    nodes = graph_json.get("nodes", [])
    edges = graph_json.get("edges", [])

    node_ids = {n["id"] for n in nodes}
    has_input = any(n.get("moduleType") == "Input" or n.get("type") == "placeholder" for n in nodes)
    has_output = any(n.get("moduleType") == "Output" or n.get("type") == "output" for n in nodes)

    if not has_input:
        errors.append("Graph has no Input node")
    if not has_output:
        errors.append("Graph has no Output node")

    # Check edges reference valid nodes
    targets_with_inputs: set[str] = set()
    for e in edges:
        if e["source"] not in node_ids:
            errors.append(f"Edge references unknown source: {e['source']}")
        if e["target"] not in node_ids:
            errors.append(f"Edge references unknown target: {e['target']}")
        targets_with_inputs.add(e["target"])

    # Check for disconnected nodes (except Input)
    for n in nodes:
        if n.get("moduleType") == "Input" or n.get("type") == "placeholder":
            continue
        if n["id"] not in targets_with_inputs:
            warnings.append(f"Node '{n.get('label', n['id'])}' has no incoming connections")

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

import threading

_train_stop_event = threading.Event()
_train_thread: threading.Thread | None = None
# CPU copy of the weights from the most recent training run, used by export
_last_trained_state: dict[str, torch.Tensor] | None = None


def _data_dir() -> str:
    """Writable per-user directory for checkpoints and downloads (set by the Tauri host)."""
    import os
    return os.environ.get("TENSORBLOOM_DATA_DIR") or os.path.join(os.path.expanduser("~"), ".tensorbloom")


def _build_model_from_graph(graph_json: dict) -> nn.Module:
    """Build a runnable nn.Module from graph JSON."""
    import re
    import operator
    nodes = graph_json["nodes"]
    edges = graph_json["edges"]

    # Build handle-aware adjacency: target_id -> [{source, sourceHandle, targetHandle}]
    adj: dict[str, list[str]] = {}
    adj_detailed: dict[str, list[dict]] = {}
    for e in edges:
        adj.setdefault(e["target"], []).append(e["source"])
        adj_detailed.setdefault(e["target"], []).append({
            "source": e["source"],
            "sourceHandle": e.get("sourceHandle", "out"),
            "targetHandle": e.get("targetHandle", "in"),
        })

    # Build unique names from labels for readable module names
    used_names: set[str] = set()
    id_to_name: dict[str, str] = {}
    LOSS_TYPES = {
        "CrossEntropyLoss", "MSELoss", "BCEWithLogitsLoss", "BCELoss",
        "L1Loss", "HuberLoss", "NLLLoss", "KLDivLoss",
        "TripletMarginLoss", "CosineEmbeddingLoss", "CTCLoss",
        "SmoothL1Loss",
    }
    functional_ops = {"Add", "Concatenate", "MatMul", "Input", "Output"} | LOSS_TYPES
    for n in nodes:
        mt = n.get("moduleType", "")
        if mt and mt not in functional_ops:
            base = re.sub(r"[^a-zA-Z0-9_]", "_", n.get("label") or n["id"])
            name = base
            i = 2
            while name in used_names:
                name = f"{base}_{i}"; i += 1
            used_names.add(name)
            id_to_name[n["id"]] = name

    # Track which nodes are code blocks
    code_block_nodes: set[str] = set()

    # Build module with sub-modules
    root = nn.Module()
    for n in nodes:
        mt = n.get("moduleType", "")
        if mt and mt not in functional_ops:
            nid = n["id"]
            # Check if node has a code block definition
            label = n.get("label") or mt
            try:
                if "codeBlock" in n and n["codeBlock"]:
                    code_block_nodes.add(nid)
                    submod = _build_code_block_module(n["codeBlock"])
                elif n.get("customCode"):
                    submod = _build_custom_block_module(n["customCode"], mt, n.get("params", {}))
                else:
                    submod = build_module(mt, n.get("params", {}))
            except Exception as e:
                raise ValueError(f"Could not build layer '{label}' ({mt}): {e}") from e
            root.add_module(id_to_name[nid], submod)

    # Build fx graph
    graph = torch.fx.Graph()
    node_map: dict[str, torch.fx.Node] = {}
    # For multi-output code blocks: node_id -> {handle_id -> fx_node}
    output_handle_map: dict[str, dict[str, torch.fx.Node]] = {}

    # Topological sort to ensure nodes are processed in dependency order
    in_degree: dict[str, int] = {n["id"]: 0 for n in nodes}
    successors: dict[str, list[str]] = {n["id"]: [] for n in nodes}
    for e in edges:
        if e["target"] in in_degree and e["source"] in in_degree:
            in_degree[e["target"]] = in_degree.get(e["target"], 0) + 1
            successors.setdefault(e["source"], []).append(e["target"])
    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    sorted_ids: list[str] = []
    while queue:
        nid = queue.pop(0)
        sorted_ids.append(nid)
        for s in successors.get(nid, []):
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    # Add any remaining nodes (in case of cycles or disconnected)
    remaining = [n["id"] for n in nodes if n["id"] not in set(sorted_ids)]
    sorted_ids.extend(remaining)
    node_by_id = {n["id"]: n for n in nodes}
    sorted_nodes = [node_by_id[nid] for nid in sorted_ids if nid in node_by_id]

    for n in sorted_nodes:
        nid = n["id"]
        mt = n.get("moduleType", "")

        if mt == "Input" or n.get("type") == "placeholder":
            fx_node = graph.placeholder(nid)
            node_map[nid] = fx_node
            continue

        if mt == "Output" or n.get("type") == "output" or mt in LOSS_TYPES:
            # Resolve input from handle map if available
            detail = adj_detailed.get(nid, [])
            # For Loss nodes, use the "input" handle (prediction), not "target" (labels)
            if mt in LOSS_TYPES:
                pred_edge = next((d for d in detail if d["targetHandle"] == "input"), None)
                if not pred_edge and detail:
                    pred_edge = detail[0]  # fallback to first edge
            else:
                pred_edge = detail[0] if detail else None

            inp = None
            if pred_edge:
                src_handle_map = output_handle_map.get(pred_edge["source"])
                if src_handle_map and pred_edge["sourceHandle"] in src_handle_map:
                    inp = src_handle_map[pred_edge["sourceHandle"]]
                elif pred_edge["source"] in node_map:
                    inp = node_map[pred_edge["source"]]
            fx_node = graph.output(inp)
            node_map[nid] = fx_node
            continue

        # For code block nodes, map inputs by target handle
        if nid in code_block_nodes:
            cb = n["codeBlock"]
            input_handles = cb.get("inputHandles", [])
            output_handles = cb.get("outputHandles", [])
            detail = adj_detailed.get(nid, [])

            # Build ordered args matching input handle order
            args = []
            for h in input_handles:
                # Find the edge targeting this handle
                edge = next((d for d in detail if d["targetHandle"] == h["id"]), None)
                if edge:
                    src_id = edge["source"]
                    src_handle = edge["sourceHandle"]
                    src_handle_map = output_handle_map.get(src_id)
                    if src_handle_map and src_handle in src_handle_map:
                        args.append(src_handle_map[src_handle])
                    elif src_id in node_map:
                        args.append(node_map[src_id])

            fx_node = graph.call_module(id_to_name[nid], args=tuple(args) if args else ())
            node_map[nid] = fx_node

            # For multi-output code blocks, use getitem to extract each output
            if len(output_handles) > 1:
                handle_map: dict[str, torch.fx.Node] = {}
                for i, h in enumerate(output_handles):
                    getitem_node = graph.call_function(operator.getitem, args=(fx_node, i))
                    handle_map[h["id"]] = getitem_node
                output_handle_map[nid] = handle_map
            continue

        # Standard nodes
        inputs = []
        detail = adj_detailed.get(nid, [])
        for d in detail:
            src_id = d["source"]
            src_handle = d["sourceHandle"]
            src_handle_map = output_handle_map.get(src_id)
            if src_handle_map and src_handle in src_handle_map:
                inputs.append(src_handle_map[src_handle])
            elif src_id in node_map:
                inputs.append(node_map[src_id])

        if mt == "Add":
            fx_node = graph.call_function(torch.add, args=(inputs[0], inputs[1]) if len(inputs) >= 2 else tuple(inputs))
        elif mt == "Concatenate":
            dim = n.get("params", {}).get("dim", 1)
            fx_node = graph.call_function(torch.cat, args=(inputs,), kwargs={"dim": dim})
        elif mt == "MatMul":
            fx_node = graph.call_function(torch.matmul, args=(inputs[0], inputs[1]) if len(inputs) >= 2 else tuple(inputs))
        else:
            fx_node = graph.call_module(id_to_name[nid], args=tuple(inputs) if inputs else ())

        node_map[nid] = fx_node

    return torch.fx.GraphModule(root, graph)


def _get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _build_optimizer(config: dict, params) -> torch.optim.Optimizer:
    opt_type = config.get("type", "Adam")
    lr = config.get("lr", 0.001)
    wd = config.get("weight_decay", 0)
    mom = config.get("momentum", 0.9)

    if opt_type == "SGD":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=mom)
    elif opt_type == "AdamW":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif opt_type == "RMSprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=wd)
    else:
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)


def _build_scheduler(config: dict | None, optimizer, epochs: int = 10, steps_per_epoch: int = 1):
    if not config or config.get("type") == "None":
        return None
    stype = config["type"]
    if stype == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.get("step_size", 5), gamma=config.get("gamma", 0.1))
    elif stype == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.get("T_max", epochs))
    elif stype == "OneCycleLR":
        # Stepped once per batch; the configured optimizer LR is the peak LR
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.get("max_lr", optimizer.param_groups[0]["lr"]),
            epochs=epochs,
            steps_per_epoch=max(steps_per_epoch, 1),
            pct_start=config.get("pct_start", 0.3),
        )
    elif stype == "ExponentialLR":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.get("gamma", 0.95))
    elif stype == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config.get("patience", 5))
    return None


def _build_loss(loss_name: str, loss_params: dict | None = None) -> nn.Module:
    cls = getattr(nn, loss_name, nn.CrossEntropyLoss)
    if loss_params:
        # Filter out None/empty params
        clean = {k: v for k, v in loss_params.items() if v is not None and k not in ("_probed", "_introspected")}
        try:
            return cls(**clean)
        except Exception:
            return cls()
    return cls()


def _extract_loss_from_graph(graph_json: dict) -> tuple[str, dict, int]:
    """Extract loss node info from graph. Returns (loss_type, loss_params, target_batch_index).
    target_batch_index = which position in the data batch is the loss target."""
    nodes = graph_json.get("nodes", [])
    edges = graph_json.get("edges", [])
    LOSS_TYPES = {
        "CrossEntropyLoss", "MSELoss", "BCEWithLogitsLoss", "BCELoss",
        "L1Loss", "HuberLoss", "NLLLoss", "KLDivLoss",
        "TripletMarginLoss", "CosineEmbeddingLoss", "CTCLoss",
        "SmoothL1Loss",
    }

    loss_node = None
    for n in nodes:
        mt = n.get("moduleType", "")
        if mt in LOSS_TYPES:
            loss_node = n
            break

    if not loss_node:
        return "", {}, 1  # No loss node in graph

    loss_type = loss_node["moduleType"]
    loss_params = loss_node.get("params", {})

    # Determine which batch index feeds the target handle
    # Look at edges targeting the loss node's "target" handle
    # and trace back to a Data node output to find the batch position
    # If no y→target edge exists, this is unsupervised (reconstruction) → use input as target
    target_batch_idx = -1  # -1 = no target edge found (reconstruction / unsupervised)
    for e in edges:
        if e.get("target") == loss_node["id"] and e.get("targetHandle") == "target":
            target_batch_idx = 1  # default: second item in batch (labels)
            # The source handle on the Data node determines batch position
            src_handle = e.get("sourceHandle", "y")
            # Find the Data node to get its output ordering
            src_node = next((n for n in nodes if n["id"] == e["source"]), None)
            if src_node and src_node.get("dataOutputs"):
                for idx, out in enumerate(src_node["dataOutputs"]):
                    if out["id"] == src_handle:
                        target_batch_idx = idx
                        break
            elif src_handle == "x":
                # Data node filtered from graph — infer from handle name
                # "x" = batch[0] = input data (reconstruction / unsupervised)
                target_batch_idx = 0
            break

    return loss_type, loss_params, target_batch_idx


def _build_augmentation(aug_config: dict | None, is_grayscale: bool = False) -> list:
    """Build augmentation transforms from config."""
    import torchvision.transforms as T
    transforms = []
    if not aug_config:
        return transforms
    if aug_config.get("horizontalFlip"):
        transforms.append(T.RandomHorizontalFlip())
    if aug_config.get("verticalFlip"):
        transforms.append(T.RandomVerticalFlip())
    if aug_config.get("randomCrop"):
        # Will be applied before this in the actual pipeline
        pass  # handled per-dataset
    rotation = aug_config.get("rotation", 0)
    if rotation > 0:
        transforms.append(T.RandomRotation(rotation))
    if aug_config.get("colorJitter") and not is_grayscale:
        transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
    return transforms


def _load_dataset(config: dict, batch_size: int, aug_config: dict | None = None, dataloader_seed: int | None = None, is_reconstruction: bool = False):
    import torchvision
    import torchvision.transforms as T

    ds_type = config.get("type", "CIFAR10")
    # Support universal Data node "source" field as alias for "type"
    if "source" in config and not config.get("type"):
        ds_type = config["source"]
        # Map source-specific root fields
        if ds_type == "HuggingFace":
            config.setdefault("root", config.get("dataset_name", "mnist"))
        elif ds_type in ("ImageFolder", "CSV", "TensorFile"):
            config.setdefault("root", config.get("path", "./data"))
        elif ds_type == "CSV":
            ds_type = "Custom CSV"
        # Presets use their name directly (CIFAR10, MNIST, etc.)
    if ds_type == "CSV":
        ds_type = "Custom CSV"
    root = config.get("root", "./data")
    val_split = config.get("val_split", 0.2)

    if ds_type in ("CIFAR10", "CIFAR100"):
        aug_transforms = _build_augmentation(aug_config, is_grayscale=False)
        crop = [T.RandomCrop(32, padding=4)] if aug_config and aug_config.get("randomCrop") else []
        # Skip normalization for reconstruction — Sigmoid output is [0,1], ToTensor gives [0,1]
        norm = [] if is_reconstruction else [T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        train_transform = T.Compose(crop + aug_transforms + [T.ToTensor()] + norm)
        test_transform = T.Compose([T.ToTensor()] + norm)
        ds_cls = getattr(torchvision.datasets, ds_type)
        train_ds = ds_cls(root=root, train=True, download=True, transform=train_transform)
        test_ds = ds_cls(root=root, train=False, download=True, transform=test_transform)
    elif ds_type in ("MNIST", "FashionMNIST"):
        aug_transforms = _build_augmentation(aug_config, is_grayscale=True)
        crop = [T.RandomCrop(28, padding=2)] if aug_config and aug_config.get("randomCrop") else []
        norm = [] if is_reconstruction else [T.Normalize((0.5,), (0.5,))]
        train_transform = T.Compose(crop + aug_transforms + [T.ToTensor()] + norm)
        test_transform = T.Compose([T.ToTensor()] + norm)
        ds_cls = getattr(torchvision.datasets, ds_type)
        train_ds = ds_cls(root=root, train=True, download=True, transform=train_transform)
        test_ds = ds_cls(root=root, train=False, download=True, transform=test_transform)
    elif ds_type == "ImageFolder":
        import os
        image_size = config.get("image_size", 224)
        channels = config.get("channels", "rgb")
        if channels == "grayscale":
            aug_transforms_if = _build_augmentation(aug_config, is_grayscale=True)
            normalize = [] if is_reconstruction else [T.Normalize((0.5,), (0.5,))]
            convert = T.Grayscale(num_output_channels=1)
        else:
            aug_transforms_if = _build_augmentation(aug_config, is_grayscale=False)
            normalize = [] if is_reconstruction else [T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
            convert = None
        crop = [T.RandomCrop(image_size, padding=image_size // 8)] if aug_config and aug_config.get("randomCrop") else []
        base = [T.Resize(image_size + 32), T.CenterCrop(image_size)]
        to_tensor = [T.ToTensor()] + normalize
        if convert:
            base = [convert] + base
        train_transform = T.Compose(base + crop + aug_transforms_if + to_tensor)
        test_transform = T.Compose(base + to_tensor)

        train_dir = os.path.join(root, "train")
        val_dir = os.path.join(root, "val")
        if os.path.isdir(train_dir):
            train_ds = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
            if os.path.isdir(val_dir):
                test_ds = torchvision.datasets.ImageFolder(val_dir, transform=test_transform)
            elif val_split > 0:
                # No val directory — split train data
                n = len(train_ds)
                n_val = int(n * val_split)
                train_ds, test_ds = torch.utils.data.random_split(train_ds, [n - n_val, n_val])
            else:
                test_ds = None
        else:
            # root is the dataset folder itself (no train/ subfolder)
            full_ds = torchvision.datasets.ImageFolder(root, transform=train_transform)
            n = len(full_ds)
            n_val = int(n * val_split)
            train_ds, test_ds = torch.utils.data.random_split(full_ds, [n - n_val, n_val])
    elif ds_type == "Custom CSV":
        import csv as csv_mod
        import numpy as np
        delimiter = config.get("delimiter", ",")
        if delimiter == "\\t":
            delimiter = "\t"
        has_header = config.get("has_header", True)
        target_column = config.get("target_column", "")
        feature_columns = config.get("feature_columns", "")
        normalization = config.get("normalization", "none")

        data_rows = []
        header = None
        with open(root, "r") as f:
            reader = csv_mod.reader(f, delimiter=delimiter)
            if has_header:
                header = next(reader, None)
            for row in reader:
                data_rows.append(row)

        arr = np.array(data_rows, dtype=np.float32)

        # Column selection
        if header and target_column:
            target_idx = header.index(target_column)
        else:
            target_idx = arr.shape[1] - 1

        if header and feature_columns:
            feat_names = [c.strip() for c in feature_columns.split(",")]
            feat_idxs = [header.index(n) for n in feat_names]
        else:
            feat_idxs = [i for i in range(arr.shape[1]) if i != target_idx]

        features = torch.tensor(arr[:, feat_idxs])
        targets = torch.tensor(arr[:, target_idx]).long()

        # Normalization
        if normalization == "standard":
            mean = features.mean(dim=0, keepdim=True)
            std = features.std(dim=0, keepdim=True).clamp(min=1e-7)
            features = (features - mean) / std
        elif normalization == "minmax":
            fmin = features.min(dim=0, keepdim=True).values
            fmax = features.max(dim=0, keepdim=True).values
            features = (features - fmin) / (fmax - fmin).clamp(min=1e-7)

        full_ds = torch.utils.data.TensorDataset(features, targets)
        split = int(len(full_ds) * (1 - val_split))
        train_ds, test_ds = torch.utils.data.random_split(full_ds, [split, len(full_ds) - split])
    elif ds_type == "HuggingFace":
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install 'datasets' package: pip install datasets")

        subset = config.get("subset", "") or None
        hf_ds = load_dataset(root, subset) if subset else load_dataset(root)
        # Get train/test splits
        train_split = hf_ds.get("train", hf_ds[list(hf_ds.keys())[0]])
        test_split = hf_ds.get("test", None)

        # Split train data if no test split exists
        if test_split is None and val_split > 0:
            split_result = train_split.train_test_split(test_size=val_split)
            train_split = split_result["train"]
            test_split = split_result["test"]

        # Custom preprocessing script — highest priority
        preprocess_code = config.get("preprocess", "").strip()
        if preprocess_code:
            # User-provided transform: receives `item` dict, returns (tensor, label)
            exec_globals = {"torch": torch, "json": json}
            indented = "\n".join("    " + line for line in preprocess_code.split("\n"))
            exec(f"def _user_preprocess(item):\n{indented}\n", exec_globals)
            _user_fn = exec_globals["_user_preprocess"]

            class HFCustomDataset(torch.utils.data.Dataset):
                def __init__(self, split):
                    self.split = split
                def __len__(self):
                    return len(self.split)
                def __getitem__(self, idx):
                    result = _user_fn(self.split[idx])
                    if isinstance(result, tuple) and len(result) == 2:
                        feat, label = result
                        if not isinstance(feat, torch.Tensor):
                            feat = torch.tensor(feat, dtype=torch.float32)
                        if not isinstance(label, (int, torch.Tensor)):
                            label = int(label)
                        return feat, label
                    raise ValueError("preprocess must return (features_tensor, label)")

            train_ds = HFCustomDataset(train_split)
            test_ds = HFCustomDataset(test_split) if test_split else None
        else:
            # Auto-detect: image, text, or tabular

        # Column configuration from user params
            user_label_col = config.get("label_column", "") or None
            user_text_col = config.get("text_column", "") or None
            user_image_col = config.get("image_column", "") or None
            image_size = config.get("image_size", 32)
            max_length = config.get("max_length", 128)
            vocab_size = config.get("vocab_size", 25000)

            col_names = train_split.column_names

            # Determine label column
            label_col = user_label_col
            if not label_col:
                for candidate in ["label", "labels", "target", "class", "y"]:
                    if candidate in col_names:
                        label_col = candidate
                        break
            if not label_col:
                label_col = col_names[-1]

            # Check if it's an image dataset
            feature_types = train_split.features
            img_col = user_image_col
            if not img_col:
                for c in col_names:
                    if "image" in str(feature_types[c]).lower() or "img" in c.lower():
                        img_col = c
                        break

            # Check if it's a text dataset
            txt_col = user_text_col
            if not txt_col:
                for c in col_names:
                    if c != label_col and ("text" in c.lower() or "sentence" in c.lower() or "content" in c.lower()):
                        txt_col = c
                        break

            if img_col:
                # Image dataset from HuggingFace
                import torchvision.transforms as T
                transform = T.Compose([
                    T.Resize(image_size),
                    T.CenterCrop(image_size),
                    T.ToTensor(),
                    T.Normalize((0.5,), (0.5,)),
                ])

                class HFImageDataset(torch.utils.data.Dataset):
                    def __init__(self, split):
                        self.split = split
                    def __len__(self):
                        return len(self.split)
                    def __getitem__(self, idx):
                        item = self.split[idx]
                        img = item[img_col]
                        if hasattr(img, 'convert'):
                            img = img.convert("RGB")
                        img = transform(img)
                        label = item[label_col]
                        return img, label

                train_ds = HFImageDataset(train_split)
                test_ds = HFImageDataset(test_split) if test_split else None
            elif txt_col:
                # Text classification dataset
                from collections import Counter
                word_counts = Counter()
                for text in train_split[txt_col]:
                    word_counts.update(str(text).lower().split()[:300])
                _hf_vocab = {"<pad>": 0, "<unk>": 1}
                for i, (w, _) in enumerate(word_counts.most_common(vocab_size - 2)):
                    _hf_vocab[w] = i + 2

                _hf_max_len = max_length

                def _hf_encode(text):
                    tokens = [_hf_vocab.get(w, 1) for w in str(text).lower().split()[:_hf_max_len]]
                    return tokens + [0] * (_hf_max_len - len(tokens))

                class HFTextDataset(torch.utils.data.Dataset):
                    def __init__(self, split):
                        self.texts = split[txt_col]
                        self.labels = split[label_col]
                    def __len__(self):
                        return len(self.labels)
                    def __getitem__(self, idx):
                        return (
                            torch.tensor(_hf_encode(self.texts[idx]), dtype=torch.long),
                            self.labels[idx],
                        )

                train_ds = HFTextDataset(train_split)
                test_ds = HFTextDataset(test_split) if test_split else None
            else:
                # Tabular dataset — handles scalar, list-of-scalar, and nested columns
                feature_cols = [c for c in col_names if c != label_col]

                # Check if columns have complex (dict/nested) types that need preprocessing
                sample_item = train_split[0]
                complex_cols = []
                scalar_cols = []
                list_of_scalar_cols = []
                for c in feature_cols:
                    v = sample_item[c]
                    if isinstance(v, dict):
                        complex_cols.append(c)
                    elif isinstance(v, (list, tuple)):
                        if len(v) > 0 and isinstance(v[0], dict):
                            complex_cols.append(c)
                        else:
                            list_of_scalar_cols.append(c)
                    elif isinstance(v, (int, float)):
                        scalar_cols.append(c)
                    else:
                        try:
                            float(v)
                            scalar_cols.append(c)
                        except (TypeError, ValueError):
                            complex_cols.append(c)

                # Check label column too
                label_sample = sample_item[label_col]
                label_is_complex = isinstance(label_sample, (dict, list, tuple))
                if label_is_complex and isinstance(label_sample, (list, tuple)) and len(label_sample) > 0:
                    label_is_complex = isinstance(label_sample[0], dict)

                if complex_cols or label_is_complex:
                    # Columns contain nested dicts/structs — can't auto-load
                    col_info = ", ".join(f"'{c}' (nested)" for c in complex_cols[:5])
                    if label_is_complex:
                        col_info += f", label '{label_col}' (nested)"
                    raise ValueError(
                        f"This dataset has complex/nested columns: {col_info}. "
                        f"Use the Preprocessing script in the Data node panel to define a custom transform. "
                        f"Example:\n"
                        f"  # item is a dict with keys: {', '.join(col_names[:6])}\n"
                        f"  features = torch.tensor([float(item['{scalar_cols[0]}'])] if scalar_cols else ['...'], dtype=torch.float32)\n"
                        f"  label = int(item['{label_col}'])\n"
                        f"  return features, label"
                    )

                # Simple case: only scalar + list-of-scalar columns
                usable_cols = scalar_cols + list_of_scalar_cols

                def _extract_features(split, s_cols, l_cols):
                    """Extract feature tensor from HF split, handling scalar + list-of-scalar columns."""
                    rows = []
                    for idx in range(len(split)):
                        item = split[idx]
                        vals = []
                        for c in s_cols:
                            vals.append(float(item[c]))
                        for c in l_cols:
                            vals.extend([float(x) for x in item[c]])
                        rows.append(vals)
                    return torch.tensor(rows, dtype=torch.float32)

                def _extract_labels(split, col):
                    """Extract label tensor from HF split."""
                    labels = split[col]
                    if isinstance(labels[0], (list, tuple)):
                        return torch.tensor(labels, dtype=torch.long)
                    return torch.tensor(labels).long()

                try:
                    train_features = _extract_features(train_split, scalar_cols, list_of_scalar_cols)
                    train_labels = _extract_labels(train_split, label_col)
                    train_ds = torch.utils.data.TensorDataset(train_features, train_labels)
                    if test_split:
                        test_features = _extract_features(test_split, scalar_cols, list_of_scalar_cols)
                        test_labels = _extract_labels(test_split, label_col)
                        test_ds = torch.utils.data.TensorDataset(test_features, test_labels)
                    else:
                        test_ds = None
                except Exception as e:
                    raise ValueError(f"Failed to load tabular dataset: {e}. Try adding a Preprocessing script.")
    elif ds_type == "TinyShakespeare":
        import os
        import urllib.request
        os.makedirs(root, exist_ok=True)
        cache_path = os.path.join(root, "tinyshakespeare.txt")
        if not os.path.exists(cache_path):
            send_notification("train.progress", {"status": "downloading", "dataset": "Tiny Shakespeare"})
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
                cache_path,
            )
        with open(cache_path, "r", encoding="utf-8") as f:
            text = f.read()
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
        seq_len = config.get("max_length", 256)
        n = int(0.9 * len(data))

        class CharLMDataset(torch.utils.data.Dataset):
            def __init__(self, data_tensor):
                self.data = data_tensor
            def __len__(self):
                return max(0, len(self.data) - seq_len)
            def __getitem__(self, idx):
                return self.data[idx : idx + seq_len], self.data[idx + 1 : idx + seq_len + 1]

        train_ds = CharLMDataset(data[:n])
        test_ds = CharLMDataset(data[n:])

    elif ds_type == "WikiText2":
        try:
            from datasets import load_dataset as hf_load
        except ImportError:
            raise ImportError("Install 'datasets' package: pip install datasets")
        send_notification("train.progress", {"status": "downloading", "dataset": "WikiText-2"})
        ds = hf_load("wikitext", "wikitext-2-raw-v1")
        train_text = "\n".join(t for t in ds["train"]["text"] if t.strip())
        test_text = "\n".join(t for t in ds["test"]["text"] if t.strip())

        from collections import Counter
        max_vocab = config.get("vocab_size", 30000)
        vocab_counter = Counter(train_text.lower().split())
        vocab = {"<pad>": 0, "<unk>": 1}
        for i, (word, _) in enumerate(vocab_counter.most_common(max_vocab - 2)):
            vocab[word] = i + 2

        def _wt2_encode(text):
            return [vocab.get(w, 1) for w in text.lower().split()]

        train_ids = torch.tensor(_wt2_encode(train_text), dtype=torch.long)
        test_ids = torch.tensor(_wt2_encode(test_text), dtype=torch.long)
        seq_len = config.get("max_length", 128)

        class WordLMDataset(torch.utils.data.Dataset):
            def __init__(self, data_tensor):
                self.data = data_tensor
            def __len__(self):
                return max(0, len(self.data) - seq_len)
            def __getitem__(self, idx):
                return self.data[idx : idx + seq_len], self.data[idx + 1 : idx + seq_len + 1]

        train_ds = WordLMDataset(train_ids)
        test_ds = WordLMDataset(test_ids)

    elif ds_type in ("IMDB", "AGNews"):
        try:
            from datasets import load_dataset as hf_load
        except ImportError:
            raise ImportError("Install 'datasets' package: pip install datasets")
        send_notification("train.progress", {"status": "downloading", "dataset": ds_type})

        if ds_type == "IMDB":
            ds = hf_load("imdb")
            text_col, label_col = "text", "label"
            seq_len = config.get("max_length", 100)
            max_vocab = config.get("vocab_size", 25000)
        else:
            ds = hf_load("ag_news")
            text_col, label_col = "text", "label"
            seq_len = config.get("max_length", 100)
            max_vocab = config.get("vocab_size", 25000)

        # Build vocabulary from training data
        from collections import Counter
        word_counts = Counter()
        for text in ds["train"][text_col]:
            word_counts.update(text.lower().split()[:300])
        _txt_vocab = {"<pad>": 0, "<unk>": 1}
        for i, (w, _) in enumerate(word_counts.most_common(max_vocab - 2)):
            _txt_vocab[w] = i + 2

        def _txt_encode(text):
            tokens = [_txt_vocab.get(w, 1) for w in text.lower().split()[:seq_len]]
            return tokens + [0] * (seq_len - len(tokens))

        class TextClassDataset(torch.utils.data.Dataset):
            def __init__(self, split):
                self.texts = split[text_col]
                self.labels = split[label_col]
            def __len__(self):
                return len(self.labels)
            def __getitem__(self, idx):
                return (
                    torch.tensor(_txt_encode(self.texts[idx]), dtype=torch.long),
                    self.labels[idx],
                )

        train_ds = TextClassDataset(ds["train"])
        test_ds = TextClassDataset(ds["test"])

    elif ds_type == "SpeechCommands":
        try:
            import torchaudio
            # Patch torchaudio.load to use soundfile if torchcodec/FFmpeg unavailable
            try:
                torchaudio.load(torch.zeros(1))  # test if default backend works
            except Exception:
                try:
                    import soundfile as sf
                    _orig_load = torchaudio.load
                    def _sf_load(filepath, *args, **kwargs):
                        data, sr = sf.read(str(filepath), dtype="float32")
                        tensor = torch.from_numpy(data).unsqueeze(0) if data.ndim == 1 else torch.from_numpy(data).T
                        return tensor, sr
                    torchaudio.load = _sf_load
                except ImportError:
                    pass
        except ImportError:
            raise ImportError("Install torchaudio: pip install torchaudio soundfile")
        send_notification("train.progress", {"status": "downloading", "dataset": "Speech Commands"})

        train_raw = torchaudio.datasets.SPEECHCOMMANDS(root, subset="training", download=True)
        test_raw = torchaudio.datasets.SPEECHCOMMANDS(root, subset="testing", download=True)

        # Build label map from the 35 commands
        all_labels = sorted({item[2] for item in train_raw})
        _sc_label2idx = {l: i for i, l in enumerate(all_labels)}

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=80, n_fft=400, hop_length=160,
        )

        class SCDataset(torch.utils.data.Dataset):
            def __init__(self, ds_raw):
                self.ds = ds_raw
            def __len__(self):
                return len(self.ds)
            def __getitem__(self, idx):
                waveform, sample_rate, label = self.ds[idx][:3]
                # Pad/truncate to 1 second (16000 samples)
                if waveform.size(1) < 16000:
                    waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.size(1)))
                else:
                    waveform = waveform[:, :16000]
                mel = mel_transform(waveform).squeeze(0)  # [80, 100]
                return mel, _sc_label2idx[label]

        train_ds = SCDataset(train_raw)
        test_ds = SCDataset(test_raw)

    elif ds_type == "TensorFile":
        import os
        import numpy as np
        features_key = config.get("features_key", "") or None
        labels_key = config.get("labels_key", "") or None
        normalization = config.get("normalization", "none")

        def _load_tensor_file(path: str):
            """Load tensors from a single file (.pt, .npy, .npz, .safetensors)."""
            ext = os.path.splitext(path)[1].lower()
            if ext == ".pt":
                return torch.load(path, map_location="cpu", weights_only=True)
            elif ext in (".npy", ".npz"):
                raw_np = np.load(path, allow_pickle=False)
                if ext == ".npz":
                    return {k: torch.from_numpy(raw_np[k]) for k in raw_np.files}
                return torch.from_numpy(raw_np)
            elif ext == ".safetensors":
                try:
                    from safetensors.torch import load_file
                    return load_file(path)
                except ImportError:
                    raise ImportError("Install safetensors: pip install safetensors")
            else:
                raise ValueError(f"Unsupported file format: {ext}")

        if os.path.isdir(root):
            # Folder mode: discover tensor files and merge them
            tensor_files = sorted([
                f for f in os.listdir(root)
                if os.path.splitext(f)[1].lower() in (".pt", ".npy", ".npz", ".safetensors")
            ])
            if not tensor_files:
                raise ValueError(f"No tensor files (.pt, .npy, .npz, .safetensors) found in {root}")
            # Load all files and merge into a single dict
            raw: dict | torch.Tensor = {}
            for tf in tensor_files:
                loaded = _load_tensor_file(os.path.join(root, tf))
                if isinstance(loaded, dict):
                    for k, v in loaded.items():
                        raw[k] = v  # type: ignore
                elif isinstance(loaded, torch.Tensor):
                    name = os.path.splitext(tf)[0]
                    raw[name] = loaded  # type: ignore
            if not raw:
                raise ValueError(f"No tensors found in files in {root}")
        else:
            raw = _load_tensor_file(root)

        # Handle-ordered loading: if data_outputs is provided, use it to order tensors
        data_outputs = config.get("data_outputs")
        if data_outputs and isinstance(raw, dict):
            DTYPE_MAP = {
                "float32": lambda t: t.float(),
                "float16": lambda t: t.half(),
                "long": lambda t: t.long(),
                "int32": lambda t: t.int(),
            }
            tensors_ordered = []
            for out in data_outputs:
                key = out.get("column") or out["id"]
                if key not in raw:
                    raise ValueError(f"Tensor '{key}' not found in dataset. Available: {list(raw.keys())}")
                t = raw[key]
                cast_fn = DTYPE_MAP.get(out.get("dtype", ""), None)
                if cast_fn:
                    t = cast_fn(t)
                tensors_ordered.append(t)

            # Verify first dimensions match
            n = tensors_ordered[0].shape[0]
            for i, t in enumerate(tensors_ordered):
                if t.shape[0] != n:
                    raise ValueError(
                        f"Tensor '{data_outputs[i]['id']}' has {t.shape[0]} samples, expected {n}"
                    )

            full_ds = torch.utils.data.TensorDataset(*tensors_ordered)
        else:
            # Legacy auto-detect: features + labels
            if isinstance(raw, dict):
                keys = list(raw.keys())
                if features_key and features_key in raw:
                    features = raw[features_key].float()
                else:
                    features = max(raw.values(), key=lambda t: t.numel()).float()
                if labels_key and labels_key in raw:
                    targets = raw[labels_key].long()
                else:
                    candidates = [v for k, v in raw.items() if v is not features]
                    targets = None
                    for c in candidates:
                        if c.dtype in (torch.int32, torch.int64, torch.long):
                            targets = c.long()
                            break
                    if targets is None and candidates:
                        targets = candidates[0].long()
                    if targets is None:
                        raise ValueError("Cannot auto-detect labels tensor. Specify labels_key.")
            elif isinstance(raw, torch.Tensor):
                if raw.dim() == 2:
                    features = raw[:, :-1].float()
                    targets = raw[:, -1].long()
                else:
                    raise ValueError("Single tensor must be 2D (samples x features+label)")
            else:
                raise ValueError(f"Unexpected data format in {root}")

            # Normalization
            if normalization == "standard":
                mean = features.mean(dim=0, keepdim=True)
                std = features.std(dim=0, keepdim=True).clamp(min=1e-7)
                features = (features - mean) / std
            elif normalization == "minmax":
                fmin = features.min(dim=0, keepdim=True).values
                fmax = features.max(dim=0, keepdim=True).values
                features = (features - fmin) / (fmax - fmin).clamp(min=1e-7)

            full_ds = torch.utils.data.TensorDataset(features, targets)

        split = int(len(full_ds) * (1 - val_split))
        train_ds, test_ds = torch.utils.data.random_split(full_ds, [split, len(full_ds) - split])

    else:
        raise ValueError(f"Unsupported dataset: {ds_type}")

    generator = None
    if dataloader_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(dataloader_seed)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0) if test_ds else None
    return train_loader, test_loader


def _compute_loss(criterion, output, target, data=None):
    """Handle classification [B,C], language modeling [B,T,C], reconstruction, and segmentation outputs."""
    # Auto-cast target dtype based on loss function requirements
    if isinstance(criterion, (nn.MSELoss, nn.L1Loss, nn.HuberLoss, nn.SmoothL1Loss)):
        # Regression losses need float targets
        target = target.float()
        # Squeeze output [B,1] to [B] to match scalar targets
        if output.dim() == 2 and output.size(-1) == 1 and target.dim() == 1:
            output = output.squeeze(-1)
    elif isinstance(criterion, (nn.CrossEntropyLoss, nn.NLLLoss)):
        # Classification losses need long targets
        target = target.long()
    elif isinstance(criterion, (nn.BCEWithLogitsLoss, nn.BCELoss)):
        # Binary losses need float targets
        target = target.float()
    elif isinstance(criterion, nn.KLDivLoss):
        target = target.float()
    if output.dim() == 3 and target.dim() <= 2:
        return criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))
    # For reconstruction losses: reshape target to match output if total elements agree
    # (e.g., FC autoencoder: output=[B,784], target=[B,1,28,28])
    if isinstance(criterion, (nn.MSELoss, nn.L1Loss, nn.HuberLoss, nn.SmoothL1Loss)):
        if output.shape != target.shape:
            try:
                target = target.reshape(output.shape).float()
            except RuntimeError:
                pass
    # For segmentation (BCEWithLogitsLoss): if output is spatial [B,C,H,W] but target is [B],
    # create random binary targets matching output shape
    if isinstance(criterion, nn.BCEWithLogitsLoss) and output.dim() >= 3 and target.dim() == 1:
        seg_target = torch.zeros_like(output)
        return criterion(output, seg_target)
    return criterion(output, target)


def _compute_correct(output, target):
    """Return (correct, total) for accuracy."""
    if output.dim() == 3 and target.dim() <= 2:
        _, pred = output.reshape(-1, output.size(-1)).max(-1)
        return pred.eq(target.reshape(-1)).sum().item(), target.numel()
    if output.dim() >= 2 and output.size(-1) > 1:
        _, pred = output.max(-1)
        return pred.eq(target).sum().item(), target.size(0)
    return 0, 0


def _train_loop(graph_json: dict, config: dict):
    """Runs in a separate thread. Captures stdout/stderr as train.log notifications."""
    global _train_stop_event, _last_trained_state
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = _LogCapture("stdout")
    sys.stderr = _LogCapture("stderr")
    try:
        # Seed initialization for reproducibility
        seed = config.get("seed")
        if seed is not None:
            import random
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            try:
                import numpy as np
                np.random.seed(seed)
            except ImportError:
                pass

        if config.get("deterministic", False):
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True

        model = _build_model_from_graph(graph_json)
        _apply_weight_init(model, config.get("weight_init", "default"))
        device_str = config.get("device", "auto")
        device = _get_device(config.get("device", "auto"))
        model = model.to(device)

        # Log model info
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model: {param_count:,} parameters")
        print(f"Device: {device}")
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available. Training on CPU (slow).", file=sys.stderr)
            print("To enable GPU: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126", file=sys.stderr)

        # Detect multi-input models (e.g. encoder-decoder Transformer)
        num_inputs = sum(1 for n in graph_json["nodes"]
                         if n.get("type") == "placeholder" or n.get("moduleType") == "Input")

        optimizer = _build_optimizer(config.get("optimizer", {}), model.parameters())

        # Extract loss from graph if a Loss node exists, otherwise use config
        graph_loss_type, graph_loss_params, target_batch_idx = _extract_loss_from_graph(graph_json)
        # Reconstruction mode: target is the input itself (Data.x → Loss.target, batch idx 0)
        is_reconstruction = target_batch_idx <= 0
        if graph_loss_type:
            criterion = _build_loss(graph_loss_type, graph_loss_params)
        else:
            criterion = _build_loss(config.get("loss", "CrossEntropyLoss"))

        loss_name = graph_loss_type or config.get("loss", "CrossEntropyLoss")
        print(f"Loss: {loss_name}" + (" (reconstruction)" if is_reconstruction else ""))
        print(f"Optimizer: {config.get('optimizer', {}).get('type', 'AdamW')} lr={config.get('optimizer', {}).get('lr', 0.001)}")

        epochs = config.get("epochs", 10)
        batch_size = config.get("batch_size", 32)

        print(f"Loading dataset...")
        train_loader, test_loader = _load_dataset(config.get("dataset", {}), batch_size, config.get("augmentation"), config.get("dataloader_seed"), is_reconstruction=is_reconstruction)
        print(f"Dataset: {len(train_loader.dataset)} train samples, batch_size={batch_size}")
        if test_loader:
            print(f"         {len(test_loader.dataset)} test samples")
        print(f"Training for {epochs} epochs...")

        scheduler = _build_scheduler(config.get("scheduler"), optimizer, epochs=epochs, steps_per_epoch=len(train_loader))
        step_per_batch = isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)

        import os
        import time
        ckpt_dir = config.get("checkpoint_dir") or os.path.join(_data_dir(), "checkpoints", time.strftime("%Y%m%d-%H%M%S"))
        print(f"Checkpoints: {os.path.abspath(ckpt_dir)}")

        use_amp = config.get("mixed_precision", False) and device.type == "cuda"
        scaler = torch.amp.GradScaler() if use_amp else None

        # Gradient clipping
        grad_clip = config.get("gradient_clipping")
        clip_max_norm = grad_clip["max_norm"] if grad_clip else None

        # Early stopping
        early_stop_cfg = config.get("early_stopping")
        best_val_loss = float("inf")
        patience_counter = 0

        # Resume from checkpoint
        resume_path = config.get("resume_checkpoint")
        start_epoch = 1
        if resume_path:
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            if "rng_state" in ckpt:
                torch.random.set_rng_state(ckpt["rng_state"])
            if "cuda_rng_state" in ckpt and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
            send_notification("train.progress", {"status": "checkpoint", "path": resume_path, "epoch": start_epoch - 1})

        steps_per_epoch = len(train_loader)
        for epoch in range(start_epoch, epochs + 1):
            if _train_stop_event.is_set():
                break

            model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                if _train_stop_event.is_set():
                    break

                data = data.to(device)
                # Reconstruction: use input as target (Data.x → Loss.target)
                target = data if is_reconstruction else target.to(device)
                optimizer.zero_grad()

                # For multi-input models (e.g. encoder-decoder), duplicate input
                model_args = (data,) * num_inputs

                if use_amp and scaler:
                    with torch.amp.autocast(device_type=device.type):
                        output = model(*model_args)
                        loss = _compute_loss(criterion, output, target, data)
                    scaler.scale(loss).backward()
                    if clip_max_norm:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(*model_args)
                    loss = _compute_loss(criterion, output, target, data)
                    loss.backward()
                    if clip_max_norm:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
                    optimizer.step()

                if step_per_batch:
                    scheduler.step()

                epoch_loss += loss.item()
                num_batches += 1

                # Report every 50 batches
                if batch_idx % 50 == 0:
                    send_notification("train.progress", {
                        "epoch": epoch,
                        "step": batch_idx,
                        "total_steps": steps_per_epoch,
                        "train_loss": loss.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                    })

            # Epoch end
            avg_loss = epoch_loss / max(num_batches, 1)

            # Skip evaluation if stop requested
            if _train_stop_event.is_set():
                break

            # Training accuracy (skip for reconstruction — accuracy is meaningless)
            train_acc = None
            if not is_reconstruction and num_batches > 0:
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for data_b, target_b in train_loader:
                        if _train_stop_event.is_set():
                            break
                        data_b, target_b = data_b.to(device), target_b.to(device)
                        out_b = model(*((data_b,) * num_inputs))
                        c, t = _compute_correct(out_b, target_b)
                        correct += c
                        total += t
                        if total >= 2000:  # Sample for speed
                            break
                if total > 0:
                    train_acc = correct / total

            # Validation
            val_loss = None
            val_acc = None
            if test_loader and not _train_stop_event.is_set():
                model.eval()
                total_val = 0.0
                n_val = 0
                v_correct = 0
                v_total = 0
                with torch.no_grad():
                    for data_b, target_b in test_loader:
                        if _train_stop_event.is_set():
                            break
                        data_b = data_b.to(device)
                        target_b = data_b if is_reconstruction else target_b.to(device)
                        output_b = model(*((data_b,) * num_inputs))
                        total_val += _compute_loss(criterion, output_b, target_b, data_b).item()
                        n_val += 1
                        if not is_reconstruction:
                            c, t = _compute_correct(output_b, target_b)
                            v_correct += c
                            v_total += t
                val_loss = total_val / max(n_val, 1)
                if not is_reconstruction and v_total > 0:
                    val_acc = v_correct / v_total

            # GPU memory stats
            gpu_info = None
            if device.type == "cuda":
                props = torch.cuda.get_device_properties(device)
                gpu_info = {
                    "allocated_mb": round(torch.cuda.memory_allocated(device) / 1024 / 1024, 1),
                    "reserved_mb": round(torch.cuda.memory_reserved(device) / 1024 / 1024, 1),
                    "max_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1024 / 1024, 1),
                    "total_mb": round(props.total_memory / 1024 / 1024, 1),
                }

            # Log epoch summary
            parts = [f"Epoch {epoch}/{epochs}", f"loss={avg_loss:.4f}"]
            if val_loss is not None:
                parts.append(f"val_loss={val_loss:.4f}")
            if train_acc is not None:
                parts.append(f"acc={train_acc*100:.1f}%")
            if val_acc is not None:
                parts.append(f"val_acc={val_acc*100:.1f}%")
            parts.append(f"lr={optimizer.param_groups[0]['lr']:.2e}")
            print("  ".join(parts))

            send_notification("train.progress", {
                "epoch": epoch,
                "step": num_batches,
                "train_loss": avg_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
                "gpu": gpu_info,
            })

            # Auto-checkpoint every 5 epochs
            if epoch % 5 == 0 or epoch == epochs:
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.abspath(os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch}.pt"))
                ckpt_data = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "rng_state": torch.random.get_rng_state(),
                }
                if torch.cuda.is_available():
                    ckpt_data["cuda_rng_state"] = torch.cuda.get_rng_state_all()
                torch.save(ckpt_data, ckpt_path)
                send_notification("train.progress", {
                    "status": "checkpoint",
                    "path": ckpt_path,
                    "epoch": epoch,
                })

            if scheduler and not step_per_batch:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss if val_loss is not None else avg_loss)
                else:
                    scheduler.step()

            # Early stopping check
            if early_stop_cfg:
                check_loss = val_loss if val_loss is not None else avg_loss
                if check_loss < best_val_loss - early_stop_cfg.get("min_delta", 0.001):
                    best_val_loss = check_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stop_cfg.get("patience", 5):
                        send_notification("train.progress", {"status": "early_stop", "epoch": epoch})
                        break

        _last_trained_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print("Training complete.")
        send_notification("train.progress", {"status": "complete"})

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback as tb
        tb.print_exc()
        send_notification("train.progress", {"status": "error", "error": str(e)})
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


def train_start(params: dict) -> dict:
    global _train_thread, _train_stop_event
    _train_stop_event.clear()
    _train_thread = threading.Thread(
        target=_train_loop,
        args=(params["graph"], params.get("config", {})),
        daemon=True,
    )
    _train_thread.start()
    return {"status": "started"}


def train_stop(params: dict) -> dict:
    _train_stop_event.set()
    return {"status": "stopping"}


def system_gpu_info(params: dict) -> dict:
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            gpus.append({
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total": props.total_memory,
                "memory_allocated": allocated,
                "memory_reserved": reserved,
                "memory_free": props.total_memory - reserved,
                "utilization_percent": round(allocated / props.total_memory * 100, 1) if props.total_memory > 0 else 0,
            })
    return {"cuda_available": torch.cuda.is_available(), "gpus": gpus}


def _apply_weight_init(model: nn.Module, init_type: str = "default"):
    """Apply weight initialization to model."""
    if init_type == "default":
        return  # PyTorch defaults
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if init_type == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight)
            elif init_type == "xavier_normal":
                nn.init.xavier_normal_(m.weight)
            elif init_type == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            elif init_type == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight)
            elif init_type == "zeros":
                nn.init.zeros_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _build_export_model(params: dict) -> tuple[nn.Module, str]:
    """Build the model for export and load the requested weights.

    params["weights"]: "trained" (last training run, default), "checkpoint"
    (params["checkpoint_path"]), or "random" (fresh initialization).
    """
    model = _build_model_from_graph(params["graph"])
    source = params.get("weights", "trained")

    if source == "checkpoint":
        path = params.get("checkpoint_path")
        if not path:
            raise ValueError("No checkpoint file selected")
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else None
        if not isinstance(state, dict):
            raise ValueError(f"{path} does not contain a model state dict")
    elif source == "trained":
        if _last_trained_state is None:
            raise ValueError("No trained weights yet. Train the model first, "
                             "pick a checkpoint file, or choose 'Untrained'.")
        state = _last_trained_state
    else:
        return model.eval(), "random"

    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        raise ValueError("The saved weights don't match the current graph. "
                         f"Did the architecture change since training?\n{e}") from e
    return model.eval(), source


def export_onnx(params: dict) -> dict:
    """Export model to ONNX format."""
    import os
    model, weights = _build_export_model(params)

    input_shapes = params.get("input_shapes", {"x": [1, 3, 224, 224]})
    dummy_inputs = []
    input_names = []
    for name, shape in input_shapes.items():
        dummy_inputs.append(torch.randn(*shape))
        input_names.append(name)

    output_path = os.path.abspath(params.get("output_path", "model.onnx"))
    dummy = tuple(dummy_inputs) if len(dummy_inputs) > 1 else dummy_inputs[0]

    # Suppress verbose output to avoid Windows encoding issues with emoji
    old_stdout = sys.stdout
    sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    try:
        torch.onnx.export(
            model, dummy, output_path,
            input_names=input_names,
            output_names=["output"],
            dynamic_axes={n: {0: "batch"} for n in input_names},
            opset_version=17,
        )
    finally:
        sys.stdout = old_stdout
    return {"path": output_path, "success": True, "weights": weights}


def export_torchscript(params: dict) -> dict:
    """Export model to TorchScript format."""
    import os
    model, weights = _build_export_model(params)

    method = params.get("method", "trace")
    output_path = os.path.abspath(params.get("output_path", "model.pt"))

    if method == "script":
        scripted = torch.jit.script(model)
        scripted.save(output_path)
    else:
        input_shapes = params.get("input_shapes", {"x": [1, 3, 224, 224]})
        dummy_inputs = [torch.randn(*shape) for shape in input_shapes.values()]
        dummy = tuple(dummy_inputs) if len(dummy_inputs) > 1 else dummy_inputs[0]
        traced = torch.jit.trace(model, dummy)
        traced.save(output_path)

    return {"path": output_path, "success": True, "weights": weights}


def _tensor_info(name: str, tensor: torch.Tensor) -> dict:
    """Build metadata + basic stats for a tensor."""
    info: dict = {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "num_samples": tensor.shape[0] if tensor.dim() > 0 else 1,
    }
    try:
        flat = tensor.reshape(-1).float()
        info["min"] = round(flat.min().item(), 4)
        info["max"] = round(flat.max().item(), 4)
        if tensor.is_floating_point():
            info["mean"] = round(flat.mean().item(), 4)
            info["std"] = round(flat.std().item(), 4)
        else:
            # Integer tensor — likely class labels
            unique = tensor.unique()
            info["num_classes"] = len(unique)
            if len(unique) <= 20:
                info["class_values"] = sorted(unique.tolist())
    except Exception:
        pass
    return info


def data_scan(params: dict) -> dict:
    """Scan a tensor file/folder and return metadata without loading all data.

    Supports:
      - .pt file (dict of tensors or single tensor)
      - .npz file (numpy archive)
      - .npy file (single numpy array)
      - .safetensors file
      - Folder containing any of the above
    """
    import os
    import numpy as np

    path = params["path"]
    if not path:
        raise ValueError("No path provided")

    def _scan_file(filepath: str) -> list[dict]:
        """Return tensor metadata from a single file."""
        ext = os.path.splitext(filepath)[1].lower()
        results = []

        if ext == ".pt":
            raw = torch.load(filepath, map_location="cpu", weights_only=True)
            if isinstance(raw, dict):
                for name, tensor in raw.items():
                    if isinstance(tensor, torch.Tensor):
                        results.append(_tensor_info(name, tensor))
            elif isinstance(raw, torch.Tensor):
                results.append(_tensor_info("data", raw))

        elif ext == ".npz":
            raw = np.load(filepath, allow_pickle=False)
            for name in raw.files:
                arr = raw[name]
                results.append(_tensor_info(name, torch.from_numpy(arr)))

        elif ext == ".npy":
            arr = np.load(filepath, allow_pickle=False)
            results.append(_tensor_info("data", torch.from_numpy(arr)))

        elif ext == ".safetensors":
            try:
                from safetensors import safe_open
                with safe_open(filepath, framework="pt") as f:
                    for name in f.keys():
                        tensor = f.get_tensor(name)
                        results.append(_tensor_info(name, tensor))
            except ImportError:
                raise ImportError("Install safetensors: pip install safetensors")
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        return results

    tensor_exts = {".pt", ".npy", ".npz", ".safetensors"}

    if os.path.isdir(path):
        files = sorted([
            f for f in os.listdir(path)
            if os.path.splitext(f)[1].lower() in tensor_exts
        ])
        if not files:
            raise ValueError(f"No tensor files (.pt, .npy, .npz, .safetensors) found in {path}")

        all_tensors = []
        for f in files:
            filepath = os.path.join(path, f)
            ext = os.path.splitext(f)[1].lower()
            scanned = _scan_file(filepath)
            for info in scanned:
                # For single-tensor files (.npy, single .pt), use filename as name
                if info["name"] == "data":
                    info["name"] = os.path.splitext(f)[0]
                all_tensors.append(info)

        total = all_tensors[0]["num_samples"] if all_tensors else 0
        return {"tensors": all_tensors, "total_samples": total, "format": "folder"}
    else:
        if not os.path.isfile(path):
            raise ValueError(f"File not found: {path}")
        tensors = _scan_file(path)
        total = tensors[0]["num_samples"] if tensors else 0
        return {"tensors": tensors, "total_samples": total, "format": os.path.splitext(path)[1].lstrip(".")}


def data_preview(params: dict) -> dict:
    """Preview dataset: return metadata about first N samples."""
    config = params.get("config", {})
    batch_size = params.get("batch_size", 32)
    try:
        train_loader, test_loader = _load_dataset(config, batch_size)
    except Exception as e:
        return {"error": str(e)}

    # Get a sample batch
    batch = next(iter(train_loader))
    data, target = batch[0], batch[1]

    # Compute stats
    train_size = len(train_loader.dataset)
    val_size = len(test_loader.dataset) if test_loader else 0

    num_classes = 0
    try:
        if hasattr(train_loader.dataset, 'dataset'):
            # random_split wraps the original dataset
            ds = train_loader.dataset.dataset
        else:
            ds = train_loader.dataset
        if hasattr(ds, 'targets'):
            num_classes = len(set(t if isinstance(t, int) else t.item() for t in ds.targets))
        elif hasattr(ds, 'tensors') and len(ds.tensors) > 1:
            num_classes = len(ds.tensors[1].unique())
        else:
            num_classes = int(target.max().item()) + 1
    except Exception:
        num_classes = int(target.max().item()) + 1

    return {
        "train_size": train_size,
        "val_size": val_size,
        "sample_shape": list(data.shape[1:]),
        "sample_dtype": str(data.dtype),
        "num_classes": num_classes,
        "feature_stats": {
            "mean": float(data.float().mean()),
            "std": float(data.float().std()),
            "min": float(data.float().min()),
            "max": float(data.float().max()),
        },
    }


def data_probe(params: dict) -> dict:
    """Probe dataset pipeline: run one batch to discover output tensor contract (shape/dtype/num_classes)."""
    config = params.get("config", {})
    batch_size = 4  # small batch — we only need shape/dtype
    try:
        train_loader, _test_loader = _load_dataset(config, batch_size)
    except Exception as e:
        return {"error": str(e)}

    batch = next(iter(train_loader))
    data, target = batch[0], batch[1]

    num_classes = 0
    try:
        if hasattr(train_loader.dataset, 'dataset'):
            ds = train_loader.dataset.dataset
        else:
            ds = train_loader.dataset
        if hasattr(ds, 'targets'):
            num_classes = len(set(t if isinstance(t, int) else t.item() for t in ds.targets))
        elif hasattr(ds, 'tensors') and len(ds.tensors) > 1:
            num_classes = len(ds.tensors[1].unique())
        else:
            num_classes = int(target.max().item()) + 1
    except Exception:
        num_classes = int(target.max().item()) + 1

    return {
        "sample_shape": list(data.shape[1:]),
        "sample_dtype": str(data.dtype).replace("torch.", ""),
        "num_classes": num_classes,
        "label_dtype": str(target.dtype).replace("torch.", ""),
    }


def data_introspect(params: dict) -> dict:
    """Introspect a dataset without fully downloading it.
    Returns column info, label values, sample shape, detected domain, splits."""
    source = params.get("source", "")
    dataset_name = params.get("dataset_name", "")
    subset = params.get("subset", "") or None
    path = params.get("path", "")

    # Known presets — return metadata immediately
    PRESET_META = {
        "CIFAR10": {"domain": "vision", "columns": ["image", "label"], "column_types": {"image": "Image", "label": "ClassLabel"},
                    "label_values": [str(i) for i in range(10)], "num_classes": 10, "sample_shape": [3, 32, 32],
                    "splits": ["train", "test"], "total_samples": 60000},
        "CIFAR100": {"domain": "vision", "columns": ["image", "label"], "column_types": {"image": "Image", "label": "ClassLabel"},
                     "label_values": [], "num_classes": 100, "sample_shape": [3, 32, 32],
                     "splits": ["train", "test"], "total_samples": 60000},
        "MNIST": {"domain": "vision", "columns": ["image", "label"], "column_types": {"image": "Image", "label": "ClassLabel"},
                  "label_values": [str(i) for i in range(10)], "num_classes": 10, "sample_shape": [1, 28, 28],
                  "splits": ["train", "test"], "total_samples": 70000},
        "FashionMNIST": {"domain": "vision", "columns": ["image", "label"], "column_types": {"image": "Image", "label": "ClassLabel"},
                         "label_values": ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"],
                         "num_classes": 10, "sample_shape": [1, 28, 28], "splits": ["train", "test"], "total_samples": 70000},
        "TinyShakespeare": {"domain": "text", "columns": ["text"], "column_types": {"text": "string"},
                           "label_values": [], "num_classes": 65, "sample_shape": [256],
                           "splits": ["train", "test"], "total_samples": 40000},
        "WikiText2": {"domain": "text", "columns": ["text"], "column_types": {"text": "string"},
                      "label_values": [], "num_classes": 30000, "sample_shape": [128],
                      "splits": ["train", "validation", "test"], "total_samples": 36718},
        "IMDB": {"domain": "text", "columns": ["text", "label"], "column_types": {"text": "string", "label": "ClassLabel"},
                 "label_values": ["neg", "pos"], "num_classes": 2, "sample_shape": [100],
                 "splits": ["train", "test"], "total_samples": 50000},
        "AGNews": {"domain": "text", "columns": ["text", "label"], "column_types": {"text": "string", "label": "ClassLabel"},
                   "label_values": ["World", "Sports", "Business", "Sci/Tech"], "num_classes": 4, "sample_shape": [100],
                   "splits": ["train", "test"], "total_samples": 127600},
        "SpeechCommands": {"domain": "audio", "columns": ["waveform", "label"], "column_types": {"waveform": "Audio", "label": "ClassLabel"},
                          "label_values": [], "num_classes": 35, "sample_shape": [80, 100],
                          "splits": ["train", "validation", "test"], "total_samples": 105829},
    }

    if source in PRESET_META:
        return PRESET_META[source]

    if source == "HuggingFace" and dataset_name:
        try:
            from datasets import load_dataset_builder
        except ImportError:
            raise ImportError("Install 'datasets' package: pip install datasets")

        try:
            builder = load_dataset_builder(dataset_name, subset)
            info = builder.info

            columns = []
            column_types = {}
            if info.features:
                for name, feat in info.features.items():
                    columns.append(name)
                    column_types[name] = type(feat).__name__

            # Detect domain from column types
            domain = "tabular"
            has_complex = False
            complex_cols = []
            for col, ctype in column_types.items():
                if ctype == "Image" or "image" in col.lower():
                    domain = "vision"
                    break
                if ctype == "Audio" or "audio" in col.lower():
                    domain = "audio"
                    break
                if ctype in ("Value", "LargeString") or "text" in col.lower() or "sentence" in col.lower():
                    domain = "text"
                # Detect complex/nested types that need preprocessing
                if ctype in ("Sequence", "List", "dict"):
                    has_complex = True
                    complex_cols.append(col)

            # Extract label info
            label_values: list = []
            num_classes = 0
            label_col_name = ""
            for col, feat in (info.features or {}).items():
                if hasattr(feat, "names"):
                    label_values = list(feat.names)
                    num_classes = len(label_values)
                    label_col_name = col
                    break

            if not label_col_name:
                for candidate in ["label", "labels", "target", "class", "y"]:
                    if candidate in columns:
                        label_col_name = candidate
                        break
                if not label_col_name and columns:
                    label_col_name = columns[-1]

            # Sample shape — hard to get without downloading, leave empty
            sample_shape: list = []

            # Splits
            splits = list(info.splits.keys()) if info.splits else []

            # Total samples
            total_samples = 0
            if info.splits:
                total_samples = sum(s.num_examples for s in info.splits.values())

            result: dict = {
                "domain": domain,
                "columns": columns,
                "column_types": column_types,
                "label_values": label_values,
                "num_classes": num_classes,
                "sample_shape": sample_shape,
                "splits": splits,
                "total_samples": total_samples,
            }

            # Flag datasets that need preprocessing
            if has_complex and domain == "tabular":
                result["needs_preprocessing"] = True
                result["complex_columns"] = complex_cols
                scalar_cols = [c for c in columns if c != label_col_name and column_types.get(c) == "Value"]
                result["detected_label"] = label_col_name
                result["example_preprocess"] = (
                    f"# Each item is a dict with keys: {', '.join(columns[:6])}{'...' if len(columns) > 6 else ''}\n"
                    + (f"feat = torch.tensor([float(item['{scalar_cols[0]}'])], dtype=torch.float32)\n" if scalar_cols else "# Complex/nested columns detected — write your preprocessing below\n# See docs: https://github.com/mdsunbeam/tensorbloom/blob/main/docs/guides/custom-data.md\nraise NotImplementedError('This dataset has complex columns that need a preprocessing script')\n")
                    + f"label = int(item['{label_col_name}'])\n"
                    + "return feat, label"
                )

            return result
        except Exception as e:
            raise ValueError(f"Failed to introspect '{dataset_name}': {e}")

    if source == "CSV" and path:
        import csv as csv_mod
        try:
            with open(path, "r") as f:
                reader = csv_mod.reader(f)
                header = next(reader, None)
                rows = []
                for i, row in enumerate(reader):
                    if i >= 10:
                        break
                    rows.append(row)
            if not header:
                return {"domain": "tabular", "columns": [], "column_types": {}, "label_values": [],
                        "num_classes": 0, "sample_shape": [], "splits": [], "total_samples": 0}

            column_types = {}
            for i, col in enumerate(header):
                vals = [r[i] for r in rows if i < len(r)]
                try:
                    [float(v) for v in vals]
                    column_types[col] = "numeric"
                except (ValueError, IndexError):
                    column_types[col] = "string"

            return {
                "domain": "tabular",
                "columns": list(header),
                "column_types": column_types,
                "label_values": [],
                "num_classes": 0,
                "sample_shape": [len(header) - 1],
                "splits": [],
                "total_samples": 0,
            }
        except Exception as e:
            raise ValueError(f"Failed to introspect CSV '{path}': {e}")

    if source == "ImageFolder" and path:
        import os
        try:
            classes = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
            # Check for train/val subdirectory structure
            if "train" in classes:
                train_dir = os.path.join(path, "train")
                classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
            total = 0
            for c in classes:
                cdir = os.path.join(path, c)
                if os.path.isdir(cdir):
                    total += len([f for f in os.listdir(cdir) if not f.startswith(".")])
            return {
                "domain": "vision",
                "columns": ["image", "label"],
                "column_types": {"image": "Image", "label": "ClassLabel"},
                "label_values": classes,
                "num_classes": len(classes),
                "sample_shape": [],
                "splits": [],
                "total_samples": total,
            }
        except Exception as e:
            raise ValueError(f"Failed to introspect ImageFolder '{path}': {e}")

    if source == "TensorFile" and path:
        import os
        import numpy as np
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".pt":
                raw = torch.load(path, map_location="cpu", weights_only=True)
            elif ext in (".npy", ".npz"):
                raw_np = np.load(path, allow_pickle=False)
                if ext == ".npz":
                    raw = {k: torch.from_numpy(raw_np[k]) for k in raw_np.files}
                else:
                    raw = torch.from_numpy(raw_np)
            else:
                raise ValueError(f"Unsupported format: {ext}")

            if isinstance(raw, dict):
                columns = list(raw.keys())
                column_types = {k: str(v.dtype) for k, v in raw.items()}
                largest = max(raw.values(), key=lambda t: t.numel())
                sample_shape = list(largest.shape[1:]) if largest.dim() > 1 else list(largest.shape)
            elif isinstance(raw, torch.Tensor):
                columns = ["data"]
                column_types = {"data": str(raw.dtype)}
                sample_shape = list(raw.shape[1:]) if raw.dim() > 1 else list(raw.shape)
            else:
                columns = []
                column_types = {}
                sample_shape = []

            return {
                "domain": "tabular",
                "columns": columns,
                "column_types": column_types,
                "label_values": [],
                "num_classes": 0,
                "sample_shape": sample_shape,
                "splits": [],
                "total_samples": 0,
            }
        except Exception as e:
            raise ValueError(f"Failed to introspect TensorFile '{path}': {e}")

    return {"domain": "auto", "columns": [], "column_types": {}, "label_values": [],
            "num_classes": 0, "sample_shape": [], "splits": [], "total_samples": 0}


# ---------------------------------------------------------------------------
# Node probe — run code block forward with dummy inputs to infer output shapes
# ---------------------------------------------------------------------------

def node_probe(params: dict) -> dict:
    """Create a temp nn.Module from init+forward code, run with dummy tensors, return output shapes."""
    init_code = params.get("init_code", "pass")
    forward_code = params.get("forward_code", "return x")
    input_specs = params.get("inputs", [])  # [{name, shape, dtype}]

    if not input_specs:
        return {"error": "No input specs provided"}

    # Build dynamic class
    init_lines = "\n".join(f"        {line}" for line in init_code.strip().split("\n"))
    forward_lines = "\n".join(f"        {line}" for line in forward_code.strip().split("\n"))
    param_names = ", ".join(spec["name"] for spec in input_specs)

    class_code = f"""
class _ProbeModule(nn.Module):
    def __init__(self):
        super().__init__()
{init_lines}

    def forward(self, {param_names}):
{forward_lines}
"""

    exec_globals = {"torch": torch, "nn": nn, "F": torch.nn.functional}
    try:
        exec(class_code, exec_globals)
    except Exception as e:
        return {"error": f"Failed to compile code block: {e}"}

    ProbeClass = exec_globals["_ProbeModule"]
    try:
        module = ProbeClass()
        module.eval()
    except Exception as e:
        return {"error": f"Failed to instantiate code block: {e}"}

    # Create dummy tensors
    dummy_inputs = []
    for spec in input_specs:
        shape = spec.get("shape", [1, 3, 32, 32])
        dtype_str = spec.get("dtype", "float32")
        dtype = getattr(torch, dtype_str, torch.float32)
        dummy_inputs.append(torch.randn(*shape, dtype=dtype))

    try:
        with torch.no_grad():
            result = module(*dummy_inputs)
    except Exception as e:
        return {"error": f"Forward pass failed: {e}"}

    # Parse output shapes
    output_shapes = {}
    if isinstance(result, tuple):
        for i, t in enumerate(result):
            if isinstance(t, torch.Tensor):
                output_shapes[f"out_{i}"] = list(t.shape)
    elif isinstance(result, torch.Tensor):
        output_shapes["out_"] = list(result.shape)
    else:
        return {"error": f"Unexpected return type: {type(result).__name__}"}

    return {"shapes": output_shapes}


# ---------------------------------------------------------------------------
# Build dynamic module from code block definition
# ---------------------------------------------------------------------------

def _build_code_block_module(code_block: dict) -> nn.Module:
    """Build an nn.Module from code block init/forward strings."""
    init_code = code_block.get("initBody", "pass")
    forward_code = code_block.get("forwardBody", "return x")
    input_handles = code_block.get("inputHandles", [{"id": "x"}])

    init_lines = "\n".join(f"        {line}" for line in init_code.strip().split("\n"))
    forward_lines = "\n".join(f"        {line}" for line in forward_code.strip().split("\n"))
    param_names = ", ".join(h["id"] for h in input_handles)

    class_code = f"""
class _DynModule(nn.Module):
    def __init__(self):
        super().__init__()
{init_lines}

    def forward(self, {param_names}):
{forward_lines}
"""

    exec_globals = {"torch": torch, "nn": nn, "F": torch.nn.functional}
    exec(class_code, exec_globals)
    return exec_globals["_DynModule"]()


def _build_custom_block_module(code: str, class_name: str, params: dict) -> nn.Module:
    """Build a module from a user-defined custom block (full class source code)."""
    exec_globals: dict[str, Any] = {"torch": torch, "nn": nn, "F": torch.nn.functional,
                                    "__name__": "tensorbloom_custom"}
    exec(code, exec_globals)
    cls = exec_globals.get(class_name)
    if not (isinstance(cls, type) and issubclass(cls, nn.Module)):
        # Fall back to the last nn.Module subclass defined in the code
        defined = [v for v in exec_globals.values()
                   if isinstance(v, type) and issubclass(v, nn.Module) and v.__module__ == "tensorbloom_custom"]
        if not defined:
            raise ValueError(f"No nn.Module subclass found in custom block '{class_name}'")
        cls = defined[-1]
    clean = {k: v for k, v in params.items() if v is not None and not k.startswith("_")}
    return cls(**clean)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

METHODS = {
    "compile": lambda params: compile_code(params["code"]),
    "decompile": lambda params: decompile_graph(params["graph"]),
    "infer_shapes": lambda params: infer_shapes(params["graph"], params.get("input_shapes", {})),
    "validate": lambda params: validate_graph(params["graph"]),
    "train.start": train_start,
    "train.stop": train_stop,
    "data.scan": data_scan,
    "data.preview": data_preview,
    "data.probe": data_probe,
    "data.introspect": data_introspect,
    "export.onnx": export_onnx,
    "export.torchscript": export_torchscript,
    "node.probe": lambda params: node_probe(params),
    "system.gpu_info": system_gpu_info,
    "system.environment": lambda params: _system_environment(params),
    "ping": lambda params: {"status": "ok", "torch_version": torch.__version__},
}


def _system_environment(params: dict) -> dict:
    return {
        "torchVersion": torch.__version__,
        "cudaVersion": torch.version.cuda if torch.cuda.is_available() else None,
        "pythonVersion": platform.python_version(),
        "platform": platform.platform(),
        "gpuName": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def main():
    send_notification("ready", {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            send_response(None, error={"code": -32700, "message": f"Parse error: {e}"})
            continue

        req_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {})

        handler = METHODS.get(method)
        if handler is None:
            send_response(req_id, error={"code": -32601, "message": f"Unknown method: {method}"})
            continue

        try:
            result = handler(params)
            send_response(req_id, result=result)
        except Exception as e:
            send_response(req_id, error={
                "code": -32000,
                "message": str(e),
                "data": traceback.format_exc(),
            })


if __name__ == "__main__":
    main()
