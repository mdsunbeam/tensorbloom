"""Test all template architectures by building models and running forward passes."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from tensorbloom_sidecar import _build_model_from_graph

DATASET_SHAPES = {
    "MNIST": (1, 28, 28),
    "CIFAR10": (3, 32, 32),
    "WikiText2": (128,),
    "TinyShakespeare": (256,),
    "AGNews": (100,),
    "IMDB": (100,),
    "SpeechCommands": (80, 100),
    "ImageFolder": (3, 224, 224),
}

def N(id, mt, params=None, **kw):
    return {"id": id, "mt": mt, "params": params or {}, **kw}

def make_graph(nodes_spec, edges_spec):
    nodes = []
    for n in nodes_spec:
        node = {"id": n["id"], "type": n.get("type", "call_module"),
                "moduleType": n.get("mt", ""), "label": n.get("label", n["id"]),
                "params": n.get("params", {})}
        if n.get("mt") == "Input" or n.get("type") == "placeholder":
            node["type"] = "placeholder"
        nodes.append(node)
    edges = []
    for e in edges_spec:
        edges.append({"id": f"e_{e[0]}_{e[1]}", "source": e[0], "target": e[1],
                       "sourceHandle": e[2] if len(e) > 2 else "out",
                       "targetHandle": e[3] if len(e) > 3 else "in"})
    return {"nodes": nodes, "edges": edges}

def ch(*ids):
    """Chain edges."""
    return [(ids[i], ids[i+1], "out", "in") for i in range(len(ids)-1)]

def out_edge(src, loss_id="loss"):
    """Edge from last layer to loss node (uses 'input' handle)."""
    return [(src, loss_id, "out", "input")]

def test(name, nodes, edges, shape, is_text=False, batch=2, vocab=100):
    try:
        model = _build_model_from_graph(make_graph(nodes, edges))
        model.eval()
        x = torch.randint(0, vocab, (batch, *shape)) if is_text else torch.randn(batch, *shape)
        num_inp = sum(1 for n in nodes if n.get("type") == "placeholder")
        with torch.no_grad():
            out = model(*((x,) * max(num_inp, 1)))
        if out is None:
            print(f"  FAIL  {name}: output is None"); return False
        print(f"  PASS  {name}: {list(x.shape)} -> {list(out.shape)}"); return True
    except Exception as e:
        print(f"  FAIL  {name}: {e}"); return False

INP = N("inp", "Input", type="placeholder")
CE = N("loss", "CrossEntropyLoss")
MSE = N("loss", "MSELoss")
BCE = N("loss", "BCEWithLogitsLoss")
results = {}

# 1. Simple MLP
n = [INP, N("fl","Flatten",{"start_dim":1,"end_dim":-1}), N("f1","Linear",{"in_features":784,"out_features":512}),
     N("r1","ReLU"), N("d1","Dropout",{"p":0.2}), N("f2","Linear",{"in_features":512,"out_features":256}),
     N("r2","ReLU"), N("d2","Dropout",{"p":0.2}), N("f3","Linear",{"in_features":256,"out_features":10}), CE]
results["Simple MLP"] = test("Simple MLP", n, ch("inp","fl","f1","r1","d1","f2","r2","d2","f3")+out_edge("f3"), DATASET_SHAPES["MNIST"])

# 2. LeNet-5
n = [INP, N("c1","Conv2d",{"in_channels":1,"out_channels":6,"kernel_size":5,"padding":0,"bias":True}),
     N("t1","Tanh"), N("p1","AvgPool2d",{"kernel_size":2}),
     N("c2","Conv2d",{"in_channels":6,"out_channels":16,"kernel_size":5,"padding":0,"bias":True}),
     N("t2","Tanh"), N("p2","AvgPool2d",{"kernel_size":2}),
     N("fl","Flatten",{"start_dim":1,"end_dim":-1}),
     N("f1","Linear",{"in_features":256,"out_features":120}), N("t3","Tanh"),
     N("f2","Linear",{"in_features":120,"out_features":84}), N("t4","Tanh"),
     N("f3","Linear",{"in_features":84,"out_features":10}), CE]
results["LeNet-5"] = test("LeNet-5", n, ch("inp","c1","t1","p1","c2","t2","p2","fl","f1","t3","f2","t4","f3")+out_edge("f3"), DATASET_SHAPES["MNIST"])

# 3. ResNet-18
n = [INP,
     N("sc","Conv2d",{"in_channels":3,"out_channels":64,"kernel_size":7,"stride":2,"padding":3,"bias":False}),
     N("sbn","BatchNorm2d",{"num_features":64}), N("sr","ReLU"),
     N("sp","MaxPool2d",{"kernel_size":3,"stride":2,"padding":1}),
     N("b1c1","Conv2d",{"in_channels":64,"out_channels":64,"kernel_size":3,"padding":1,"bias":False}),
     N("b1n1","BatchNorm2d",{"num_features":64}), N("b1r1","ReLU"),
     N("b1c2","Conv2d",{"in_channels":64,"out_channels":64,"kernel_size":3,"padding":1,"bias":False}),
     N("b1n2","BatchNorm2d",{"num_features":64}),
     N("b1a","Add"), N("b1r2","ReLU"),
     N("b2c1","Conv2d",{"in_channels":64,"out_channels":128,"kernel_size":3,"stride":2,"padding":1,"bias":False}),
     N("b2n1","BatchNorm2d",{"num_features":128}), N("b2r1","ReLU"),
     N("b2c2","Conv2d",{"in_channels":128,"out_channels":128,"kernel_size":3,"padding":1,"bias":False}),
     N("b2n2","BatchNorm2d",{"num_features":128}),
     N("dc","Conv2d",{"in_channels":64,"out_channels":128,"kernel_size":1,"stride":2,"padding":0,"bias":False}),
     N("dn","BatchNorm2d",{"num_features":128}),
     N("b2a","Add"), N("b2r2","ReLU"),
     N("gap","AdaptiveAvgPool2d",{"output_size":[1,1]}),
     N("fl","Flatten",{"start_dim":1,"end_dim":-1}),
     N("fc","Linear",{"in_features":128,"out_features":10}), CE]
e = (ch("inp","sc","sbn","sr","sp") + ch("sp","b1c1","b1n1","b1r1","b1c2","b1n2")
     + [("b1n2","b1a","out","a"),("sp","b1a","out","b")] + ch("b1a","b1r2")
     + ch("b1r2","b2c1","b2n1","b2r1","b2c2","b2n2") + ch("b1r2","dc","dn")
     + [("b2n2","b2a","out","a"),("dn","b2a","out","b")] + ch("b2a","b2r2","gap","fl","fc") + out_edge("fc"))
results["ResNet-18"] = test("ResNet-18", n, e, DATASET_SHAPES["CIFAR10"])

# 4. ViT
n = [INP, N("pe","Conv2d",{"in_channels":3,"out_channels":384,"kernel_size":16,"stride":16,"padding":0}),
     N("fl","Flatten",{"start_dim":2,"end_dim":-1}), N("pm","Permute",{"dims":"[0, 2, 1]"}),
     N("e1","TransformerEncoderLayer",{"d_model":384,"nhead":6,"dim_feedforward":1536,"dropout":0.1,"batch_first":True,"norm_first":True,"activation":"gelu"}),
     N("e2","TransformerEncoderLayer",{"d_model":384,"nhead":6,"dim_feedforward":1536,"dropout":0.1,"batch_first":True,"norm_first":True,"activation":"gelu"}),
     N("ln","LayerNorm",{"normalized_shape":"[384]"}),
     N("pm2","Permute",{"dims":"[0, 2, 1]"}), N("po","AdaptiveAvgPool1d",{"output_size":1}),
     N("fl2","Flatten",{"start_dim":1,"end_dim":-1}), N("hd","Linear",{"in_features":384,"out_features":10}), CE]
results["ViT"] = test("ViT", n, ch("inp","pe","fl","pm","e1","e2","ln","pm2","po","fl2","hd")+out_edge("hd"), DATASET_SHAPES["CIFAR10"])

# 5. MobileNetV2
n = [INP,
     N("ic","Conv2d",{"in_channels":3,"out_channels":32,"kernel_size":3,"stride":2,"padding":1,"bias":False}),
     N("ibn","BatchNorm2d",{"num_features":32}), N("ir","ReLU6"),
     N("b1e","Conv2d",{"in_channels":32,"out_channels":192,"kernel_size":1,"bias":False}),
     N("b1ebn","BatchNorm2d",{"num_features":192}), N("b1er","ReLU6"),
     N("b1dw","Conv2d",{"in_channels":192,"out_channels":192,"kernel_size":3,"padding":1,"groups":192,"bias":False}),
     N("b1dwbn","BatchNorm2d",{"num_features":192}), N("b1dwr","ReLU6"),
     N("b1p","Conv2d",{"in_channels":192,"out_channels":32,"kernel_size":1,"bias":False}),
     N("b1pbn","BatchNorm2d",{"num_features":32}), N("b1a","Add"),
     N("b2e","Conv2d",{"in_channels":32,"out_channels":192,"kernel_size":1,"bias":False}),
     N("b2ebn","BatchNorm2d",{"num_features":192}), N("b2er","ReLU6"),
     N("b2dw","Conv2d",{"in_channels":192,"out_channels":192,"kernel_size":3,"stride":2,"padding":1,"groups":192,"bias":False}),
     N("b2dwbn","BatchNorm2d",{"num_features":192}), N("b2dwr","ReLU6"),
     N("b2p","Conv2d",{"in_channels":192,"out_channels":64,"kernel_size":1,"bias":False}),
     N("b2pbn","BatchNorm2d",{"num_features":64}),
     N("gap","AdaptiveAvgPool2d",{"output_size":[1,1]}),
     N("fl","Flatten",{"start_dim":1,"end_dim":-1}), N("fc","Linear",{"in_features":64,"out_features":10}), CE]
e = (ch("inp","ic","ibn","ir") + ch("ir","b1e","b1ebn","b1er","b1dw","b1dwbn","b1dwr","b1p","b1pbn")
     + [("b1pbn","b1a","out","a"),("ir","b1a","out","b")]
     + ch("b1a","b2e","b2ebn","b2er","b2dw","b2dwbn","b2dwr","b2p","b2pbn")
     + ch("b2pbn","gap","fl","fc") + out_edge("fc"))
results["MobileNetV2"] = test("MobileNetV2", n, e, DATASET_SHAPES["CIFAR10"])

# 6. U-Net
n = [INP,
     N("e1c","Conv2d",{"in_channels":3,"out_channels":64,"kernel_size":3,"padding":1}), N("e1r","ReLU"),
     N("e1p","MaxPool2d",{"kernel_size":2}),
     N("e2c","Conv2d",{"in_channels":64,"out_channels":128,"kernel_size":3,"padding":1}), N("e2r","ReLU"),
     N("e2p","MaxPool2d",{"kernel_size":2}),
     N("br","Conv2d",{"in_channels":128,"out_channels":256,"kernel_size":3,"padding":1}), N("brr","ReLU"),
     N("u2","ConvTranspose2d",{"in_channels":256,"out_channels":128,"kernel_size":2,"stride":2}), N("ct2","Concatenate",{"dim":1}),
     N("d2c","Conv2d",{"in_channels":256,"out_channels":128,"kernel_size":3,"padding":1}), N("d2r","ReLU"),
     N("u1","ConvTranspose2d",{"in_channels":128,"out_channels":64,"kernel_size":2,"stride":2}), N("ct1","Concatenate",{"dim":1}),
     N("d1c","Conv2d",{"in_channels":128,"out_channels":64,"kernel_size":3,"padding":1}), N("d1r","ReLU"),
     N("oc","Conv2d",{"in_channels":64,"out_channels":1,"kernel_size":1,"padding":0}), BCE]
e = (ch("inp","e1c","e1r") + ch("e1r","e1p","e2c","e2r") + ch("e2r","e2p","br","brr","u2")
     + [("u2","ct2","out","a"),("e2r","ct2","out","b")] + ch("ct2","d2c","d2r","u1")
     + [("u1","ct1","out","a"),("e1r","ct1","out","b")] + ch("ct1","d1c","d1r","oc") + out_edge("oc"))
results["U-Net"] = test("U-Net", n, e, DATASET_SHAPES["ImageFolder"])

# 7. Transformer
n = [INP,
     N("se","Embedding",{"num_embeddings":30000,"embedding_dim":512}),
     N("te","Embedding",{"num_embeddings":30000,"embedding_dim":512}),
     N("tf","Transformer",{"d_model":512,"nhead":8,"num_encoder_layers":6,"num_decoder_layers":6,"dim_feedforward":2048,"dropout":0.1,"batch_first":True}),
     N("hd","Linear",{"in_features":512,"out_features":30000}), CE]
e = [("inp","se","out","in"),("inp","te","out","in"),("se","tf","out","src"),("te","tf","out","tgt"),
     ("tf","hd","out","in")] + out_edge("hd")
results["Transformer"] = test("Transformer", n, e, DATASET_SHAPES["WikiText2"], is_text=True)

# 8. nanoGPT
n = [INP, N("emb","Embedding",{"num_embeddings":65,"embedding_dim":384}), N("dr","Dropout",{"p":0.2}),
     N("b1","TransformerDecoderLayer",{"d_model":384,"nhead":6,"dim_feedforward":1536,"dropout":0.2,"batch_first":True,"norm_first":True,"activation":"gelu"}),
     N("b2","TransformerDecoderLayer",{"d_model":384,"nhead":6,"dim_feedforward":1536,"dropout":0.2,"batch_first":True,"norm_first":True,"activation":"gelu"}),
     N("b3","TransformerDecoderLayer",{"d_model":384,"nhead":6,"dim_feedforward":1536,"dropout":0.2,"batch_first":True,"norm_first":True,"activation":"gelu"}),
     N("ln","LayerNorm",{"normalized_shape":"[384]"}),
     N("hd","Linear",{"in_features":384,"out_features":65}), CE]
results["nanoGPT"] = test("nanoGPT", n, ch("inp","emb","dr","b1","b2","b3","ln","hd")+out_edge("hd"), DATASET_SHAPES["TinyShakespeare"], is_text=True, vocab=65)

# 9. BERT Base (reduced to 4 layers for speed)
n = [INP, N("emb","Embedding",{"num_embeddings":30522,"embedding_dim":768}),
     N("eln","LayerNorm",{"normalized_shape":"[768]"}), N("edr","Dropout",{"p":0.1}),
     N("e1","TransformerEncoderLayer",{"d_model":768,"nhead":12,"dim_feedforward":3072,"dropout":0.1,"batch_first":True,"norm_first":False,"activation":"gelu"}),
     N("e2","TransformerEncoderLayer",{"d_model":768,"nhead":12,"dim_feedforward":3072,"dropout":0.1,"batch_first":True,"norm_first":False,"activation":"gelu"}),
     N("pm","Permute",{"dims":"[0, 2, 1]"}), N("po","AdaptiveAvgPool1d",{"output_size":1}),
     N("fl","Flatten",{"start_dim":1,"end_dim":-1}), N("cls","Linear",{"in_features":768,"out_features":4}), CE]
results["BERT Base"] = test("BERT Base", n, ch("inp","emb","eln","edr","e1","e2","pm","po","fl","cls")+out_edge("cls"), DATASET_SHAPES["AGNews"], is_text=True)

# 10. LSTM Classifier
n = [INP, N("emb","Embedding",{"num_embeddings":25000,"embedding_dim":128}),
     N("lstm","LSTM",{"input_size":128,"hidden_size":256,"num_layers":2,"dropout":0.3,"batch_first":True,"bidirectional":True}),
     N("pm","Permute",{"dims":"[0, 2, 1]"}), N("po","AdaptiveAvgPool1d",{"output_size":1}),
     N("fl","Flatten",{"start_dim":1,"end_dim":-1}), N("dr","Dropout",{"p":0.5}),
     N("cls","Linear",{"in_features":512,"out_features":2}), CE]
results["LSTM Classifier"] = test("LSTM Classifier", n, ch("inp","emb","lstm","pm","po","fl","dr","cls")+out_edge("cls"), DATASET_SHAPES["IMDB"], is_text=True)

# 11. Whisper Encoder
n = [INP, N("c1","Conv1d",{"in_channels":80,"out_channels":384,"kernel_size":3,"padding":1,"bias":True}),
     N("g1","GELU"), N("c2","Conv1d",{"in_channels":384,"out_channels":384,"kernel_size":3,"stride":2,"padding":1,"bias":True}),
     N("g2","GELU"), N("pm","Permute",{"dims":"[0, 2, 1]"}),
     N("e1","TransformerEncoderLayer",{"d_model":384,"nhead":6,"dim_feedforward":1536,"dropout":0.0,"batch_first":True,"norm_first":True,"activation":"gelu"}),
     N("e2","TransformerEncoderLayer",{"d_model":384,"nhead":6,"dim_feedforward":1536,"dropout":0.0,"batch_first":True,"norm_first":True,"activation":"gelu"}),
     N("ln","LayerNorm",{"normalized_shape":"[384]"}),
     N("pm2","Permute",{"dims":"[0, 2, 1]"}), N("po","AdaptiveAvgPool1d",{"output_size":1}),
     N("fl","Flatten",{"start_dim":1,"end_dim":-1}), N("cls","Linear",{"in_features":384,"out_features":35}), CE]
results["Whisper Encoder"] = test("Whisper Encoder", n, ch("inp","c1","g1","c2","g2","pm","e1","e2","ln","pm2","po","fl","cls")+out_edge("cls"), DATASET_SHAPES["SpeechCommands"])

# 12. WaveNet Block
n = [INP, N("ic","Conv1d",{"in_channels":80,"out_channels":64,"kernel_size":1,"bias":True}),
     N("d1c","Conv1d",{"in_channels":64,"out_channels":64,"kernel_size":3,"padding":1,"dilation":1,"bias":True}),
     N("d1t","Tanh"), N("d1x","Conv1d",{"in_channels":64,"out_channels":64,"kernel_size":1,"bias":True}), N("d1a","Add"),
     N("d2c","Conv1d",{"in_channels":64,"out_channels":64,"kernel_size":3,"padding":2,"dilation":2,"bias":True}),
     N("d2t","Tanh"), N("d2x","Conv1d",{"in_channels":64,"out_channels":64,"kernel_size":1,"bias":True}), N("d2a","Add"),
     N("d3c","Conv1d",{"in_channels":64,"out_channels":64,"kernel_size":3,"padding":4,"dilation":4,"bias":True}),
     N("d3t","Tanh"), N("d3x","Conv1d",{"in_channels":64,"out_channels":64,"kernel_size":1,"bias":True}), N("d3a","Add"),
     N("ro","ReLU"), N("po","AdaptiveAvgPool1d",{"output_size":1}),
     N("fl","Flatten",{"start_dim":1,"end_dim":-1}), N("cls","Linear",{"in_features":64,"out_features":35}), CE]
e = (ch("inp","ic","d1c","d1t","d1x") + [("d1x","d1a","out","a"),("ic","d1a","out","b")]
     + ch("d1a","d2c","d2t","d2x") + [("d2x","d2a","out","a"),("d1a","d2a","out","b")]
     + ch("d2a","d3c","d3t","d3x") + [("d3x","d3a","out","a"),("d2a","d3a","out","b")]
     + ch("d3a","ro","po","fl","cls") + out_edge("cls"))
results["WaveNet Block"] = test("WaveNet Block", n, e, DATASET_SHAPES["SpeechCommands"])

# 13. Autoencoder
n = [INP, N("fl","Flatten",{"start_dim":1,"end_dim":-1}),
     N("e1","Linear",{"in_features":784,"out_features":256}), N("er1","ReLU"),
     N("e2","Linear",{"in_features":256,"out_features":64}), N("er2","ReLU"),
     N("bn","Linear",{"in_features":64,"out_features":16}), N("bnr","ReLU"),
     N("d1","Linear",{"in_features":16,"out_features":64}), N("dr1","ReLU"),
     N("d2","Linear",{"in_features":64,"out_features":256}), N("dr2","ReLU"),
     N("d3","Linear",{"in_features":256,"out_features":784}), N("sig","Sigmoid"), MSE]
results["Autoencoder"] = test("Autoencoder", n, ch("inp","fl","e1","er1","e2","er2","bn","bnr","d1","dr1","d2","dr2","d3","sig")+out_edge("sig"), DATASET_SHAPES["MNIST"])

# 14. Conv Autoencoder
n = [INP,
     N("ec1","Conv2d",{"in_channels":1,"out_channels":16,"kernel_size":3,"padding":1}), N("er1","ReLU"),
     N("ep1","MaxPool2d",{"kernel_size":2,"stride":2}),
     N("ec2","Conv2d",{"in_channels":16,"out_channels":32,"kernel_size":3,"padding":1}), N("er2","ReLU"),
     N("ep2","MaxPool2d",{"kernel_size":2,"stride":2}),
     N("du1","ConvTranspose2d",{"in_channels":32,"out_channels":16,"kernel_size":2,"stride":2}), N("dr1","ReLU"),
     N("du2","ConvTranspose2d",{"in_channels":16,"out_channels":1,"kernel_size":2,"stride":2}), N("sig","Sigmoid"), MSE]
results["Conv AE"] = test("Conv AE", n, ch("inp","ec1","er1","ep1","ec2","er2","ep2","du1","dr1","du2","sig")+out_edge("sig"), DATASET_SHAPES["MNIST"])

# 15. Embedding Classifier
n = [INP,
     N("c1","Conv2d",{"in_channels":1,"out_channels":32,"kernel_size":3,"padding":1}), N("r1","ReLU"),
     N("p1","MaxPool2d",{"kernel_size":2,"stride":2}),
     N("c2","Conv2d",{"in_channels":32,"out_channels":64,"kernel_size":3,"padding":1}), N("r2","ReLU"),
     N("p2","MaxPool2d",{"kernel_size":2,"stride":2}),
     N("fl","Flatten",{"start_dim":1,"end_dim":-1}),
     N("em","Linear",{"in_features":3136,"out_features":128}), N("er","ReLU"),
     N("cls","Linear",{"in_features":128,"out_features":10}), CE]
results["Embed Classifier"] = test("Embed Classifier", n, ch("inp","c1","r1","p1","c2","r2","p2","fl","em","er","cls")+out_edge("cls"), DATASET_SHAPES["MNIST"])

# 16. DenseNet Block
n = [INP,
     N("ic","Conv2d",{"in_channels":3,"out_channels":64,"kernel_size":3,"padding":1}),
     N("ibn","BatchNorm2d",{"num_features":64}), N("ir","ReLU"),
     N("d1","Conv2d",{"in_channels":64,"out_channels":32,"kernel_size":3,"padding":1}), N("d1r","ReLU"),
     N("ct1","Concatenate",{"dim":1}),
     N("d2","Conv2d",{"in_channels":96,"out_channels":32,"kernel_size":3,"padding":1}), N("d2r","ReLU"),
     N("ct2","Concatenate",{"dim":1}),
     N("tc","Conv2d",{"in_channels":128,"out_channels":64,"kernel_size":1}), N("tp","AvgPool2d",{"kernel_size":2}),
     N("gap","AdaptiveAvgPool2d",{"output_size":[1,1]}),
     N("fl","Flatten",{"start_dim":1,"end_dim":-1}), N("fc","Linear",{"in_features":64,"out_features":10}), CE]
e = (ch("inp","ic","ibn","ir") + ch("ir","d1","d1r")
     + [("ir","ct1","out","a"),("d1r","ct1","out","b")] + ch("ct1","d2","d2r")
     + [("ct1","ct2","out","a"),("d2r","ct2","out","b")]
     + ch("ct2","tc","tp","gap","fl","fc") + out_edge("fc"))
results["DenseNet Block"] = test("DenseNet Block", n, e, DATASET_SHAPES["CIFAR10"])

# Summary
print("\n" + "="*60)
passed = sum(1 for v in results.values() if v)
failed = sum(1 for v in results.values() if not v)
print(f"Results: {passed} passed, {failed} failed out of {len(results)}")
if failed:
    print("FAILED:")
    for name, ok in results.items():
        if not ok: print(f"  - {name}")
