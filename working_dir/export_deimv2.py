import argparse
import re
from copy import deepcopy
from pathlib import Path

from ultralytics import RTDETRDEIM
from ultralytics.engine.exporter import Exporter


def parse_args():
    p = argparse.ArgumentParser(description="Export RTDETRDEIM with upstream deploy conversion.")
    p.add_argument("weights", type=str, help="Path to RTDETRDEIM .pt checkpoint.")
    p.add_argument("--format", type=str, default="onnx", help="Export format.")
    p.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset.")
    p.add_argument("--device", default=0, help="CUDA device id or 'cpu'.")
    p.add_argument("--batch", type=int, default=1, help="Batch size.")
    p.add_argument("--half", action="store_true", help="FP16 export.")
    p.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify ONNX graph. Upstream DEIMv2 keeps this disabled by default.",
    )
    p.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional experiment stem. Defaults to '<weights>_op{opset}_{sim|nosim}'.",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to the weights directory.",
    )
    p.add_argument("--workspace", type=float, default=None, help="TensorRT workspace size in GB.")
    p.add_argument(
        "--no-fp32-attn",
        action="store_true",
        help="Disable DINOv3-safe fp32 attention pinning for TRT fp16 builds. "
        "Without this flag, attention MatMul/Softmax and norm internals are "
        "forced to fp32 to avoid overflow (NVIDIA/TensorRT#4723).",
    )
    p.set_defaults(simplify=False)
    return p.parse_args()


def build_engine_fp16(onnx_path, engine_path, workspace=None, half=True, shape=(1, 3, 640, 640), fp32_attn=True):
    """Build a TensorRT fp16 engine with DINOv3-safe precision overrides.

    DINOv3 ViT backbones use decomposed self-attention (MatMul → Softmax) in
    TRT 10.x.  The QK^T MatMul output materialises as fp16 and overflows for
    attention logit spreads > ~11 (d_k=64), producing NaN that propagates
    through the entire decoder.  See NVIDIA/TensorRT#4723.

    When ``fp32_attn=True`` (default) this builder pins every Softmax, every
    attention-path MatMul, and every norm-internal Reduce/Pow/Unary/Elementwise
    to fp32 while keeping the rest of the graph in fp16.
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    ws = int((workspace or 4) * (1 << 30))
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ws)

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            print("PARSE ERR:", parser.get_error(i))
        raise RuntimeError(f"Failed to parse ONNX: {onnx_path}")

    if half and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    if half and fp32_attn:
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        # Match actual TensorRT layer names parsed from the DEIMv2 ONNX:
        # attention is decomposed into MatMul/Softmax, while some decoder norms
        # are still lowered into Reduce/Pow/Unary/Elementwise subgraphs.
        attn_re = re.compile(r"/attn/|/self_attn/|/cross_attn/")
        norm_re = re.compile(r"/(?:norm\d*|gateway/norm)(?:/|$)")
        norm_internal_types = {trt.LayerType.ELEMENTWISE, trt.LayerType.REDUCE, trt.LayerType.UNARY}
        n_pinned = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            name = layer.name or ""
            pin = False
            if layer.type == trt.LayerType.SOFTMAX:
                pin = True
            elif layer.type == trt.LayerType.NORMALIZATION:
                pin = True
            # elif layer.type == trt.LayerType.MATRIX_MULTIPLY and attn_re.search(name):
            #     pin = True
            elif norm_re.search(name) and layer.type in norm_internal_types:
                pin = True
            if pin:
                layer.precision = trt.float32
                for o in range(layer.num_outputs):
                    layer.set_output_type(o, trt.float32)
                n_pinned += 1
        print(f"DINOv3-safe: pinned {n_pinned} attention/norm layers to fp32")

    print(f"Building {'FP16' if half else 'FP32'} engine → {engine_path}")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed — check logs above")

    with open(engine_path, "wb") as f:
        f.write(bytes(serialized))
    print(f"Saved {engine_path}")


def build_output_paths(args):
    """Build experiment-specific output paths to avoid clobbering previous exports."""
    weights_path = Path(args.weights)
    outdir = Path(args.outdir) if args.outdir else weights_path.parent
    stem = Path(args.name).stem if args.name else f"{weights_path.stem}_op{args.opset}_{'sim' if args.simplify else 'nosim'}"

    # Keep the intermediate ONNX explicitly marked as FP32 for engine builds, since FP16 should be applied by TRT.
    onnx_precision = "fp16" if args.format == "onnx" and args.half else "fp32"
    engine_precision = "fp16" if args.half else "fp32"

    onnx_path = outdir / f"{stem}_{onnx_precision}.onnx"
    engine_path = outdir / f"{stem}_{engine_precision}.engine"
    return onnx_path, engine_path


def main():
    args = parse_args()
    onnx_path, engine_path = build_output_paths(args)

    # Deploy conversion is destructive (trims DFINE decoder layers, swaps weighting_function);
    # operate on a copy so the wrapper's live model is preserved.
    deploy_model = deepcopy(RTDETRDEIM(args.weights).model).eval().float()
    deploy_model.pt_path = str(onnx_path.with_suffix(".pt"))
    for p in deploy_model.parameters():
        p.requires_grad = False
    for m in deploy_model.modules():
        if hasattr(m, "convert_to_deploy"):
            m.convert_to_deploy()

    export_format = "onnx" if args.format == "engine" else args.format
    print(
        f"Exporting with format={export_format}, imgsz={args.imgsz}, opset={args.opset}, "
        f"device={args.device}, batch={args.batch}, half={args.half}, simplify={args.simplify}"
    )
    print(f"Intermediate ONNX: {onnx_path}")
    if args.format == "engine":
        print(f"Target engine: {engine_path}")
    exporter = Exporter(overrides={
        "format": export_format,
        "imgsz": args.imgsz,
        "opset": args.opset,
        "device": args.device,
        "batch": args.batch,
        "half": args.half if args.format != "engine" else False,
        "simplify": args.simplify,
    })
    artifact = Path(str(exporter(model=deploy_model)))

    if args.format == "engine":
        if artifact.suffix != ".onnx":
            raise RuntimeError(f"Expected ONNX export before TensorRT build, got: {artifact}")
        build_engine_fp16(
            onnx_path=artifact,
            engine_path=engine_path,
            workspace=args.workspace,
            half=args.half,
            shape=(args.batch, 3, args.imgsz, args.imgsz),
            fp32_attn=not args.no_fp32_attn,
        )
        artifact = engine_path

    print(artifact)


if __name__ == "__main__":
    main()
