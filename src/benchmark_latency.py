"""Inference latency benchmarking for ONNX model on CPU / embedded targets.

Governed by architecture.md §5 and prd.md Success Criterion #5.
Measures wall-clock inference latency (ms/window) over 1,000 iterations.
"""

import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import onnxruntime as ort


def benchmark_onnx_latency(
    onnx_path: str = "experiments/phase3_conditioning_physics/model.onnx",
    n_warmup: int = 100,
    n_runs: int = 1000,
    batch_size: int = 1,
    seq_len: int = 100,
    in_channels: int = 102,
    save_dir: str = "experiments/results",
) -> dict:
    """Benchmark inference latency of an ONNX model."""
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        
    print(f"==================================================")
    print(f"BENCHMARKING INFERENCE LATENCY (ONNXRuntime CPU)")
    print(f"Model: {onnx_path}")
    print(f"Runs: {n_runs} (Warmup: {n_warmup}) | Batch Size: {batch_size}")
    print(f"==================================================")
    
    # Configure ONNXRuntime session with single-threaded deterministic CPU profiling
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
    
    # Create representative dummy inputs
    x_input = np.random.randn(batch_size, seq_len, in_channels).astype(np.float32)
    speed_input = np.array([[0.5]] * batch_size, dtype=np.float32)
    
    input_feed = {
        session.get_inputs()[0].name: x_input,
        session.get_inputs()[1].name: speed_input,
    }
    
    # Warm-up passes
    print(f"Running {n_warmup} warm-up iterations...")
    for _ in range(n_warmup):
        _ = session.run(None, input_feed)
        
    # High-precision benchmark passes
    print(f"Profiling {n_runs} benchmark iterations...")
    latencies_ns = []
    for _ in range(n_runs):
        t0 = time.perf_counter_ns()
        _ = session.run(None, input_feed)
        t1 = time.perf_counter_ns()
        latencies_ns.append(t1 - t0)
        
    latencies_ms = np.array(latencies_ns, dtype=np.float64) / 1e6
    
    mean_lat = float(np.mean(latencies_ms))
    median_lat = float(np.median(latencies_ms))
    p95_lat = float(np.percentile(latencies_ms, 95))
    p99_lat = float(np.percentile(latencies_ms, 99))
    min_lat = float(np.min(latencies_ms))
    max_lat = float(np.max(latencies_ms))
    std_lat = float(np.std(latencies_ms))
    throughput = float(1000.0 / mean_lat * batch_size)
    
    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    
    results = {
        "model_path": onnx_path,
        "device": "CPU",
        "latency_estimate": True,  # Per architecture.md §5: host CPU profiled estimate
        "batch_size": batch_size,
        "n_warmup": n_warmup,
        "n_runs": n_runs,
        "file_size_mb": file_size_mb,
        "mean_latency_ms": mean_lat,
        "median_latency_ms": median_lat,
        "p95_latency_ms": p95_lat,
        "p99_latency_ms": p99_lat,
        "min_latency_ms": min_lat,
        "max_latency_ms": max_lat,
        "std_latency_ms": std_lat,
        "throughput_windows_per_second": throughput,
    }
    
    print("\n--- Benchmark Results ---")
    print(f"  Model Size:          {file_size_mb:.2f} MB")
    print(f"  Mean Latency:        {mean_lat:.3f} ms / window")
    print(f"  Median Latency:      {median_lat:.3f} ms / window")
    print(f"  95th Percentile:     {p95_lat:.3f} ms / window")
    print(f"  99th Percentile:     {p99_lat:.3f} ms / window")
    print(f"  Throughput:          {throughput:.1f} windows / sec")
    print(f"  Profile Status:      CPU-Profiled Estimate (latency_estimate=True)")
    print("==================================================")
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_file = os.path.join(save_dir, "latency_benchmark.json")
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Latency results saved to {out_file}")
        
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="experiments/phase3_conditioning_physics/model.onnx")
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="experiments/results")
    args = parser.parse_args()
    
    benchmark_onnx_latency(
        onnx_path=args.onnx,
        n_warmup=args.warmup,
        n_runs=args.runs,
        save_dir=args.save_dir,
    )
