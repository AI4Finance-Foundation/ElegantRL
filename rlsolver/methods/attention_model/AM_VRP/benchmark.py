"""Benchmark sampling efficiency for VRP solver with different batch sizes."""

import torch
import time
from datetime import datetime
import os
from typing import Dict, List, Tuple

from models import VRPSolver

# Benchmark configuration
BATCH_SIZES = [2**i for i in range(7, 13)]  # [128, 256, 512, 1024, 2048, 4096]
NUM_STEPS = 2000  # Number of steps to measure per batch size
WARMUP_STEPS = 10  # Warmup steps before measurement
SEQ_LEN = 30  # Number of nodes in VRP
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model configuration (from config.py)
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 128
N_HEAD = 4
C = 10.0


def benchmark_sampling_efficiency() -> Tuple[str, Dict]:
    """Run benchmark for different batch sizes and record sampling efficiency."""
    
    # Create output directory
    os.makedirs('benchmark_results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'benchmark_results/sampling_efficiency_{timestamp}.txt'
    
    # Initialize model
    print("Loading model...")
    model = VRPSolver(
        embedding_size=EMBEDDING_SIZE,
        hidden_size=HIDDEN_SIZE,
        seq_len=SEQ_LEN,
        n_head=N_HEAD,
        C=C
    ).to(DEVICE)
    
    # Load pretrained weights if available
    if os.path.exists('model.pth'):
        print("Loading pretrained weights...")
        model.load_state_dict(torch.load('model.pth', map_location=DEVICE))
    
    model.eval()
    
    # Store results for table
    results_table = {}
    
    # Open output file
    with open(output_file, 'w') as f:
        # Write header
        f.write("VRP Sampling Efficiency Benchmark\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Sequence Length: {SEQ_LEN}\n")
        f.write(f"Number of Steps: {NUM_STEPS}\n")
        f.write("="*60 + "\n\n")
        
        # Benchmark each batch size
        for batch_size in BATCH_SIZES:
            print(f"\nBenchmarking batch size: {batch_size}")
            f.write(f"Batch Size: {batch_size}\n")
            f.write("-"*40 + "\n")
            
            # Check memory feasibility
            try:
                # Generate dummy input
                dummy_input = torch.rand(batch_size, SEQ_LEN, 2).to(DEVICE)
                
                # Warmup
                print(f"  Warming up...")
                with torch.no_grad():
                    for _ in range(WARMUP_STEPS):
                        _ = model(dummy_input)
                
                # Clear cache for accurate measurement
                if DEVICE == 'cuda':
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                
                # Measure sampling efficiency
                step_times = []
                step_throughputs = []
                
                print(f"  Measuring {NUM_STEPS} steps...")
                progress_interval = NUM_STEPS // 10  # Print 10 progress updates
                
                with torch.no_grad():
                    for step in range(NUM_STEPS):
                        # Generate new input for each step
                        inputs = torch.rand(batch_size, SEQ_LEN, 2).to(DEVICE)
                        
                        # Synchronize before timing
                        if DEVICE == 'cuda':
                            torch.cuda.synchronize()
                        
                        # Time the forward pass
                        start_time = time.perf_counter()
                        
                        rewards, log_probs, actions = model(inputs)
                        
                        # Synchronize after computation
                        if DEVICE == 'cuda':
                            torch.cuda.synchronize()
                        
                        end_time = time.perf_counter()
                        
                        # Calculate metrics
                        step_time = end_time - start_time
                        samples_per_second = batch_size / step_time
                        
                        step_times.append(step_time)
                        step_throughputs.append(samples_per_second)
                        
                        # Print progress at intervals
                        if (step + 1) % progress_interval == 0:
                            current_avg = sum(step_throughputs[-100:]) / len(step_throughputs[-100:])
                            print(f"    Step {step+1}/{NUM_STEPS} - "
                                  f"Recent avg: {current_avg:.1f} samples/sec")
                
                # Calculate statistics
                avg_time = sum(step_times) / len(step_times)
                avg_throughput = sum(step_throughputs) / len(step_throughputs)
                min_throughput = min(step_throughputs)
                max_throughput = max(step_throughputs)
                std_throughput = torch.std(torch.tensor(step_throughputs)).item()
                
                # Store for table
                results_table[batch_size] = {
                    'avg_throughput': avg_throughput,
                    'min_throughput': min_throughput,
                    'max_throughput': max_throughput,
                    'std_throughput': std_throughput,
                    'avg_time': avg_time
                }
                
                # Write summary statistics
                f.write("-"*40 + "\n")
                f.write(f"Average Throughput: {avg_throughput:10.2f} samples/sec\n")
                f.write(f"Min Throughput:     {min_throughput:10.2f} samples/sec\n")
                f.write(f"Max Throughput:     {max_throughput:10.2f} samples/sec\n")
                f.write(f"Std Deviation:      {std_throughput:10.2f} samples/sec\n")
                f.write(f"Average Time/Step:  {avg_time*1000:10.3f} ms\n")
                f.write(f"Total Samples:      {batch_size * NUM_STEPS:10d}\n")
                f.write("\n")
                
                # Print summary
                print(f"  Average: {avg_throughput:.1f} samples/sec")
                print(f"  Range: [{min_throughput:.1f}, {max_throughput:.1f}] samples/sec")
                
                # Clear memory
                del dummy_input, inputs, rewards, log_probs, actions
                if DEVICE == 'cuda':
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  Batch size {batch_size} too large for available memory")
                    f.write(f"ERROR: Out of memory for batch size {batch_size}\n\n")
                    results_table[batch_size] = None
                    if DEVICE == 'cuda':
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        # Write final summary
        f.write("="*60 + "\n")
        f.write("Benchmark completed successfully\n")
    
    print(f"\nResults saved to: {output_file}")
    return output_file, results_table


def print_results_table(results: Dict):
    """Print a formatted results table."""
    print("\n" + "="*80)
    print("BENCHMARK RESULTS TABLE")
    print("="*80)
    print(f"{'环境数量':<15} {'采样速度 (samples/sec)':<25} {'相对加速比':<15} {'备注':<20}")
    print("-"*80)
    
    # Get baseline throughput (smallest batch size with valid result)
    baseline_throughput = None
    for batch_size in sorted(results.keys()):
        if results[batch_size] is not None:
            baseline_throughput = results[batch_size]['avg_throughput']
            break
    
    # Print each row
    for batch_size in sorted(results.keys()):
        if results[batch_size] is not None:
            avg_throughput = results[batch_size]['avg_throughput']
            min_throughput = results[batch_size]['min_throughput']
            max_throughput = results[batch_size]['max_throughput']
            
            # Calculate speedup
            if baseline_throughput:
                speedup = avg_throughput / baseline_throughput
            else:
                speedup = 1.0
            
            # Format environment count
            env_str = f"{batch_size:,}"
            
            # Format throughput with range
            throughput_str = f"{avg_throughput:,.2f}"
            
            # Format speedup
            speedup_str = f"{speedup:.2f}x"
            
            # Add note about variance if high
            std = results[batch_size]['std_throughput']
            if std / avg_throughput > 0.1:  # More than 10% variance
                note = "高方差"
            else:
                note = "稳定"
            
            print(f"{env_str:<15} {throughput_str:<25} {speedup_str:<15} {note:<20}")
        else:
            env_str = f"{batch_size:,}"
            print(f"{env_str:<15} {'内存不足':<25} {'-':<15} {'跳过':<20}")
    
    print("-"*80)
    
    # Calculate and print scaling efficiency
    valid_results = {k: v for k, v in results.items() if v is not None}
    if len(valid_results) >= 2:
        sizes = sorted(valid_results.keys())
        first_size = sizes[0]
        last_size = sizes[-1]
        
        size_ratio = last_size / first_size
        throughput_ratio = valid_results[last_size]['avg_throughput'] / valid_results[first_size]['avg_throughput']
        scaling_efficiency = (throughput_ratio / size_ratio) * 100
        
        print(f"\n扩展效率分析:")
        print(f"  从 {first_size:,} 到 {last_size:,} 环境")
        print(f"  环境数量增长: {size_ratio:.1f}x")
        print(f"  吞吐量增长: {throughput_ratio:.1f}x")
        print(f"  扩展效率: {scaling_efficiency:.1f}%")
    
    # Print additional statistics
    print(f"\n测试配置:")
    print(f"  设备: {DEVICE}")
    print(f"  序列长度: {SEQ_LEN}")
    print(f"  测试步数: {NUM_STEPS:,}")
    print("="*80)


def save_csv_table(results: Dict, filename: str = None):
    """Save results to CSV file for further analysis."""
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'benchmark_results/results_table_{timestamp}.csv'
    
    with open(filename, 'w') as f:
        # Write header
        f.write("环境数量,平均采样速度(samples/sec),最小速度,最大速度,标准差,平均时间(ms)\n")
        
        # Write data
        for batch_size in sorted(results.keys()):
            if results[batch_size] is not None:
                r = results[batch_size]
                f.write(f"{batch_size},{r['avg_throughput']:.2f},{r['min_throughput']:.2f},"
                       f"{r['max_throughput']:.2f},{r['std_throughput']:.2f},"
                       f"{r['avg_time']*1000:.3f}\n")
            else:
                f.write(f"{batch_size},N/A,N/A,N/A,N/A,N/A\n")
    
    print(f"\nCSV table saved to: {filename}")


if __name__ == '__main__':
    # Run benchmark
    output_file, results = benchmark_sampling_efficiency()
    
    # Print formatted table
    print_results_table(results)
    
    # Save CSV for further analysis
    save_csv_table(results)