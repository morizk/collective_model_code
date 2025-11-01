"""
GPU Monitoring Script

Monitors GPU utilization, memory usage, and power consumption during training.
Helps identify bottlenecks and optimize batch sizes for hyperparameter search.

Usage:
    # Monitor for 5 minutes
    python scripts/monitor_gpu.py --duration 300
    
    # Monitor with CSV output
    python scripts/monitor_gpu.py --duration 300 --output gpu_stats.csv
    
    # Monitor with custom interval
    python scripts/monitor_gpu.py --duration 300 --interval 0.5
"""

import argparse
import time
import subprocess
import csv
import sys
from datetime import datetime
from pathlib import Path


def get_gpu_stats():
    """Get GPU statistics using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu', '--format=csv'],
            capture_output=True,
            text=True,
            check=True
        )
        
        lines = result.stdout.strip().split('\n')
        if len(lines) < 2:
            return None
        
        # Parse CSV: header is first line, data is second line
        import csv as csv_module
        reader = csv_module.reader(lines)
        header = next(reader)
        data = next(reader)
        
        stats = {}
        for h, d in zip(header, data):
            key = h.strip()
            value = d.strip()
            stats[key] = value
        
        return stats
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error getting GPU stats: {e}", file=sys.stderr)
        return None


def format_bytes(mem_str):
    """Convert memory string like '1024 MiB' to MB."""
    if 'MiB' in mem_str:
        return float(mem_str.replace(' MiB', ''))
    elif 'GiB' in mem_str:
        return float(mem_str.replace(' GiB', '')) * 1024
    return 0.0


def format_power(power_str):
    """Convert power string like '45.23 W' to float."""
    if 'W' in power_str:
        return float(power_str.replace(' W', ''))
    return 0.0


def monitor_gpu(duration, interval=1.0, output_file=None):
    """
    Monitor GPU for specified duration.
    
    Args:
        duration: Duration in seconds to monitor
        interval: Sampling interval in seconds
        output_file: Optional CSV file to save stats
    """
    print(f"Starting GPU monitoring for {duration} seconds (interval: {interval}s)")
    print("=" * 80)
    
    if output_file:
        csv_file = open(output_file, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'timestamp', 'gpu_index', 'gpu_name', 'utilization_gpu_%',
            'memory_used_mb', 'memory_total_mb', 'memory_usage_%',
            'power_draw_w', 'power_limit_w', 'power_usage_%', 'temperature_c'
        ])
    
    start_time = time.time()
    samples = []
    
    try:
        while time.time() - start_time < duration:
            stats = get_gpu_stats()
            
            if stats:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                gpu_index = stats.get('index', 'N/A')
                gpu_name = stats.get('name', 'N/A')
                util_gpu = stats.get('utilization.gpu [%]', '0').replace(' %', '')
                mem_used_str = stats.get('memory.used [MiB]', '0 MiB')
                mem_total_str = stats.get('memory.total [MiB]', '0 MiB')
                power_draw_str = stats.get('power.draw [W]', '0 W')
                power_limit_str = stats.get('power.limit [W]', '0 W')
                temp = stats.get('temperature.gpu [C]', '0').replace(' C', '')
                
                mem_used = format_bytes(mem_used_str)
                mem_total = format_bytes(mem_total_str)
                mem_pct = (mem_used / mem_total * 100) if mem_total > 0 else 0
                power_draw = format_power(power_draw_str)
                # Handle [N/A] for power limit
                if '[N/A]' in power_limit_str or 'N/A' in power_limit_str:
                    power_limit = 0
                    power_pct = 0
                else:
                    power_limit = format_power(power_limit_str)
                    power_pct = (power_draw / power_limit * 100) if power_limit > 0 else 0
                
                # Parse temperature (remove spaces and units)
                temp_val = temp.replace(' C', '').replace('°C', '').strip()
                try:
                    temp_val = float(temp_val) if temp_val else 0
                except ValueError:
                    temp_val = 0
                
                # Print current stats
                print(f"[{timestamp}] GPU {gpu_index}: {gpu_name}")
                print(f"  Utilization: {util_gpu}%")
                print(f"  Memory: {mem_used:.0f} MB / {mem_total:.0f} MB ({mem_pct:.1f}%)")
                print(f"  Power: {power_draw:.1f} W / {power_limit:.1f} W ({power_pct:.1f}%)")
                print(f"  Temperature: {temp_val}°C")
                print("-" * 80)
                
                # Save to CSV
                if output_file:
                    csv_writer.writerow([
                        timestamp, gpu_index, gpu_name, util_gpu,
                        f"{mem_used:.0f}", f"{mem_total:.0f}", f"{mem_pct:.1f}",
                        f"{power_draw:.1f}", f"{power_limit:.1f}", f"{power_pct:.1f}", f"{temp_val:.0f}"
                    ])
                
                # Store for summary
                samples.append({
                    'util': float(util_gpu) if util_gpu != 'N/A' else 0,
                    'mem_pct': mem_pct,
                    'power_pct': power_pct,
                    'mem_used': mem_used,
                    'power_draw': power_draw
                })
            else:
                print("Warning: Could not get GPU stats")
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
    
    finally:
        if output_file:
            csv_file.close()
            print(f"\nStats saved to: {output_file}")
    
    # Print summary statistics
    if samples:
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        
        avg_util = sum(s['util'] for s in samples) / len(samples)
        max_util = max(s['util'] for s in samples)
        min_util = min(s['util'] for s in samples)
        
        avg_mem_pct = sum(s['mem_pct'] for s in samples) / len(samples)
        max_mem_pct = max(s['mem_pct'] for s in samples)
        max_mem_used = max(s['mem_used'] for s in samples)
        
        avg_power_pct = sum(s['power_pct'] for s in samples) / len(samples)
        max_power_pct = max(s['power_pct'] for s in samples)
        max_power_draw = max(s['power_draw'] for s in samples)
        
        print(f"GPU Utilization:")
        print(f"  Average: {avg_util:.1f}%")
        print(f"  Min: {min_util:.1f}%")
        print(f"  Max: {max_util:.1f}%")
        
        print(f"\nMemory Usage:")
        print(f"  Average: {avg_mem_pct:.1f}%")
        print(f"  Peak: {max_mem_pct:.1f}% ({max_mem_used:.0f} MB)")
        
        print(f"\nPower Usage:")
        print(f"  Average: {avg_power_pct:.1f}%")
        print(f"  Peak: {max_power_pct:.1f}% ({max_power_draw:.1f} W)")
        
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS:")
        print("=" * 80)
        
        if avg_util < 50:
            print("⚠️  GPU utilization is low (<50%). Consider:")
            print("   - Increasing batch size")
            print("   - Using more data augmentation")
            print("   - Checking if training is CPU-bound")
        
        if avg_util > 90:
            print("✓ GPU utilization is high (>90%). Good!")
        
        if max_mem_pct < 70:
            print("⚠️  Memory usage is low (<70%). Consider:")
            print("   - Increasing batch size")
            print("   - Using larger models")
            print("   - Enabling mixed precision training (FP16)")
        
        if max_mem_pct > 95:
            print("⚠️  Memory usage is very high (>95%). Consider:")
            print("   - Reducing batch size")
            print("   - Using gradient accumulation")
            print("   - Reducing model size")
        
        if avg_power_pct < 70:
            print("⚠️  Power usage is low. GPU may not be fully utilized.")
        
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Monitor GPU utilization, memory, and power during training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor for 5 minutes
  python scripts/monitor_gpu.py --duration 300
  
  # Monitor and save to CSV
  python scripts/monitor_gpu.py --duration 300 --output gpu_stats.csv
  
  # Monitor with 0.5s interval
  python scripts/monitor_gpu.py --duration 300 --interval 0.5
        """
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=300,
        help='Duration to monitor in seconds (default: 300)'
    )
    
    parser.add_argument(
        '--interval',
        type=float,
        default=1.0,
        help='Sampling interval in seconds (default: 1.0)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (optional)'
    )
    
    args = parser.parse_args()
    
    # Check if nvidia-smi is available
    try:
        subprocess.run(['nvidia-smi', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: nvidia-smi not found. This script requires NVIDIA GPU and drivers.", file=sys.stderr)
        sys.exit(1)
    
    monitor_gpu(args.duration, args.interval, args.output)


if __name__ == '__main__':
    main()

