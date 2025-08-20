#!/usr/bin/env python3
"""
AWS S3 Upload Performance Analysis and Plotting Script
Extracts performance data from benchmark log files and creates plots.
"""

import os
import re
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict

def parse_log_file(file_path):
    """
    Parse a single log file and extract performance metrics for both upload and download.
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract configuration info from filename
        filename = os.path.basename(file_path)
        # Format: benchmark_Xnodes_Ytasks_SIZE.log
        match = re.match(r'benchmark_(\d+)nodes_(\d+)tasks_(.+)\.log', filename)
        if not match:
            return None
        
        nodes = int(match.group(1))
        tasks_per_node = int(match.group(2))
        transfer_size = match.group(3)
        
        # Try to extract upload statistics (new format)
        upload_stats_match = re.search(
            r'FINAL UPLOAD STATISTICS ACROSS ALL TRIALS.*?'
            r'Successful upload trials: (\d+)/(\d+).*?'
            r'Upload bandwidth per rank:\s*'
            r'Mean: ([\d.]+) MB/s\s*'
            r'Std Dev: ([\d.]+) MB/s.*?'
            r'Aggregate upload throughput: ([\d.]+) MB/s',
            content, re.DOTALL
        )
        
        # Try to extract download statistics (new format)
        download_stats_match = re.search(
            r'FINAL DOWNLOAD STATISTICS ACROSS ALL TRIALS.*?'
            r'Successful download trials: (\d+)/(\d+).*?'
            r'Download bandwidth per rank:\s*'
            r'Mean: ([\d.]+) MB/s\s*'
            r'Std Dev: ([\d.]+) MB/s.*?'
            r'Aggregate download throughput: ([\d.]+) MB/s',
            content, re.DOTALL
        )
        
        # Fallback: Try to extract old format statistics
        old_format_match = re.search(
            r'FINAL STATISTICS ACROSS ALL TRIALS.*?'
            r'Bandwidth per rank:\s*'
            r'Mean: ([\d.]+) MB/s\s*'
            r'Std Dev: ([\d.]+) MB/s.*?'
            r'Aggregate throughput: ([\d.]+) MB/s',
            content, re.DOTALL
        )
        
        # Extract file size (default to 1000 MB if not found)
        file_size_match = re.search(r'File size: (\d+) MB', content)
        file_size_mb = int(file_size_match.group(1)) if file_size_match else 1000
        
        # Calculate operations per upload based on transfer size and file size
        operations_per_upload = calculate_operation_count(transfer_size, file_size_mb)
        # Calculate total operations across all tasks and nodes
        total_operations = tasks_per_node * nodes * operations_per_upload
        
        # Initialize result structure
        result = {
            'nodes': nodes,
            'tasks_per_node': tasks_per_node,
            'transfer_size': transfer_size,
            'file_path': file_path,
            'file_size_mb': file_size_mb,
            'operations_per_upload': operations_per_upload,
            'total_operations': total_operations,
            'has_upload_stats': False,
            'has_download_stats': False,
            'has_old_format': False
        }
        
        # Parse upload statistics if available
        if upload_stats_match:
            result.update({
                'has_upload_stats': True,
                'upload_successful_trials': int(upload_stats_match.group(1)),
                'upload_total_trials': int(upload_stats_match.group(2)),
                'upload_mean_bandwidth': float(upload_stats_match.group(3)),
                'upload_std_bandwidth': float(upload_stats_match.group(4)),
                'upload_aggregate_throughput': float(upload_stats_match.group(5))
            })
        
        # Parse download statistics if available
        if download_stats_match:
            result.update({
                'has_download_stats': True,
                'download_successful_trials': int(download_stats_match.group(1)),
                'download_total_trials': int(download_stats_match.group(2)),
                'download_mean_bandwidth': float(download_stats_match.group(3)),
                'download_std_bandwidth': float(download_stats_match.group(4)),
                'download_aggregate_throughput': float(download_stats_match.group(5))
            })
        
        # Parse old format if no new format found
        if not upload_stats_match and not download_stats_match and old_format_match:
            # Extract trial success info for old format
            success_match = re.search(r'Successful trials: (\d+)/(\d+)', content)
            successful_trials = int(success_match.group(1)) if success_match else 0
            total_trials = int(success_match.group(2)) if success_match else 3
            
            result.update({
                'has_old_format': True,
                'mean_bandwidth': float(old_format_match.group(1)),
                'std_bandwidth': float(old_format_match.group(2)),
                'aggregate_throughput': float(old_format_match.group(3)),
                'successful_trials': successful_trials,
                'total_trials': total_trials
            })
        
        # Check if we got any valid data
        if not (result['has_upload_stats'] or result['has_download_stats'] or result['has_old_format']):
            print(f"Warning: Could not parse any statistics from {file_path}")
            return None
        
        return result
        
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def calculate_operation_count(transfer_size, file_size_mb):
    """
    Calculate the number of operations (parts) for upload based on transfer size and file size.
    """
    file_size_bytes = file_size_mb * 1024 * 1024  # Convert MB to bytes
    
    if transfer_size == 'DEFAULT':
        # DEFAULT uses AWS SDK default (single operation)
        return 1
    elif transfer_size == '1M':
        # 1M is less than 5MB AWS multipart minimum, so single operation
        return 1
    elif transfer_size == '50M':
        # 50MB chunks
        chunk_size_bytes = 50 * 1024 * 1024
        return max(1, (file_size_bytes + chunk_size_bytes - 1) // chunk_size_bytes)
    elif transfer_size == '100M':
        # 100MB chunks
        chunk_size_bytes = 100 * 1024 * 1024
        return max(1, (file_size_bytes + chunk_size_bytes - 1) // chunk_size_bytes)
    else:
        # Try to parse size (e.g., "4K", "8K", etc.)
        size_match = re.match(r'(\d+)([KMG])', transfer_size.upper())
        if size_match:
            size_value = int(size_match.group(1))
            size_unit = size_match.group(2)
            
            multiplier = {'K': 1024, 'M': 1024*1024, 'G': 1024*1024*1024}
            chunk_size_bytes = size_value * multiplier.get(size_unit, 1)
            
            # If chunk size < 5MB, use single operation (AWS multipart minimum)
            if chunk_size_bytes < 5 * 1024 * 1024:
                return 1
            else:
                return max(1, (file_size_bytes + chunk_size_bytes - 1) // chunk_size_bytes)
        else:
            # Unknown format, assume single operation
            return 1

def collect_all_data(base_directory="."):
    """
    Collect performance data from all log files in the directory structure.
    """
    all_data = []
    
    # Find all benchmark log files
    for node_dir in ["results_1nodes", "results_2nodes", "results_4nodes", "results_8nodes"]:
        log_pattern = os.path.join(base_directory, node_dir, "benchmark_*.log")
        log_files = glob.glob(log_pattern)
        
        print(f"Found {len(log_files)} log files in {node_dir}")
        
        for log_file in log_files:
            result = parse_log_file(log_file)
            if result:
                all_data.append(result)
            else:
                print(f"Failed to parse: {log_file}")
    
    return all_data

def create_upload_performance_plots(data, show_trendlines=False):
    """
    Create separate upload performance plots for each node configuration.
    
    Args:
        data: Performance data list
        show_trendlines: Boolean to show/hide trendlines (default: False)
    """
    # Filter data for upload statistics only
    upload_data = [entry for entry in data if entry.get('has_upload_stats', False)]
    
    if not upload_data:
        print("No upload data found")
        return 0
    
    # Organize data by nodes
    data_by_nodes = defaultdict(list)
    for entry in upload_data:
        data_by_nodes[entry['nodes']].append(entry)
    
    # Define transfer size order and colors
    transfer_sizes = ['1M', '50M', '100M', 'DEFAULT']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    markers = ['o', 's', '^', 'D']  # Circle, Square, Triangle, Diamond
    
    node_configs = sorted(data_by_nodes.keys())
    
    # Create separate upload plots for each node configuration
    for nodes in node_configs:
        plt.figure(figsize=(10, 8))
        
        node_data = data_by_nodes[nodes]
        
        # Organize data by transfer size
        size_data = defaultdict(lambda: {'tasks': [], 'bandwidth': [], 'std': []})
        
        for entry in node_data:
            transfer_size = entry['transfer_size']
            if entry.get('upload_successful_trials', 0) > 0:  # Only include successful trials
                size_data[transfer_size]['tasks'].append(entry['tasks_per_node'])
                size_data[transfer_size]['bandwidth'].append(entry['upload_mean_bandwidth'])
                size_data[transfer_size]['std'].append(entry['upload_std_bandwidth'])
        
        # Plot each transfer size
        for i, transfer_size in enumerate(transfer_sizes):
            if transfer_size in size_data and size_data[transfer_size]['tasks']:
                tasks = np.array(size_data[transfer_size]['tasks'])
                bandwidth = np.array(size_data[transfer_size]['bandwidth'])
                std_dev = np.array(size_data[transfer_size]['std'])
                
                # Sort by tasks per node
                sort_idx = np.argsort(tasks)
                tasks = tasks[sort_idx]
                bandwidth = bandwidth[sort_idx]
                std_dev = std_dev[sort_idx]
                
                # Plot points with error bars
                linestyle = '-' if show_trendlines else 'None'
                plt.errorbar(tasks, bandwidth, yerr=std_dev, 
                           color=colors[i], marker=markers[i], markersize=10,
                           label=f'{transfer_size}', linewidth=2.5, capsize=6,
                           linestyle=linestyle, alpha=0.8, markerfacecolor=colors[i],
                           markeredgecolor='black', markeredgewidth=0.5)
        
        # Customize plot
        plt.xlabel('Tasks per Node (log scale)', fontsize=14, fontweight='bold')
        plt.ylabel('Upload Bandwidth per Rank (MB/s)', fontsize=14, fontweight='bold')
        plt.title(f'AWS S3 Upload Performance - {nodes} Node{"s" if nodes > 1 else ""}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        
        # Set x-axis to log scale
        plt.xscale('log')
        
        # Set x-axis ticks
        if size_data:
            all_tasks = []
            for size_info in size_data.values():
                all_tasks.extend(size_info['tasks'])
            if all_tasks:
                unique_tasks = sorted(set(all_tasks))
                plt.xticks(unique_tasks, [str(x) for x in unique_tasks], fontsize=12)
        
        plt.yticks(fontsize=12)
        
        # Add stats to the plot
        total_configs = len(node_data)
        successful_configs = sum(1 for entry in node_data if entry.get('upload_successful_trials', 0) > 0)
        
        # Find best upload performance
        valid_entries = [e for e in node_data if e.get('upload_successful_trials', 0) > 0]
        best_entry = max(valid_entries, key=lambda x: x['upload_mean_bandwidth'], default=None) if valid_entries else None
        
        info_text = f'Upload Configs: {successful_configs}/{total_configs} successful'
        if best_entry:
            info_text += f'\nBest Upload: {best_entry["upload_mean_bandwidth"]:.2f} MB/s '
            info_text += f'({best_entry["tasks_per_node"]} tasks/node, {best_entry["transfer_size"]})'
        
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save upload plot
        trendline_suffix = "_with_trendlines" if show_trendlines else ""
        output_file = f'aws_s3_upload_performance_{nodes}nodes{trendline_suffix}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Upload performance plot saved as: {output_file}")
        
        plt.show()
    
    return len(node_configs)

def create_download_performance_plots(data, show_trendlines=False):
    """
    Create separate download performance plots for each node configuration.
    
    Args:
        data: Performance data list
        show_trendlines: Boolean to show/hide trendlines (default: False)
    """
    # Filter data for download statistics only
    download_data = [entry for entry in data if entry.get('has_download_stats', False)]
    
    if not download_data:
        print("No download data found")
        return 0
    
    # Organize data by nodes
    data_by_nodes = defaultdict(list)
    for entry in download_data:
        data_by_nodes[entry['nodes']].append(entry)
    
    # Define transfer size order and colors
    transfer_sizes = ['1M', '50M', '100M', 'DEFAULT']
    colors = ['#e74c3c', '#9b59b6', '#f39c12', '#27ae60']  # Red, Purple, Orange, Green (different from upload)
    markers = ['s', '^', 'D', 'v']  # Square, Triangle up, Diamond, Triangle down (different from upload)
    
    node_configs = sorted(data_by_nodes.keys())
    
    # Create separate download plots for each node configuration
    for nodes in node_configs:
        plt.figure(figsize=(10, 8))
        
        node_data = data_by_nodes[nodes]
        
        # Organize data by transfer size
        size_data = defaultdict(lambda: {'tasks': [], 'bandwidth': [], 'std': []})
        
        for entry in node_data:
            transfer_size = entry['transfer_size']
            if entry.get('download_successful_trials', 0) > 0:  # Only include successful trials
                size_data[transfer_size]['tasks'].append(entry['tasks_per_node'])
                size_data[transfer_size]['bandwidth'].append(entry['download_mean_bandwidth'])
                size_data[transfer_size]['std'].append(entry['download_std_bandwidth'])
        
        # Plot each transfer size
        for i, transfer_size in enumerate(transfer_sizes):
            if transfer_size in size_data and size_data[transfer_size]['tasks']:
                tasks = np.array(size_data[transfer_size]['tasks'])
                bandwidth = np.array(size_data[transfer_size]['bandwidth'])
                std_dev = np.array(size_data[transfer_size]['std'])
                
                # Sort by tasks per node
                sort_idx = np.argsort(tasks)
                tasks = tasks[sort_idx]
                bandwidth = bandwidth[sort_idx]
                std_dev = std_dev[sort_idx]
                
                # Plot points with error bars
                linestyle = '-' if show_trendlines else 'None'
                plt.errorbar(tasks, bandwidth, yerr=std_dev, 
                           color=colors[i], marker=markers[i], markersize=10,
                           label=f'{transfer_size}', linewidth=2.5, capsize=6,
                           linestyle=linestyle, alpha=0.8, markerfacecolor=colors[i],
                           markeredgecolor='black', markeredgewidth=0.5)
        
        # Customize plot
        plt.xlabel('Tasks per Node (log scale)', fontsize=14, fontweight='bold')
        plt.ylabel('Download Bandwidth per Rank (MB/s)', fontsize=14, fontweight='bold')
        plt.title(f'AWS S3 Download Performance - {nodes} Node{"s" if nodes > 1 else ""}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        
        # Set x-axis to log scale
        plt.xscale('log')
        
        # Set x-axis ticks
        if size_data:
            all_tasks = []
            for size_info in size_data.values():
                all_tasks.extend(size_info['tasks'])
            if all_tasks:
                unique_tasks = sorted(set(all_tasks))
                plt.xticks(unique_tasks, [str(x) for x in unique_tasks], fontsize=12)
        
        plt.yticks(fontsize=12)
        
        # Add stats to the plot
        total_configs = len(node_data)
        successful_configs = sum(1 for entry in node_data if entry.get('download_successful_trials', 0) > 0)
        
        # Find best download performance
        valid_entries = [e for e in node_data if e.get('download_successful_trials', 0) > 0]
        best_entry = max(valid_entries, key=lambda x: x['download_mean_bandwidth'], default=None) if valid_entries else None
        
        info_text = f'Download Configs: {successful_configs}/{total_configs} successful'
        if best_entry:
            info_text += f'\nBest Download: {best_entry["download_mean_bandwidth"]:.2f} MB/s '
            info_text += f'({best_entry["tasks_per_node"]} tasks/node, {best_entry["transfer_size"]})'
        
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8))
        
        plt.tight_layout()
        
        # Save download plot
        trendline_suffix = "_with_trendlines" if show_trendlines else ""
        output_file = f'aws_s3_download_performance_{nodes}nodes{trendline_suffix}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Download performance plot saved as: {output_file}")
        
        plt.show()
    
    return len(node_configs)

def create_old_format_performance_plots(data, show_trendlines=False):
    """
    Create performance plots for old format data (backwards compatibility).
    
    Args:
        data: Performance data list
        show_trendlines: Boolean to show/hide trendlines (default: False)
    """
    # Filter data for old format only
    old_data = [entry for entry in data if entry.get('has_old_format', False)]
    
    if not old_data:
        print("No old format data found")
        return 0
    
    # Organize data by nodes
    data_by_nodes = defaultdict(list)
    for entry in old_data:
        data_by_nodes[entry['nodes']].append(entry)
    
    # Define transfer size order and colors
    transfer_sizes = ['1M', '50M', '100M', 'DEFAULT']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    markers = ['o', 's', '^', 'D']  # Circle, Square, Triangle, Diamond
    
    node_configs = sorted(data_by_nodes.keys())
    
    # Create separate plots for each node configuration
    for nodes in node_configs:
        plt.figure(figsize=(10, 8))
        
        node_data = data_by_nodes[nodes]
        
        # Organize data by transfer size
        size_data = defaultdict(lambda: {'tasks': [], 'bandwidth': [], 'std': []})
        
        for entry in node_data:
            transfer_size = entry['transfer_size']
            if entry.get('successful_trials', 0) > 0:  # Only include successful trials
                size_data[transfer_size]['tasks'].append(entry['tasks_per_node'])
                size_data[transfer_size]['bandwidth'].append(entry['mean_bandwidth'])
                size_data[transfer_size]['std'].append(entry['std_bandwidth'])
        
        # Plot each transfer size
        for i, transfer_size in enumerate(transfer_sizes):
            if transfer_size in size_data and size_data[transfer_size]['tasks']:
                tasks = np.array(size_data[transfer_size]['tasks'])
                bandwidth = np.array(size_data[transfer_size]['bandwidth'])
                std_dev = np.array(size_data[transfer_size]['std'])
                
                # Sort by tasks per node
                sort_idx = np.argsort(tasks)
                tasks = tasks[sort_idx]
                bandwidth = bandwidth[sort_idx]
                std_dev = std_dev[sort_idx]
                
                # Plot points with error bars
                linestyle = '-' if show_trendlines else 'None'
                plt.errorbar(tasks, bandwidth, yerr=std_dev, 
                           color=colors[i], marker=markers[i], markersize=10,
                           label=f'{transfer_size}', linewidth=2.5, capsize=6,
                           linestyle=linestyle, alpha=0.8, markerfacecolor=colors[i],
                           markeredgecolor='black', markeredgewidth=0.5)
        
        # Customize plot
        plt.xlabel('Tasks per Node (log scale)', fontsize=14, fontweight='bold')
        plt.ylabel('Bandwidth per Rank (MB/s)', fontsize=14, fontweight='bold')
        plt.title(f'AWS S3 Performance - {nodes} Node{"s" if nodes > 1 else ""}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        
        # Set x-axis to log scale
        plt.xscale('log')
        
        # Set x-axis ticks
        if size_data:
            all_tasks = []
            for size_info in size_data.values():
                all_tasks.extend(size_info['tasks'])
            if all_tasks:
                unique_tasks = sorted(set(all_tasks))
                plt.xticks(unique_tasks, [str(x) for x in unique_tasks], fontsize=12)
        
        plt.yticks(fontsize=12)
        
        # Add stats to the plot
        total_configs = len(node_data)
        successful_configs = sum(1 for entry in node_data if entry.get('successful_trials', 0) > 0)
        
        # Find best performance
        valid_entries = [e for e in node_data if e.get('successful_trials', 0) > 0]
        best_entry = max(valid_entries, key=lambda x: x['mean_bandwidth'], default=None) if valid_entries else None
        
        info_text = f'Configurations: {successful_configs}/{total_configs} successful'
        if best_entry:
            info_text += f'\nBest: {best_entry["mean_bandwidth"]:.2f} MB/s '
            info_text += f'({best_entry["tasks_per_node"]} tasks/node, {best_entry["transfer_size"]})'
        
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        trendline_suffix = "_with_trendlines" if show_trendlines else ""
        output_file = f'aws_s3_performance_{nodes}nodes{trendline_suffix}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Performance plot (old format) saved as: {output_file}")
        
        plt.show()
    
    return len(node_configs)
    """
    Create separate operation count plots for each node configuration.
    
    Args:
        data: Performance data list
        show_trendlines: Boolean to show/hide trendlines (default: False)
    """
    # Organize data by nodes
    data_by_nodes = defaultdict(list)
    for entry in data:
        data_by_nodes[entry['nodes']].append(entry)
    
    # Define transfer size order and colors
    transfer_sizes = ['1M', '50M', '100M', 'DEFAULT']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    markers = ['o', 's', '^', 'D']  # Circle, Square, Triangle, Diamond
    
    node_configs = sorted(data_by_nodes.keys())
    
    # Create separate operation count plots for each node configuration
    for nodes in node_configs:
        plt.figure(figsize=(10, 8))
        
        node_data = data_by_nodes[nodes]
        
        # Organize data by transfer size
        size_data = defaultdict(lambda: {'tasks': [], 'operation_count': []})
        
        for entry in node_data:
            transfer_size = entry['transfer_size']
            if entry['successful_trials'] > 0:  # Only include successful trials
                size_data[transfer_size]['tasks'].append(entry['tasks_per_node'])
                size_data[transfer_size]['operation_count'].append(entry['total_operations'])
        
        # Plot each transfer size
        for i, transfer_size in enumerate(transfer_sizes):
            if transfer_size in size_data and size_data[transfer_size]['tasks']:
                tasks = np.array(size_data[transfer_size]['tasks'])
                operation_count = np.array(size_data[transfer_size]['operation_count'])
                
                # Sort by tasks per node
                sort_idx = np.argsort(tasks)
                tasks = tasks[sort_idx]
                operation_count = operation_count[sort_idx]
                
                # Plot points (no connecting lines by default)
                linestyle = '-' if show_trendlines else 'None'
                plt.plot(tasks, operation_count, 
                        color=colors[i], marker=markers[i], markersize=10,
                        label=f'{transfer_size}', linewidth=2.5,
                        linestyle=linestyle, alpha=0.8, markerfacecolor=colors[i],
                        markeredgecolor='black', markeredgewidth=0.5)
        
        # Customize plot with log scale for x-axis
        plt.xlabel('Tasks per Node (log scale)', fontsize=14, fontweight='bold')
        plt.ylabel('Total Operations (All Tasks & Nodes)', fontsize=14, fontweight='bold')
        plt.title(f'AWS S3 Upload Total Operations - {nodes} Node{"s" if nodes > 1 else ""}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        
        # Set x-axis to log scale
        plt.xscale('log')
        
        # Set x-axis ticks to show actual values
        if size_data:
            all_tasks = []
            for size_info in size_data.values():
                all_tasks.extend(size_info['tasks'])
            if all_tasks:
                unique_tasks = sorted(set(all_tasks))
                plt.xticks(unique_tasks, [str(x) for x in unique_tasks], fontsize=12)
        
        plt.yticks(fontsize=12)
        
        # Add some stats to the plot
        total_configs = len(node_data)
        successful_configs = sum(1 for entry in node_data if entry['successful_trials'] > 0)
        
        # Find configuration info for this node configuration
        file_sizes = set(entry['file_size_mb'] for entry in node_data)
        file_size_text = f"File size: {list(file_sizes)[0]} MB" if len(file_sizes) == 1 else f"File sizes: {sorted(file_sizes)} MB"
        
        info_text = f'Configurations: {successful_configs}/{total_configs} successful\n{file_size_text}'
        
        # Add operation details showing operations per upload for each transfer size
        operation_details = []
        for transfer_size in transfer_sizes:
            if transfer_size in size_data and size_data[transfer_size]['operation_count']:
                # Get operations per upload (divide total by tasks*nodes)
                total_ops = size_data[transfer_size]['operation_count'][0]
                tasks = size_data[transfer_size]['tasks'][0]
                ops_per_upload = total_ops // (tasks * nodes) if tasks > 0 else 0
                operation_details.append(f'{transfer_size}: {ops_per_upload} ops/upload')
        
        if operation_details:
            info_text += f'\nOps per upload: {", ".join(operation_details)}'
        
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        
        # Save individual plot
        trendline_suffix = "_with_trendlines" if show_trendlines else ""
        output_file = f'aws_s3_upload_operations_{nodes}nodes{trendline_suffix}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Operation count plot saved as: {output_file}")
        
        plt.show()
    
    return len(node_configs)

def create_combined_plots(data, show_trendlines=False):
    """
    Create combined plots showing both performance and operation count for each node configuration.
    
    Args:
        data: Performance data list
        show_trendlines: Boolean to show/hide trendlines (default: False)
    """
    # Organize data by nodes
    data_by_nodes = defaultdict(list)
    for entry in data:
        data_by_nodes[entry['nodes']].append(entry)
    
    # Define transfer size order and colors
    transfer_sizes = ['1M', '50M', '100M', 'DEFAULT']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    markers = ['o', 's', '^', 'D']  # Circle, Square, Triangle, Diamond
    
    node_configs = sorted(data_by_nodes.keys())
    
    # Create combined plots for each node configuration
    for nodes in node_configs:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        node_data = data_by_nodes[nodes]
        
        # Organize data by transfer size
        size_data = defaultdict(lambda: {'tasks': [], 'bandwidth': [], 'std': [], 'operation_count': []})
        
        for entry in node_data:
            transfer_size = entry['transfer_size']
            if entry['successful_trials'] > 0:  # Only include successful trials
                size_data[transfer_size]['tasks'].append(entry['tasks_per_node'])
                size_data[transfer_size]['bandwidth'].append(entry['mean_bandwidth'])
                size_data[transfer_size]['std'].append(entry['std_bandwidth'])
                size_data[transfer_size]['operation_count'].append(entry['total_operations'])
        
        # Plot performance (top subplot)
        for i, transfer_size in enumerate(transfer_sizes):
            if transfer_size in size_data and size_data[transfer_size]['tasks']:
                tasks = np.array(size_data[transfer_size]['tasks'])
                bandwidth = np.array(size_data[transfer_size]['bandwidth'])
                std_dev = np.array(size_data[transfer_size]['std'])
                
                # Sort by tasks per node
                sort_idx = np.argsort(tasks)
                tasks = tasks[sort_idx]
                bandwidth = bandwidth[sort_idx]
                std_dev = std_dev[sort_idx]
                
                # Plot points with error bars
                linestyle = '-' if show_trendlines else 'None'
                ax1.errorbar(tasks, bandwidth, yerr=std_dev, 
                           color=colors[i], marker=markers[i], markersize=8,
                           label=f'{transfer_size}', linewidth=2, capsize=4,
                           linestyle=linestyle, alpha=0.8, markerfacecolor=colors[i],
                           markeredgecolor='black', markeredgewidth=0.5)
        
        # Customize performance plot
        ax1.set_xlabel('Tasks per Node (log scale)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Bandwidth per Rank (MB/s)', fontsize=12, fontweight='bold')
        ax1.set_title(f'AWS S3 Upload Performance - {nodes} Node{"s" if nodes > 1 else ""}', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
        ax1.set_xscale('log')
        
        # Plot operation count (bottom subplot)
        for i, transfer_size in enumerate(transfer_sizes):
            if transfer_size in size_data and size_data[transfer_size]['tasks']:
                tasks = np.array(size_data[transfer_size]['tasks'])
                operation_count = np.array(size_data[transfer_size]['operation_count'])
                
                # Sort by tasks per node
                sort_idx = np.argsort(tasks)
                tasks = tasks[sort_idx]
                operation_count = operation_count[sort_idx]
                
                # Plot points
                linestyle = '-' if show_trendlines else 'None'
                ax2.plot(tasks, operation_count, 
                        color=colors[i], marker=markers[i], markersize=8,
                        label=f'{transfer_size}', linewidth=2,
                        linestyle=linestyle, alpha=0.8, markerfacecolor=colors[i],
                        markeredgecolor='black', markeredgewidth=0.5)
        
        # Customize operation count plot
        ax2.set_xlabel('Tasks per Node (log scale)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Total Operations (All Tasks & Nodes)', fontsize=12, fontweight='bold')
        ax2.set_title(f'AWS S3 Upload Total Operations - {nodes} Node{"s" if nodes > 1 else ""}', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
        ax2.set_xscale('log')
        
        # Set x-axis ticks for both plots
        if size_data:
            all_tasks = []
            for size_info in size_data.values():
                all_tasks.extend(size_info['tasks'])
            if all_tasks:
                unique_tasks = sorted(set(all_tasks))
                ax1.set_xticks(unique_tasks)
                ax1.set_xticklabels([str(x) for x in unique_tasks], fontsize=10)
                ax2.set_xticks(unique_tasks)
                ax2.set_xticklabels([str(x) for x in unique_tasks], fontsize=10)
        
        plt.tight_layout()
        
        # Save combined plot
        trendline_suffix = "_with_trendlines" if show_trendlines else ""
        output_file = f'aws_s3_upload_combined_{nodes}nodes{trendline_suffix}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Combined plot saved as: {output_file}")
        
        plt.show()
    
    return len(node_configs)

def create_summary_table(data):
    """
    Create a summary table of all results for both upload and download operations.
    """
    print("\n" + "="*100)
    print("SUMMARY OF ALL BENCHMARK RESULTS")
    print("="*100)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(data)
    
    # Separate data by operation type
    upload_df = df[df['has_upload_stats'] == True].copy()
    download_df = df[df['has_download_stats'] == True].copy()
    old_format_df = df[df['has_old_format'] == True].copy()
    
    print(f"Total configurations parsed: {len(df)}")
    print(f"Configurations with upload stats: {len(upload_df)}")
    print(f"Configurations with download stats: {len(download_df)}")
    print(f"Configurations with old format: {len(old_format_df)}")
    
    # Display summary for upload operations
    if not upload_df.empty:
        print("\n" + "="*80)
        print("UPLOAD PERFORMANCE SUMMARY")
        print("="*80)
        
        successful_upload_df = upload_df[upload_df['upload_successful_trials'] > 0].copy()
        
        if not successful_upload_df.empty:
            print(f"Successful upload configurations: {len(successful_upload_df)}/{len(upload_df)}")
            print(f"Upload success rate: {len(successful_upload_df)/len(upload_df)*100:.1f}%")
            
            best_upload_bandwidth = successful_upload_df.loc[successful_upload_df['upload_mean_bandwidth'].idxmax()]
            best_upload_throughput = successful_upload_df.loc[successful_upload_df['upload_aggregate_throughput'].idxmax()]
            
            print(f"\nBest upload bandwidth per rank: {best_upload_bandwidth['upload_mean_bandwidth']:.2f} MB/s")
            print(f"  Configuration: {best_upload_bandwidth['nodes']} nodes, {best_upload_bandwidth['tasks_per_node']} tasks/node, {best_upload_bandwidth['transfer_size']}")
            
            print(f"\nBest upload aggregate throughput: {best_upload_throughput['upload_aggregate_throughput']:.2f} MB/s")
            print(f"  Configuration: {best_upload_throughput['nodes']} nodes, {best_upload_throughput['tasks_per_node']} tasks/node, {best_upload_throughput['transfer_size']}")
            
            # Upload performance by node count
            print(f"\nUPLOAD PERFORMANCE BY NODE COUNT")
            print(f"{'Nodes':<6} {'Configs':<8} {'Success':<8} {'Best BW/Rank':<12} {'Best Throughput':<15}")
            print("-" * 60)
            
            for nodes in sorted(upload_df['nodes'].unique()):
                node_df = upload_df[upload_df['nodes'] == nodes]
                successful_node_df = successful_upload_df[successful_upload_df['nodes'] == nodes]
                
                total_configs = len(node_df)
                successful_configs = len(successful_node_df)
                
                if not successful_node_df.empty:
                    best_bw = successful_node_df['upload_mean_bandwidth'].max()
                    best_tp = successful_node_df['upload_aggregate_throughput'].max()
                else:
                    best_bw = 0
                    best_tp = 0
                
                print(f"{nodes:<6} {total_configs:<8} {successful_configs:<8} {best_bw:<12.2f} {best_tp:<15.2f}")
    
    # Display summary for download operations
    if not download_df.empty:
        print("\n" + "="*80)
        print("DOWNLOAD PERFORMANCE SUMMARY")
        print("="*80)
        
        successful_download_df = download_df[download_df['download_successful_trials'] > 0].copy()
        
        if not successful_download_df.empty:
            print(f"Successful download configurations: {len(successful_download_df)}/{len(download_df)}")
            print(f"Download success rate: {len(successful_download_df)/len(download_df)*100:.1f}%")
            
            best_download_bandwidth = successful_download_df.loc[successful_download_df['download_mean_bandwidth'].idxmax()]
            best_download_throughput = successful_download_df.loc[successful_download_df['download_aggregate_throughput'].idxmax()]
            
            print(f"\nBest download bandwidth per rank: {best_download_bandwidth['download_mean_bandwidth']:.2f} MB/s")
            print(f"  Configuration: {best_download_bandwidth['nodes']} nodes, {best_download_bandwidth['tasks_per_node']} tasks/node, {best_download_bandwidth['transfer_size']}")
            
            print(f"\nBest download aggregate throughput: {best_download_throughput['download_aggregate_throughput']:.2f} MB/s")
            print(f"  Configuration: {best_download_throughput['nodes']} nodes, {best_download_throughput['tasks_per_node']} tasks/node, {best_download_throughput['transfer_size']}")
            
            # Download performance by node count
            print(f"\nDOWNLOAD PERFORMANCE BY NODE COUNT")
            print(f"{'Nodes':<6} {'Configs':<8} {'Success':<8} {'Best BW/Rank':<12} {'Best Throughput':<15}")
            print("-" * 60)
            
            for nodes in sorted(download_df['nodes'].unique()):
                node_df = download_df[download_df['nodes'] == nodes]
                successful_node_df = successful_download_df[successful_download_df['nodes'] == nodes]
                
                total_configs = len(node_df)
                successful_configs = len(successful_node_df)
                
                if not successful_node_df.empty:
                    best_bw = successful_node_df['download_mean_bandwidth'].max()
                    best_tp = successful_node_df['download_aggregate_throughput'].max()
                else:
                    best_bw = 0
                    best_tp = 0
                
                print(f"{nodes:<6} {total_configs:<8} {successful_configs:<8} {best_bw:<12.2f} {best_tp:<15.2f}")
    
    # Display summary for old format (backwards compatibility)
    if not old_format_df.empty:
        print("\n" + "="*80)
        print("OLD FORMAT PERFORMANCE SUMMARY")
        print("="*80)
        
        successful_old_df = old_format_df[old_format_df['successful_trials'] > 0].copy()
        
        if not successful_old_df.empty:
            print(f"Successful configurations: {len(successful_old_df)}/{len(old_format_df)}")
            print(f"Success rate: {len(successful_old_df)/len(old_format_df)*100:.1f}%")
            
            best_bandwidth = successful_old_df.loc[successful_old_df['mean_bandwidth'].idxmax()]
            best_throughput = successful_old_df.loc[successful_old_df['aggregate_throughput'].idxmax()]
            
            print(f"\nBest bandwidth per rank: {best_bandwidth['mean_bandwidth']:.2f} MB/s")
            print(f"  Configuration: {best_bandwidth['nodes']} nodes, {best_bandwidth['tasks_per_node']} tasks/node, {best_bandwidth['transfer_size']}")
            
            print(f"\nBest aggregate throughput: {best_throughput['aggregate_throughput']:.2f} MB/s")
            print(f"  Configuration: {best_throughput['nodes']} nodes, {best_throughput['tasks_per_node']} tasks/node, {best_throughput['transfer_size']}")
    
    # Overall configuration summary
    print(f"\nNode configurations: {sorted(df['nodes'].unique())}")
    print(f"Tasks per node tested: {sorted(df['tasks_per_node'].unique())}")
    print(f"Transfer sizes tested: {sorted(df['transfer_size'].unique())}")

def main():
    """
    Main function to run the analysis.
    """
    print("AWS S3 Upload Performance Analysis")
    print("=" * 50)
    
    # Check if we're in the right directory
    expected_dirs = ["results_1nodes", "results_2nodes", "results_4nodes", "results_8nodes"]
    missing_dirs = [d for d in expected_dirs if not os.path.exists(d)]
    
    if missing_dirs:
        print(f"Warning: Missing directories: {missing_dirs}")
        print("Make sure you're running this script from the directory containing the results folders.")
    
    # Collect all performance data
    print("Collecting performance data from log files...")
    data = collect_all_data()
    
    if not data:
        print("No data found! Check your directory structure and log files.")
        return
    
    print(f"Successfully parsed {len(data)} benchmark configurations")
    
    # Create summary table
    create_summary_table(data)
    
    # Create performance plots
    print("\nGenerating performance plots...")
    
    # Options for plot generation
    show_trendlines = False  # Change to True if you want connecting lines
    create_operation_plots = False  # Change to True to generate operation count plots
    create_combined_plots_flag = False  # Change to True to generate combined plots
    
    print(f"Trendlines: {'Enabled' if show_trendlines else 'Disabled'}")
    print(f"Operation count plots: {'Enabled' if create_operation_plots else 'Disabled'}")
    print(f"Combined plots: {'Enabled' if create_combined_plots_flag else 'Disabled'}")
    print("X-axis: Log scale")
    
    create_upload_performance_plots(data, show_trendlines=show_trendlines)
    
    create_download_performance_plots(data, show_trendlines=show_trendlines)
    
    # Create operation count plots (optional)
    if create_operation_plots:
        print("\nGenerating operation count plots...")
        create_operation_plots(data, show_trendlines=show_trendlines)
    
    # Create combined plots (optional)
    if create_combined_plots_flag:
        print("\nGenerating combined plots...")
        create_combined_plots(data, show_trendlines=show_trendlines)
    
    # Save data to CSV for further analysis
    df = pd.DataFrame(data)
    csv_file = 'aws_performance_summary.csv'
    df.to_csv(csv_file, index=False)
    print(f"Data exported to: {csv_file}")
    
    print("\nTo enable trendlines, set 'show_trendlines = True' in the main() function")
    print("To enable operation count plots, set 'create_operation_plots = True' in the main() function")
    print("To enable combined plots, set 'create_combined_plots_flag = True' in the main() function")

if __name__ == "__main__":
    main()