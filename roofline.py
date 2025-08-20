import os
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from labellines import labelLine
import json
import glob

def parse_size(size_str):
    units = {"K": 1024, "M": 1024**2, "G": 1024**3}
    size_str = size_str.strip().upper()

    num = ''
    unit = ''
    for char in size_str:
        if char.isdigit() or char == '.':
            num += char
        else:
            unit += char

    if not unit or unit not in units:
        raise ValueError(f"Unbekannte Einheit: '{unit}'")

    return int(float(num) * units[unit])

class Peak:
    def __init__(self,task_per_node=1, bw=0.0, iops=-1, transfer_size=0, interface="native-s3"):
        self.task_per_node = task_per_node
        self.bw = bw
        self.iops = iops
        self.transfer_size = transfer_size
        self.interface = interface

    def __repr__(self):
        return (f"tasks_per_node={self.task_per_node}, "
                f"bw={self.bw:.2f} MB/s, iops={self.iops}, "
                f"transfer_size={self.transfer_size} Bytes, "
                f"interface={self.interface}")  

def draw_roofline_multi(df1, df2, df3, start=10**-9, end=10**-5, peaks=None,
                        is_read=False, is_aggregated=False, custom_peaks=True,
                        fname="roofline.png"):

    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import numpy as np

    app_cmap = plt.get_cmap('Set1')
    base_cmap = plt.get_cmap('Dark2')
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '+']

    fig, axs = plt.subplots(1, 3, figsize=(21, 6), sharey=True)
    dfs = [df1, df2, df3]
    titles = ["NFS-mounted Storage Gateway (POSIX)", "SDK-based transfers (native-s3)", "Data migration through Storage Gateway (Copy)"]

    for ax, df, title in zip(axs, dfs, titles):
        ax.set_title(f"{title} ({'Read' if is_read else 'Write'})", fontsize=14)

        peak_handles = []
        point_handles = []

        if custom_peaks and peaks:
            for p_index, peak in enumerate(peaks):
                x = np.linspace(start, end, 100000)
                y = np.minimum(x * peak['peak_bw'], peak['peak_ops'])
                line, = ax.plot(x, y, label=f"{peak['peak_name']}", color=base_cmap(p_index))
                peak_handles.append(line)

        if df is not None and not df.empty:
            df = df[df['is_read'] == is_read]
            unique_apis = df['api'].unique()
            api_color_map = {api: app_cmap(i) for i, api in enumerate(unique_apis)}

            for api in unique_apis:
                api_df = df[df['api'] == api]
                xs = api_df['OI'].values
                ys = api_df['peak_ops'].values
                names = api_df['peak_name'].values

                color = api_color_map[api]
                for i, (x, y, name) in enumerate(zip(xs, ys, names)):
                    marker = markers[i % len(markers)]
                    pt = ax.scatter(x, y, color=app_cmap(i), marker=marker)
                    point_handles.append((mlines.Line2D([], [], color=app_cmap(i), marker=marker, linestyle='', markersize=8), name))

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="major", ls="-.")
        ax.set_xlabel("IOI [IOP/Byte]", fontsize=14)

        if peak_handles:
            leg1 = ax.legend(
                handles=peak_handles,
                loc='lower right',
                fontsize=13
            )
            ax.add_artist(leg1)

        if point_handles:
            handles, labels = zip(*point_handles)
            seen = set()
            unique = []
            unique_labels = []
            for h, l in zip(handles, labels):
                if l not in seen:
                    unique.append(h)
                    unique_labels.append(l)
                    seen.add(l)
            leg2 = ax.legend(
                unique, unique_labels,
                loc='upper left',
                fontsize=13
            )
            ax.add_artist(leg2)

    axs[0].set_ylabel("IOPS", fontsize=14)
    axs[0].set_ylim(bottom=10**0, top=2*10**5)

    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')
    plt.show()


def collect_all_data(base_directory=".", ts=1048576):
    """
    Collect performance data from all log files in the directory structure.
    """
    all_data = []
    
    # Find all benchmark log files
    for node_dir in ["results_1nodes", "results_2nodes", "results_4nodes", "results_8nodes", "results_10nodes"]:
        log_pattern = os.path.join(base_directory, node_dir, "benchmark_*.log")
        log_files = glob.glob(log_pattern)
        
        # print(f"Found {len(log_files)} log files in {node_dir}")
        
        for log_file in log_files:
            result = parse_log_file(log_file, ts=ts)
            if result:
                all_data.append(result)
            else:
                print(f"Failed to parse: {log_file}")
    
    return all_data

def parse_log_file(file_path, ts=1048576):
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
            'transfer_size': parse_size(transfer_size),
            'file_path': file_path,
            'file_size_mb': file_size_mb,
            'operations_per_upload': operations_per_upload,
            'total_operations': total_operations,
            'has_upload_stats': False,
            'has_download_stats': False,
            'has_old_format': False
        }
        
        # Parse upload statistics if available
        if upload_stats_match and parse_size(transfer_size) == ts:
            result.update({
                'has_upload_stats': True,
                'upload_successful_trials': int(upload_stats_match.group(1)),
                'upload_total_trials': int(upload_stats_match.group(2)),
                'upload_mean_bandwidth': float(upload_stats_match.group(3)),
                'upload_std_bandwidth': float(upload_stats_match.group(4)),
                'upload_aggregate_throughput': float(upload_stats_match.group(5))
            })
        
        # Parse download statistics if available
        if download_stats_match and parse_size(transfer_size) == ts:
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

def get_peak_aws(data, is_upload=True, nodes=2):
    """
    Create a summary table of all results for both upload and download operations.
    """
    # print("\n" + "="*100)
    # print("SUMMARY OF ALL BENCHMARK RESULTS")
    # print("="*100)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(data)
    
    # Separate data by operation type
    upload_df = df[df['has_upload_stats'] == True].copy()
    download_df = df[df['has_download_stats'] == True].copy()

    peak = Peak(iops=1293116)

    if is_upload:
        successful_upload_df = upload_df[upload_df['upload_successful_trials'] > 0].copy()
        
        if not successful_upload_df.empty:
            best_upload_for_node = successful_upload_df[
                (successful_upload_df['nodes'] == nodes)
            ].loc[lambda df: df['upload_aggregate_throughput'].idxmax()]
            peak.bw = best_upload_for_node['upload_aggregate_throughput']
            peak.task_per_node = best_upload_for_node['tasks_per_node']
            peak.transfer_size = best_upload_for_node['transfer_size']
            print(peak.bw)
            peak.iops = (peak.bw *1000**2) / (best_upload_for_node['transfer_size']) 
    else:
            successful_download_df = download_df[download_df['download_successful_trials'] > 0].copy()
            
            if not successful_download_df.empty:
                best_download_for_node = successful_download_df[
                    (successful_download_df['nodes'] == nodes)
                ].loc[lambda df: df['download_aggregate_throughput'].idxmax()]
                peak.bw = best_download_for_node['download_aggregate_throughput']
                peak.task_per_node = best_download_for_node['tasks_per_node']
                peak.transfer_size = best_download_for_node['transfer_size']
                peak.iops = (peak.bw *1000**2) / (best_download_for_node['transfer_size']) 
    return peak

def get_peak_ior(pks, is_read=False, nodes=1):
    if is_read:
        return pks['read'].get(nodes, None)
    else:
        return pks['write'].get(nodes, None)

def load_ior_data(node_cnt, directory):
    """
    Load and parse bandwidth.log files from the specified directory.

    Args:
        directory (str): Path to the directory containing bandwidth.log files.

    Returns:
        pd.DataFrame: DataFrame containing parsed data.
    """
    data = []

    for test_folder in os.listdir(directory):
        test_folder_path = os.path.join(directory, test_folder)

        # Skip if not a directory
        if not os.path.isdir(test_folder_path):
            continue

        # print(f"test_folder: {test_folder}")

        # Parse test configuration from folder name
        test_config_str = test_folder.split('_')
        if len(test_config_str) < 4:
            print(f"Warning: Unexpected folder name format: {test_folder}")
            continue

        tasks_per_node = test_config_str[2]
        ts_size = test_config_str[3].replace("ts", "")

        # Look for bandwidth.log files in the test folder
        for file in os.listdir(test_folder_path):
            if file.endswith("bandwidth.log"):
                file_path = os.path.join(test_folder_path, file)  # Fixed: use test_folder_path

                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        match = re.search(r'Copy bandwidth: ([\d.]+) MB/s', content)
                        if match:
                            bandwidth = float(match.group(1))
                            line_data = {
                                "ts_size": ts_size,
                                "tasks_per_node": int(tasks_per_node),
                                "bandwidth": bandwidth,
                                "latency": 0,
                                "node_count": int(node_cnt)
                            }
                            data.append(line_data)
                            # print(f"line_data : {line_data}")
                        else:
                            print(f"Warning: No bandwidth data found in {file_path}")

                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    return pd.DataFrame(data)

def find_best_bandwidth_ior(base_dir, is_read=False, ts=1048576):
    """
    Find the single best config separately for write and read,
    print details and save each to its own CSV.
    """
    best_write = None
    best_read = None

    # Scan JSON files and track the best result per operation
    for root, _, files in os.walk(base_dir):
        for file in files:
            if not file.endswith(".json"):
                continue
            json_path = os.path.join(root, file)
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue

            for entry in data.get("summary", []):
                op = entry.get("operation")
                tasks_node = entry.get("tasksPerNode")
                bw_mean = entry.get("bwMeanMIB")
                xfer_size = entry.get("transferSize")
                iops = entry.get("OPsMean")

                if None in (op, tasks_node, bw_mean, xfer_size, iops):
                    continue

                candidate = {
                    "operation": op,
                    "tasksPerNode": tasks_node,
                    "transferSize": xfer_size,
                    "bwMeanMIB": bw_mean,
                    "iops": iops,
                }

                if op == "write" and xfer_size == ts:
                    if best_write is None or bw_mean > best_write["bwMeanMIB"]:
                        best_write = candidate
                elif op == "read" and xfer_size == ts:
                    if best_read is None or bw_mean > best_read["bwMeanMIB"]:
                        best_read = candidate

    # Select based on is_read
    op = "read" if is_read else "write"
    best_entry = best_read if is_read else best_write

    if not best_entry:
        print(f"No valid {op} entries found.")
        return
    # Create and return the Peak object, however we need to adjust the bw from MIB/s to MB/s
    return Peak(best_entry['tasksPerNode'], best_entry['bwMeanMIB']*1.049, best_entry['iops'], best_entry['transferSize'], "ior-posix")

def load_bandwidth_data(node_cnt, directory, ts=1048576):
    """
    Load and parse bandwidth.log files from the specified directory.

    Args:
        directory (str): Path to the directory containing bandwidth.log files.

    Returns:
        pd.DataFrame: DataFrame containing parsed data.
    """
    data_upload = []
    data_download = []

    for test_folder in os.listdir(directory):
        test_folder_path = os.path.join(directory, test_folder)

        # Skip if not a directory
        if not os.path.isdir(test_folder_path):
            continue

        # print(f"test_folder: {test_folder}")

        # Parse test configuration from folder name
        test_config_str = test_folder.split('_')
        if len(test_config_str) < 4:
            print(f"Warning: Unexpected folder name format: {test_folder}")
            continue

        tasks_per_node = test_config_str[3]
        ts_size = test_config_str[4].replace("ts", "")
        ts_size = "100M"
        if ts == parse_size(ts_size) and node_cnt == test_config_str[0].split('n')[0]:
        # Look for bandwidth.log files in the test folder
            for file in os.listdir(test_folder_path):
                if file.endswith("download_bandwidth.log"):
                    file_path = os.path.join(test_folder_path, file)  # Fixed: use test_folder_path

                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                            match = re.search(r'Download bandwidth: ([\d.]+) MB/s', content)
                            if match:
                                bandwidth = float(match.group(1))
                                line_data = {
                                    "ts_size": parse_size(ts_size),
                                    "tasks_per_node": int(tasks_per_node),
                                    "bandwidth": bandwidth,
                                    "latency": 0,
                                    "node_count": int(node_cnt)
                                }
                                data_download.append(line_data)
                                # print(f"line_data : {line_data}")
                            else:
                                print(f"Warning: No bandwidth data found in {file_path}")

                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
                elif file.endswith("upload_bandwidth.log"):
                    file_path = os.path.join(test_folder_path, file)
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                            match = re.search(r'Upload bandwidth: ([\d.]+) MB/s', content)
                            if match:
                                bandwidth = float(match.group(1))
                                line_data = {
                                    "ts_size": parse_size(ts_size),
                                    "tasks_per_node": int(tasks_per_node),
                                    "bandwidth": bandwidth,
                                    "latency": 0,
                                    "node_count": int(node_cnt)
                                }
                                data_upload.append(line_data)
                                # print(f"line_data : {line_data}")
                            else:
                                print(f"Warning: No bandwidth data found in {file_path}")
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

    return pd.DataFrame(data_download), pd.DataFrame(data_upload)


def find_best_bandwidth_upload_download(data):

    """
    Find the single best bandwidth config from the DataFrame.
    Input:
        data: pd.DataFrame with at least columns ['tasks_per_node', 'bandwidth']
    """
    if data.empty:
        print("Warning: Input data is empty")
        return

    # Find row with max bandwidth
    best_row = data.loc[data['bandwidth'].idxmax()]
    # print(best_row)
    return Peak(best_row['tasks_per_node'], best_row['bandwidth'], (best_row['bandwidth']*1000**2)/best_row['ts_size'], best_row['ts_size'], "copy")
    

def get_peaks_100(ts=1048576):
    base_root = './'

    configs= {
        '1': '1n_gateway_results_100bs',
        '2': '2n_gateway_results_100bs',
        '4': '4n_gateway_results_100bs',
        '8': '8n_gateway_results_100bs',
        '10': '10n_gateway_results_100bs',
    }

    peaks = {'read':{}, 'write':{}}

    # For IOR_Read
    for node_cnt, subdir in configs.items():
        input_dir = os.path.join(base_root, subdir)
        output_file = f'best_read_bandwidth_100mb_pertask_{node_cnt}n'
        peaks['read'][node_cnt] = find_best_bandwidth_ior(input_dir, is_read=True, ts=ts)

    # For_IOR_WRITE
    for node_cnt, subdir in configs.items():
        input_dir = os.path.join(base_root, subdir)
        output_file = f'best_write_bandwidth_100mb_pertask_{node_cnt}n'
        peaks['write'][node_cnt] = find_best_bandwidth_ior(input_dir, is_read=False, ts=ts)
    return peaks



def get_cp_100(ts=1048576):
    base_root = './'

    configs_cp = {
        '1': "1n_gateway_cp_results_100bs",
        '2': "2n_gateway_cp_results_100bs",
        '4': "4n_gateway_cp_results_100bs",
        '8': "8n_gateway_cp_results_100bs",
        '10': "10n_gateway_cp_results_100bs",
    }


    peaks = {'upload':{}, 'download': {}}

    for node_cnt, subdir in configs_cp.items():
        input_dir = os.path.join(base_root, subdir)
        df_down, df_up = load_bandwidth_data(node_cnt, input_dir, ts=ts)

        peaks['download'][node_cnt] = find_best_bandwidth_upload_download(df_down)
        peaks['upload'][node_cnt] = find_best_bandwidth_upload_download(df_up)

    return peaks

def get_peak_cp(pks, is_read=False, nodes=1):
    if is_read:
        return pks['download'].get(nodes, None)
    else:
        return pks['upload'].get(nodes, None)



if __name__ == '__main__':

    # ts = 1048576
    # is_read=False
    # node=8
    aws_ior_peaks = []
    write_roof_peaks = []
    cp_peaks = []
    # Add GEthernet peak
    small = 1500
    mid = 1000**2
    large = 100 * 1000**2

    peak_net_10_small = {'peak_ops': (1250 * (1000 ** 2) )/ small, 'peak_bw': 1250 * (1000 ** 2), 'peak_name': f"10GbE({small/1000**2:.4f} MB) {1250} MB/s, {(1250 * (1000 ** 2) )/ small:.0f} IOPS"}
    peak_net_10_mid = {'peak_ops': (1250 * (1000 ** 2) )/ mid, 'peak_bw': 1250 * (1000 ** 2), 'peak_name': f"10GbE({mid/1000**2:.0f} MB) {1250} MB/s, {(1250 * (1000 ** 2) )/ mid:.0f} IOPS"}
    peak_net_10_large = {'peak_ops': (1250 * (1000 ** 2) )/ large, 'peak_bw': 1250 * (1000 ** 2), 'peak_name': f"10GbE({large/1000**2:.0f} MB) {1250} MB/s, {(1250 * (1000 ** 2) )/ large:.0f} IOPS"}
    write_roof_peaks = [peak_net_10_large, peak_net_10_mid, peak_net_10_small]

    apps_1 = []
    apps_2 = []
    apps_3 = []

    for node in [1, 2, 4, 8, 10]:
        for ts in [1048576, 104857600]:
            for is_read in [True, False]:

                #Need App performance and think about the peak IOPS for AWS --> Currently setup to 1293116 frames/s 

                print(f"Processing for {node} nodes, ts={ts}, is_read={is_read}")
                ior_100_ts_1mb = get_peaks_100(ts=ts)
                aws_100_ts_1mb = collect_all_data("./aws_s3_100mb_per_task_results", ts=ts)
                cp_100_ts_1mb = get_cp_100(ts=ts)
                
                
                p_ior = get_peak_ior(ior_100_ts_1mb, is_read=is_read, nodes=str(node))
                p_aws = get_peak_aws(aws_100_ts_1mb, is_upload=False if is_read else True, nodes=node)
                p_cp = get_peak_cp(cp_100_ts_1mb, is_read=is_read, nodes=str(node))

                apps_1.append({'peak_ops': p_ior.iops, 'peak_bw': p_ior.bw*(1000 ** 2), 'peak_name': f"{p_ior.task_per_node}TPN_{node}N: {p_ior.bw:.0f} MB/s, {p_ior.iops:.0f} IOPS", 'api': 'POSIX (IOR)', 'is_read': is_read})
                apps_2.append({'peak_ops': p_aws.iops, 'peak_bw': p_aws.bw*(1000 ** 2), 'peak_name': f"{p_ior.task_per_node}TPN_{node}N: {p_aws.bw:.0f} MB/s, {p_aws.iops:.0f} IOPS", 'api': 'S3 (Native)' , 'is_read': is_read})
                if p_cp is not None:
                    apps_3.append({'peak_ops': p_cp.iops, 'peak_bw': p_cp.bw*(1000 ** 2), 'peak_name': f"{p_ior.task_per_node}TPN_{node}N: {p_cp.bw:.0f} MB/s, {p_cp.iops:.0f} IOPS", 'api': 'S3 (Copy)' , 'is_read': is_read})


    df1 = pd.DataFrame(apps_1)
    df2 = pd.DataFrame(apps_2)
    df3 = pd.DataFrame(apps_3)

    # OI berechnen
    for df in [df1, df2, df3]:
        df['OI'] = df['peak_ops'] / df['peak_bw']

    # Read-Plot
    draw_roofline_multi(df1, df2, df3, peaks=write_roof_peaks, is_read=True,
                        custom_peaks=True, fname="roofline_read_all.png")

    # Write-Plot
    draw_roofline_multi(df1, df2, df3, peaks=write_roof_peaks, is_read=False,
                        custom_peaks=True, fname="roofline_write_all.png")