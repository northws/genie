import os
import glob
import pandas as pd
import subprocess
import re
import multiprocessing
import time
import argparse
import sys
from tqdm import tqdm
from collections import defaultdict

# Configuration
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Default to relative paths if not provided
DEFAULT_RUN_DIR = os.path.join(BASE_DIR, "runs", "final_final-v0", "version_3", "samples", "epoch_499", "evaluations")
DEFAULT_REF_DB_DIR = os.path.join(BASE_DIR, "data", "pdbstyle-2.08")
DEFAULT_TMALIGN_EXEC = os.path.join(BASE_DIR, "packages", "TMscore", "TMalign")

# Globals to be set in main() or used as defaults
RUN_DIR = DEFAULT_RUN_DIR
RAW_DESIGN_DIR = os.path.join(RUN_DIR, "designs") 
REF_DB_DIR = DEFAULT_REF_DB_DIR
OUTPUT_CSV = os.path.join(RUN_DIR, "novelty.csv")
TMALIGN_EXEC = DEFAULT_TMALIGN_EXEC
INFO_CSV = os.path.join(RUN_DIR, "info.csv")

NUM_WORKERS = os.cpu_count() # Use all available cores

# Optimization parameters
LENGTH_TOLERANCE = 0.3  # Only compare structures with length within ±30%
EARLY_STOP_TM = 0.5     # Stop early if TM-score exceeds this (for novelty detection)
ENABLE_EARLY_STOP = True  # Enable early stopping optimization

def get_ref_list():
    print(f"Scanning reference PDBs in {REF_DB_DIR}...")
    refs = glob.glob(os.path.join(REF_DB_DIR, "**/*.ent"), recursive=True)
    print(f"Found {len(refs)} reference structures.")
    return refs

def get_pdb_length(pdb_path):
    """Get the number of CA atoms (residues) in a PDB file."""
    count = 0
    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                    count += 1
    except:
        pass
    return count

def build_ref_length_index(refs):
    """Build a length-indexed dictionary for fast filtering."""
    print("Building reference length index for fast filtering...")
    length_index = defaultdict(list)
    for ref_path in tqdm(refs, desc="Indexing references"):
        length = get_pdb_length(ref_path)
        if length > 0:
            length_index[length].append(ref_path)
    print(f"Indexed {sum(len(v) for v in length_index.values())} references by length.")
    return length_index

def get_refs_by_length(design_length, length_index, tolerance=LENGTH_TOLERANCE):
    """Get references within length tolerance of design."""
    min_len = int(design_length * (1 - tolerance))
    max_len = int(design_length * (1 + tolerance))
    filtered_refs = []
    for length in range(min_len, max_len + 1):
        filtered_refs.extend(length_index.get(length, []))
    return filtered_refs

def get_design_list():
    if os.path.exists(INFO_CSV):
        print(f"Loading metadata from {INFO_CSV}...")
        df = pd.read_csv(INFO_CSV)
        
        # Filter for Confidently Designable: scTM > 0.5 and pLDDT > 70
        df_conf = df[(df['scTM'] > 0.5) & (df['pLDDT'] > 70)]
        valid_domains = set(df_conf['domain'].astype(str))
        print(f"Filtered {len(valid_domains)} 'Confidently Designable' domains from info.csv.")
        
        all_designs = glob.glob(os.path.join(RAW_DESIGN_DIR, "*.pdb"))
        # Filter files
        designs = [d for d in all_designs if os.path.basename(d).replace(".pdb", "") in valid_domains]
        return designs
    else:
        print("info.csv not found, scanning design dir...")
        return glob.glob(os.path.join(RAW_DESIGN_DIR, "*.pdb"))

# Global list for workers
REF_LIST = []
REF_LENGTH_INDEX = {}

def init_worker(refs, tmalign_exec, length_index):
    global REF_LIST, TMALIGN_EXEC, REF_LENGTH_INDEX
    REF_LIST = refs
    TMALIGN_EXEC = tmalign_exec
    REF_LENGTH_INDEX = length_index

def processing_func(design_path):
    domain = os.path.basename(design_path).replace(".pdb", "")
    max_tm = 0.0
    closest_pdb = ""
    
    global REF_LIST, REF_LENGTH_INDEX
    
    # Optimization 1: Length-based pre-filtering
    design_length = get_pdb_length(design_path)
    if design_length > 0 and REF_LENGTH_INDEX:
        # Use length-filtered references (typically reduces search space by 80-90%)
        refs_to_compare = get_refs_by_length(design_length, REF_LENGTH_INDEX)
    else:
        # Fallback to full list
        refs_to_compare = REF_LIST
    
    comparisons_made = 0
    for ref_path in refs_to_compare:
        # Run TMalign
        cmd = [TMALIGN_EXEC, design_path, ref_path]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            comparisons_made += 1
            
            # Parse output
            match = re.search(r'TM-score=\s*([\d\.]+)\s*\(if normalized by length of Chain_1\)', res.stdout)
            if match:
                tm = float(match.group(1))
                if tm > max_tm:
                    max_tm = tm
                    closest_pdb = ref_path
                    
                    # Optimization 2: Early stopping for novelty detection
                    # If we find TM > threshold, this design is NOT novel, stop searching
                    if ENABLE_EARLY_STOP and tm > EARLY_STOP_TM:
                        break
        except Exception as e:
            continue
            
    return (domain, max_tm, closest_pdb, design_length, len(refs_to_compare), comparisons_made)

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate Novelty via TM-score")
    parser.add_argument("-i", "--input_dir", type=str, default=None, 
                        help=f"Input directory (default: {DEFAULT_RUN_DIR})")
    parser.add_argument("-o", "--output_csv", type=str, default=None, 
                        help="Output CSV file path (default: <input_dir>/novelty.csv)")
    parser.add_argument("--ref_dir", type=str, default=DEFAULT_REF_DB_DIR, 
                        help=f"Reference database directory (default: {DEFAULT_REF_DB_DIR})")
    parser.add_argument("--tmalign", type=str, default=DEFAULT_TMALIGN_EXEC, 
                        help=f"Path to TMalign executable (default: {DEFAULT_TMALIGN_EXEC})")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(),
                        help="Number of worker processes")
    parser.add_argument("--length_tolerance", type=float, default=0.3,
                        help="Length tolerance for pre-filtering (default: 0.3 = ±30%%)")
    parser.add_argument("--early_stop_tm", type=float, default=0.5,
                        help="TM-score threshold for early stopping (default: 0.5)")
    parser.add_argument("--no_early_stop", action="store_true",
                        help="Disable early stopping (find exact max TM)")
    parser.add_argument("--no_length_filter", action="store_true",
                        help="Disable length-based pre-filtering")
    return parser.parse_args()

def main():
    args = parse_args()

    # Update Globals based on args
    global RAW_DESIGN_DIR, OUTPUT_CSV, REF_DB_DIR, TMALIGN_EXEC, INFO_CSV, NUM_WORKERS
    global LENGTH_TOLERANCE, EARLY_STOP_TM, ENABLE_EARLY_STOP
    
    # Optimization settings
    LENGTH_TOLERANCE = args.length_tolerance
    EARLY_STOP_TM = args.early_stop_tm
    ENABLE_EARLY_STOP = not args.no_early_stop
    use_length_filter = not args.no_length_filter
    
    # 1. Setup Input Directory
    if args.input_dir:

        RAW_DESIGN_DIR = args.input_dir

        if os.path.exists(os.path.join(args.input_dir, "designs")):
             RAW_DESIGN_DIR = os.path.join(args.input_dir, "designs")
             INFO_CSV = os.path.join(args.input_dir, "info.csv")
        else:
             # Assume input_dir is the designs folder itself
             RAW_DESIGN_DIR = args.input_dir
             INFO_CSV = os.path.join(os.path.dirname(args.input_dir), "info.csv")
    else:
        # Use default structure
        RAW_DESIGN_DIR = os.path.join(DEFAULT_RUN_DIR, "designs")
        INFO_CSV = os.path.join(DEFAULT_RUN_DIR, "info.csv")
    
    # 2. Setup Output CSV
    if args.output_csv:
        OUTPUT_CSV = args.output_csv
    else:
        # Save in the parent of RAW_DESIGN_DIR usually
        OUTPUT_CSV = os.path.join(os.path.dirname(RAW_DESIGN_DIR), "novelty.csv")
        
    # 3. Setup Refs and Exec
    REF_DB_DIR = args.ref_dir
    TMALIGN_EXEC = args.tmalign
    NUM_WORKERS = args.num_workers

    if not os.path.exists(TMALIGN_EXEC):
        print(f"Error: TMalign not found at {TMALIGN_EXEC}")
        return

    designs = get_design_list()
    print(f"Found {len(designs)} designs to process.")
    
    refs = get_ref_list()
    if not refs:
        print("No references found.")
        return

    # To save time for demonstration, we process a small logical batch
    
    existing_domains = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            df_exist = pd.read_csv(OUTPUT_CSV)
            existing_domains = set(df_exist['domain'].astype(str))
            print(f"Resuming... {len(existing_domains)} already computed.")
        except:
            pass
            
    designs_to_run = [d for d in designs if os.path.basename(d).replace(".pdb", "") not in existing_domains]
    print(f"Remaining designs to compute: {len(designs_to_run)}")
    
    if not designs_to_run:
        print("All done.")
        return

    # Build length index for optimization
    if use_length_filter:
        length_index = build_ref_length_index(refs)
        print(f"\n=== Optimization Settings ===")
        print(f"  Length tolerance: ±{LENGTH_TOLERANCE*100:.0f}%")
        print(f"  Early stopping: {'Enabled (TM > ' + str(EARLY_STOP_TM) + ')' if ENABLE_EARLY_STOP else 'Disabled'}")
        print(f"  Estimated speedup: 5-20x (depending on data distribution)")
        print(f"=============================\n")
    else:
        length_index = {}
        print("Length filtering disabled, using full reference set.")

    print(f"Starting computation on {NUM_WORKERS} cores...")
    print("Reminder: For faster evaluation, consider using Novelty_Evaluation_GPU.py with Foldseek")
    
    # Batch processing to save progress
    batch_size = 10
    
    with multiprocessing.Pool(NUM_WORKERS, initializer=init_worker, initargs=(refs, TMALIGN_EXEC, length_index)) as pool:
        results = []
        total_comparisons = 0
        total_candidates = 0
        
        for i, res in tqdm(enumerate(pool.imap_unordered(processing_func, designs_to_run)), total=len(designs_to_run)):
            results.append(res[:3])  # Only save domain, max_tm, closest_pdb
            total_candidates += res[4]  # candidates considered
            total_comparisons += res[5]  # actual comparisons made
            
            if (i + 1) % batch_size == 0:
                save_results(results, OUTPUT_CSV)
                results = [] # Clear buffer
        
        # Save remaining
        if results:
            save_results(results, OUTPUT_CSV)
    
    # Print statistics
    avg_candidates = total_candidates / len(designs_to_run) if designs_to_run else 0
    avg_comparisons = total_comparisons / len(designs_to_run) if designs_to_run else 0
    reduction = (1 - avg_comparisons / len(refs)) * 100 if refs else 0
    print(f"\n=== Statistics ===")
    print(f"  Avg candidates per design: {avg_candidates:.0f} / {len(refs)} ({avg_candidates/len(refs)*100:.1f}%)")
    print(f"  Avg comparisons per design: {avg_comparisons:.0f} (search reduction: {reduction:.1f}%)")
    print(f"==================\n")

def save_results(results, output_file):
    # Append to file
    df = pd.DataFrame(results, columns=['domain', 'max_tm_to_pdb', 'closest_pdb_path'])
    hdr = not os.path.exists(output_file)
    df.to_csv(output_file, mode='a', header=hdr, index=False)
    print(f" Saved {len(results)} rows.")

if __name__ == "__main__":
    main()
