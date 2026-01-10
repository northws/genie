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

# Configuration
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

def get_ref_list():
    print(f"Scanning reference PDBs in {REF_DB_DIR}...")
    refs = glob.glob(os.path.join(REF_DB_DIR, "**/*.ent"), recursive=True)
    print(f"Found {len(refs)} reference structures.")
    return refs

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

def init_worker(refs, tmalign_exec):
    global REF_LIST, TMALIGN_EXEC
    REF_LIST = refs
    TMALIGN_EXEC = tmalign_exec

def processing_func(design_path):
    domain = os.path.basename(design_path).replace(".pdb", "")
    max_tm = 0.0
    closest_pdb = ""
    
    # Iterate all references
    # Note: This is computationally expensive (M * N)
    # Using a fast pre-filter (like length check) could help, but here we want strict TM.
    
    # We parse the length of the design once if possible, but TMalign does it.
    
    global REF_LIST
    
    for ref_path in REF_LIST:
        # Run TMalign
        # args: design ref
        cmd = [TMALIGN_EXEC, design_path, ref_path]
        try:
            # -a T means TM-score standardized by query length? 
            # Default TMalign outputs both.
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # Parse output
            # Look for: TM-score= 0.50000 (if normalized by length of Chain_1)
            # Assuming Chain_1 is the first argument (design_path)
            match = re.search(r'TM-score=\s*([\d\.]+)\s*\(if normalized by length of Chain_1\)', res.stdout)
            if match:
                tm = float(match.group(1))
                if tm > max_tm:
                    max_tm = tm
                    closest_pdb = ref_path
        except Exception as e:
            continue
            
    return (domain, max_tm, closest_pdb)

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
    return parser.parse_args()

def main():
    args = parse_args()

    # Update Globals based on args
    global RAW_DESIGN_DIR, OUTPUT_CSV, REF_DB_DIR, TMALIGN_EXEC, INFO_CSV, NUM_WORKERS
    
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

    print(f"Starting computation on {NUM_WORKERS} cores...")
    print("Reminder: Strictly comparing the novelty of each sample requires a significant amount of CPU computing time. When computing resources are limited, it is recommended to use Novelty_Evaluation_GPU.py")
    
    # Batch processing to save progress
    batch_size = 10
    
    with multiprocessing.Pool(NUM_WORKERS, initializer=init_worker, initargs=(refs, TMALIGN_EXEC)) as pool:
        results = []
        for i, res in tqdm(enumerate(pool.imap_unordered(processing_func, designs_to_run)), total=len(designs_to_run)):
            results.append(res)
            
            if (i + 1) % batch_size == 0:
                save_results(results, OUTPUT_CSV)
                results = [] # Clear buffer
        
        # Save remaining
        if results:
            save_results(results, OUTPUT_CSV)

def save_results(results, output_file):
    # Append to file
    df = pd.DataFrame(results, columns=['domain', 'max_tm_to_pdb', 'closest_pdb_path'])
    hdr = not os.path.exists(output_file)
    df.to_csv(output_file, mode='a', header=hdr, index=False)
    print(f" Saved {len(results)} rows.")

if __name__ == "__main__":
    main()
