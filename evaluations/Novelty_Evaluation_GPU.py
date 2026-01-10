import os
import sys
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import subprocess
import re
import copy
from multiprocessing import Pool
import functools
import multiprocessing
import argparse

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR = os.path.join(BASE_DIR, "runs", "final_final-v0", "version_3", "samples", "epoch_499", "evaluations")
RAW_DESIGN_DIR = os.path.join(RUN_DIR, "designs")
REF_DB_DIR = os.path.join(BASE_DIR, "data", "pdbstyle-2.08")
OUTPUT_CSV = os.path.join(RUN_DIR, "novelty_hybrid.csv")
TMALIGN_EXEC = os.path.join(BASE_DIR, "packages", "TMscore", "TMalign")
INFO_CSV = os.path.join(RUN_DIR, "info.csv")

# Constants
K_NEIGHBORS = 30 
BATCH_SIZE = 25 
TOP_K_SCREEN = 150  # Increased from 50 for better recall
SIM_CHUNK_SIZE = 5000  # Chunk size for similarity computation to avoid OOM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import ProteinMPNN
try:
    from protein_mpnn_utils import parse_PDB, StructureDatasetPDB, ProteinMPNN, cat_neighbors_nodes
except ImportError:
    try:
        sys.path.append(BASE_DIR)
        from protein_mpnn_utils import parse_PDB, StructureDatasetPDB, ProteinMPNN, cat_neighbors_nodes
    except ImportError:
        print("Error: protein_mpnn_utils.py not found.")
        sys.exit(1)

# Helper for gather_nodes if not imported
def gather_nodes(nodes, neighbor_idx):
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def get_mpnn_model():
    weight_path = os.path.join(BASE_DIR, "packages", "ProteinMPNN", "ca_model_weights", "v_48_020.pt")
    if not os.path.exists(weight_path):
        print(f"Error: Weights not found at {weight_path}")
        sys.exit(1)
    checkpoint = torch.load(weight_path, map_location=DEVICE)
    model = ProteinMPNN(
        num_letters=21, node_features=128, edge_features=128, hidden_dim=128, 
        num_encoder_layers=3, num_decoder_layers=3, augment_eps=0.0, k_neighbors=checkpoint['num_edges'],
        ca_only=True
    ) 
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model

def compute_embeddings(model, pdb_files):
    all_embeddings = []
    valid_files_out = []
    
    chunks = [pdb_files[i:i + BATCH_SIZE] for i in range(0, len(pdb_files), BATCH_SIZE)]
    
    with torch.no_grad():
        for chunk in tqdm(chunks, desc="Embedding Batch"):
            pdb_dicts = []
            for p_file in chunk:
                try:
                    # Try chain A first, fallback to all chains
                    d_list = parse_PDB(p_file, input_chain_list=['A'], ca_only=True)
                    if not d_list:
                        d_list = parse_PDB(p_file, ca_only=True)  # Try all chains
                    for d in d_list:
                        d['full_path'] = p_file # Attach full path
                    pdb_dicts.extend(d_list)
                except Exception as e:
                    print(f"Warning: Failed to parse {p_file}: {e}")
            
            if not pdb_dicts: continue
            
            batch_len = len(pdb_dicts)
            lengths = [len(p['seq']) for p in pdb_dicts]
            max_len = max(lengths)
            
            X = torch.zeros(batch_len, max_len, 3, device=DEVICE)
            mask = torch.zeros(batch_len, max_len, device=DEVICE)
            residue_idx = torch.zeros(batch_len, max_len, dtype=torch.long, device=DEVICE)
            chain_encoding_all = torch.zeros(batch_len, max_len, dtype=torch.long, device=DEVICE)
            
            valid_indices = []
            valid_files_temp = [] 
            
            for i, p in enumerate(pdb_dicts):
                l = lengths[i]
                
                # Check for coords
                keys = list(p.keys())
                coord_keys = [k for k in keys if k.startswith('coords_')]
                if not coord_keys: continue
                
                # Take first chain
                key = coord_keys[0] 
                chain_id = key.split('_')[-1]
                coords_dict = p[key] 
                
                try:
                    ca_key = f"CA_chain_{chain_id}"
                    
                    if ca_key not in coords_dict: continue 
                    
                    CA = np.array(coords_dict[ca_key]) # Should be [L, 1, 3]
                    if CA.ndim == 3 and CA.shape[1] == 1:
                        CA = CA[:, 0, :] # [L, 3]
                    
                except Exception:
                    continue

                if len(CA) == 0: continue

                c = torch.tensor(CA, dtype=torch.float32, device=DEVICE) 
                
                actual_l = c.shape[0]
                if actual_l > max_len: 
                     c = c[:max_len]
                     actual_l = max_len
                
                X[i, :actual_l] = c
                mask[i, :actual_l] = 1.0
                residue_idx[i, :actual_l] = torch.arange(actual_l, device=DEVICE)
                
                valid_indices.append(i)
                valid_files_temp.append(p.get('full_path', 'unknown')) 

            if not valid_indices: continue
            
            X = X[valid_indices]
            mask = mask[valid_indices]
            residue_idx = residue_idx[valid_indices]
            chain_encoding_all = chain_encoding_all[valid_indices]

            # Handle NaNs
            X = torch.nan_to_num(X, nan=0.0)

            try:
                # 1. Features
                E, E_idx = model.features(X, mask, residue_idx, chain_encoding_all)
                
                # 2. Initial h_V, h_E
                h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=DEVICE)
                h_E = model.W_e(E)
                
                # 3. Encoder Layers
                mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
                mask_attend = mask.unsqueeze(-1) * mask_attend
                
                for layer in model.encoder_layers:
                    h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
                
                # Global Mean Pooling
                h_V_masked = h_V * mask.unsqueeze(-1)
                div = mask.sum(dim=1, keepdim=True)
                div = torch.clamp(div, min=1.0)
                emb = h_V_masked.sum(dim=1) / div
                
                all_embeddings.append(emb.cpu())
                valid_files_out.extend(valid_files_temp)
                
            except Exception as e:
                print(f"Skipping batch due to error: {e}")
                continue

    if not all_embeddings: return None, []
    return torch.cat(all_embeddings, dim=0), valid_files_out

def process_design(i, design_paths, ref_paths_all, candidate_indices_list):
    d_path = design_paths[i]
    domain = os.path.basename(d_path).replace(".pdb", "")
    
    candidate_indices = candidate_indices_list[i]
    if len(candidate_indices) == 0:
         return f"{domain},0.0,\n"

    candidate_refs = [ref_paths_all[idx] for idx in candidate_indices]
    
    max_tm = 0.0
    best_ref = ""
    
    # Run TMalign
    for ref_p in candidate_refs:
        if not os.path.exists(ref_p):
            continue
        cmd = [TMALIGN_EXEC, d_path, ref_p]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            # Try both normalization patterns
            match = re.search(r'TM-score=\s*([\d\.]+)\s*\(if normalized by length of Chain_1', res.stdout)
            if not match:
                match = re.search(r'TM-score=\s*([\d\.]+)', res.stdout)
            if match:
                tm = float(match.group(1))
                if tm > max_tm:
                    max_tm = tm
                    best_ref = ref_p
        except subprocess.TimeoutExpired:
            continue
        except Exception as e:
            continue
    return f"{domain},{max_tm:.4f},{best_ref}\n"

def main():
    global RAW_DESIGN_DIR, OUTPUT_CSV, REF_DB_DIR

    parser = argparse.ArgumentParser(description="Hybrid GPU Screening")
    parser.add_argument("-i", "--input_dir", type=str, default=RAW_DESIGN_DIR, help="Input directory containing PDB designs")
    parser.add_argument("-o", "--output_csv", type=str, default=None, help="Output CSV file path")
    parser.add_argument("-r", "--ref_dir", type=str, default=REF_DB_DIR, help="Reference database directory")
    args = parser.parse_args()
    
    # Auto-detect designs folder if user points to parent directory
    if os.path.exists(os.path.join(args.input_dir, "designs")):
        RAW_DESIGN_DIR = os.path.join(args.input_dir, "designs")
        print(f"Detected 'designs' subdirectory. Setting input directory to: {RAW_DESIGN_DIR}")
        default_output_dir = args.input_dir
    else:
        RAW_DESIGN_DIR = args.input_dir
        if os.path.basename(os.path.normpath(RAW_DESIGN_DIR)) == 'designs':
            default_output_dir = os.path.dirname(os.path.normpath(RAW_DESIGN_DIR))
        else:
            default_output_dir = RAW_DESIGN_DIR

    if args.output_csv:
        OUTPUT_CSV = args.output_csv
    else:
        OUTPUT_CSV = os.path.join(default_output_dir, "novelty_hybrid.csv")

    REF_DB_DIR = args.ref_dir

    print("Initializing Hybrid GPU Screening...")
    
    # 0. Load Model
    model = get_mpnn_model()
    
    # 1. Load References
    print("Indexing Reference Database...")
    refs = glob.glob(os.path.join(REF_DB_DIR, "**/*.ent"), recursive=True) 
    
    if not refs:
         refs = glob.glob(os.path.join(REF_DB_DIR, "**/*"), recursive=True)
         refs = [f for f in refs if os.path.isfile(f) and not f.endswith('DIR')]
    
    # Filter only .ent or .pdb
    refs = [r for r in refs if r.endswith('.ent') or r.endswith('.pdb')]
    
    print(f"Found {len(refs)} reference files. Embedding References uses GPU.")
    ref_embs, ref_names = compute_embeddings(model, refs)
    if ref_embs is None: 
        print("Reference embedding failed.")
        return
    
    ref_embs = torch.nn.functional.normalize(ref_embs, p=2, dim=1).to(DEVICE)
    
    # 2. Load Designs (All)
    print("Loading all designs from directory (Processing all samples)...")
    designs = glob.glob(os.path.join(RAW_DESIGN_DIR, "*.pdb"))
    print(f"Processing all {len(designs)} designs.")
        
    print(f"Computing Embeddings for {len(designs)} designs...")
    design_embs, design_names = compute_embeddings(model, designs)
    if design_embs is None: 
        print("No designs embedded successfully.")
        return
    
    design_embs = torch.nn.functional.normalize(design_embs, p=2, dim=1).to(DEVICE)
    
    # 3. Compute Similarity Matrix (chunked to avoid OOM)
    print("Computing Similarity Matrix (chunked)...")
    num_designs = design_embs.shape[0]
    num_refs = ref_embs.shape[0]
    
    # Adjust TOP_K if reference set is smaller
    actual_top_k = min(TOP_K_SCREEN, num_refs)
    
    # Chunked computation for large matrices
    all_top_idxs = []
    all_top_vals = []
    
    for i in range(0, num_designs, SIM_CHUNK_SIZE):
        end_i = min(i + SIM_CHUNK_SIZE, num_designs)
        design_chunk = design_embs[i:end_i]
        
        # Compute similarity for this chunk
        sim_chunk = torch.matmul(design_chunk, ref_embs.T)
        
        # Get top-k for this chunk
        top_vals_chunk, top_idxs_chunk = torch.topk(sim_chunk, k=actual_top_k, dim=1)
        all_top_idxs.append(top_idxs_chunk.cpu())
        all_top_vals.append(top_vals_chunk.cpu())
        
        # Free memory
        del sim_chunk
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    top_idxs = torch.cat(all_top_idxs, dim=0).numpy()
    top_vals = torch.cat(all_top_vals, dim=0)
    
    # Debug stats
    print(f"DEBUG Stats: Top Sim Mean={top_vals.mean().item():.4f} Std={top_vals.std().item():.4f}")
    print(f"DEBUG Stats: Top Sim Max={top_vals.max().item():.4f} Min={top_vals.min().item():.4f}")
    
    # 4. Top-K Selection
    print(f"Selected Top-{actual_top_k} candidates for each design.")
    
    # Init CSV
    with open(OUTPUT_CSV, 'w') as f:
        f.write("domain,max_tm_to_pdb,closest_pdb_path\n")
            
    # Parallelize TM-align using multiprocessing
    num_designs = len(design_names)
    indices = list(range(num_designs))
    
    worker = functools.partial(process_design, design_paths=design_names, ref_paths_all=ref_names, candidate_indices_list=top_idxs)
    
    N_WORKERS = 20
    print(f"Running TMalign verification with {N_WORKERS} workers...")
    
    results = []
    with Pool(N_WORKERS) as p:
        for res_str in tqdm(p.imap_unordered(worker, indices), total=num_designs):
            results.append(res_str)
            if len(results) >= 50:
                with open(OUTPUT_CSV, 'a') as f:
                     for line in results: f.write(line)
                results = []

    if results:
        with open(OUTPUT_CSV, 'a') as f:
            for line in results:
                f.write(line)
    
    print(f"Finished. Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn') 
    try:
        main()
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print('\n' + '='*60)
            print('CRITICAL ERROR: CUDA Out of Memory (OOM) during screening.')
            print('='*60)
            print('Suggestions:')
            print('1. Reduce BATCH_SIZE in the script (currently defined as constant).')
            print('2. Check if other processes are using the GPU.')
            print('='*60 + '\n')
            sys.exit(1)
        else:
            raise e
