
import pytorch_lightning as pl
import torch
import warnings

class OOMMonitorCallback(pl.Callback):
    """
    Callback that attempts to catch and log OutOfMemory errors, 
    and prints memory stats before crashing.
    """
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        try:
            # Check memory before batch execution
            if torch.cuda.is_available():
                # If memory is dangerously high (>95%), warn
                mem_alloc = torch.cuda.memory_allocated()
                mem_res = torch.cuda.memory_reserved()
                max_mem = torch.cuda.get_device_properties(0).total_memory
                
                # if (mem_res / max_mem) > 0.95:
                #     warnings.warn(f"Warning: GPU Memory usage is very high! ({mem_res/1024**3:.2f} GB / {max_mem/1024**3:.2f} GB)")
        except Exception:
            pass

    def on_exception(self, trainer, pl_module, exception):
        if isinstance(exception, RuntimeError) and "out of memory" in str(exception).lower():
            print("\n" + "="*80)
            print("CRITICAL ERROR: CUDA OUT OF MEMORY (OOM) DETECTED")
            print("="*80)
            if torch.cuda.is_available():
                print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
                print(f"Reserved:  {torch.cuda.memory_reserved()/1024**3:.2f} GB")
                print(f"Max Alloc: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
            print("\nPossible solutions:")
            print("1. Decrease batch_size in your configuration file.")
            print("2. Decrease max_n_res (max protein length).")
            print("3. Decrease model size (layers, hidden dimensions).")
            print("4. Enable mixed_precision (already enabled if amp_level used).")
            print("="*80 + "\n")
