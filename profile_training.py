"""Profile training to identify bottlenecks"""
import torch
import torch.profiler
import time
import numpy as np
from genie.config import Config
from genie.diffusion.diffusion import Genie
from genie.data.data_module import SCOPeDataModule

def profile_training():
    # Load config
    config = Config(filename='runs/test_flash_ipa/on/configuration_l_ac_microbatch')
    
    # Setup
    dm = SCOPeDataModule(**config.io, batch_size=config.training['batch_size'], num_workers=config.training['num_workers'])
    dm.setup()
    
    model = Genie(config)
    model = model.cuda()
    model.train()
    
    # Get a batch
    dataloader = dm.train_dataloader()
    batch = next(iter(dataloader))
    batch = [b.cuda() for b in batch]
    
    # Warmup
    print("Warming up...")
    for _ in range(3):
        loss = model.training_step(batch, 0)
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()
    
    # Profile different components
    print("\n" + "="*60)
    print("PROFILING RESULTS")
    print("="*60)
    
    # 1. Data loading time
    print("\n[1] Data Loading:")
    times = []
    for i, batch in enumerate(dataloader):
        if i >= 5:
            break
        t0 = time.time()
        batch = [b.cuda(non_blocking=True) for b in batch]
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    print(f"    Avg batch load + transfer: {np.mean(times)*1000:.2f} ms")
    
    # 2. Forward pass
    print("\n[2] Forward Pass:")
    batch = next(iter(dataloader))
    batch = [b.cuda() for b in batch]
    torch.cuda.synchronize()
    
    times = []
    for _ in range(5):
        t0 = time.time()
        loss = model.training_step(batch, 0)
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    print(f"    Avg forward time: {np.mean(times)*1000:.2f} ms")
    
    # 3. Backward pass
    print("\n[3] Backward Pass:")
    times = []
    for _ in range(5):
        loss = model.training_step(batch, 0)
        torch.cuda.synchronize()
        t0 = time.time()
        loss.backward()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        model.zero_grad()
    print(f"    Avg backward time: {np.mean(times)*1000:.2f} ms")
    
    # 4. Full step
    print("\n[4] Full Training Step (forward + backward):")
    times = []
    for _ in range(5):
        torch.cuda.synchronize()
        t0 = time.time()
        loss = model.training_step(batch, 0)
        loss.backward()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        model.zero_grad()
    print(f"    Avg full step: {np.mean(times)*1000:.2f} ms")
    print(f"    Throughput: {config.training['batch_size'] / np.mean(times):.1f} samples/sec")
    
    # 5. GPU utilization check
    print("\n[5] Memory Usage:")
    print(f"    Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"    Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    print(f"    Max Allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

if __name__ == '__main__':
    profile_training()
