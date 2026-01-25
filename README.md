# PyTorch Priority Memory Tracker

A simple, user-controlled memory tracker for PyTorch that lets you allocate tensors with priorities and **manually evict low-priority allocations** when GPU memory pressure is high.

### Why this exists

In real workloads (especially RNN/LSTM-style models with long sequences), PyTorch often wastes GPU memory on large temporary buffers while critical tensors (like hidden/cell states) get squeezed.  

This PoC:
- Tracks allocations with user-defined priorities  
- Lets you manually evict low-priority tensors (e.g. big output buffers)  
- Helps avoid OOM by protecting high-priority items (e.g. recurrent states)

It is **not** a replacement for PyTorch's caching allocator — it is a **simulated, user-side tool** to experiment with priority-aware memory management.

Tested on RTX 3050 Laptop GPU (4 GB VRAM).

### Demo results (example run)

- Model: Custom LSTMCell loop  
- Input: batch=64, seq_len=512, hidden=1024  
- Tracked explicit allocations: ~135 MB  
- Peak GPU memory: ~2.1 GB  
- After manual eviction: current usage drops significantly (mostly left with high-priority hidden state)

### Requirements

- Python 3.10+  
- NVIDIA GPU with CUDA support (tested on RTX 3050 Laptop)  
- PyTorch with CUDA (cu121 or cu124 recommended)

### Environment Setup

1. Create and activate a virtual environment (recommended)

   ```bash
   # Create venv
   python -m venv myenv-cuda

   # Activate (Windows PowerShell)
   .\myenv-cuda\Scripts\Activate.ps1