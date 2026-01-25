import torch

class PriorityTracker:  # renamed to avoid confusion - it's still a tracker, not real allocator
    def __init__(self, device='cuda:0 ', max_vram_fraction=0.8):
      if 'cuda' in str(device).lower():
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available - PriorityTracker needs GPU")
        self.device = device
        total_mem = torch.cuda.get_device_properties(device).total_memory
        self.budget = int(total_mem * max_vram_fraction)
        self.used = 0
        # List of (priority, bytes_used, tensor) for potential eviction
        self.tracked = []

    def alloc_with_priority(self, shape, priority=1.0, dtype=torch.float32, desc="tensor"):
        # Calculate bytes needed
        numel = 1
        for dim in shape:
            numel *= dim
        elem_size = torch.tensor(1, dtype=dtype).element_size()
        bytes_needed = numel * elem_size

        # Check budget (simulated eviction warning)
        if self.used + bytes_needed > self.budget:
            print(f"Budget exceeded! ({self.used / 1e6:.1f} / {self.budget / 1e6:.1f} MB)")
            print("  → Would evict low-priority tensors here if real allocator")

        # Allocate²
        tensor = torch.empty(shape, dtype=dtype, device=self.device)
        self.used += bytes_needed
        self.tracked.append((priority, bytes_needed, tensor, desc))

        print(f"Allocated {desc:<20} | {bytes_needed / 1e6:6.1f} MB | priority {priority:.1f}")
        return tensor
    
    def evict_low_priority(self, needed_bytes=0):
         """
         Evict lowest-priority tracked tensors to free up space.
         - needed_bytes=0 means "free as much as possible" (cleanup mode)
         """
         if not self.tracked:
            print("No tracked allocations to evict")
            return

        # Sort by priority ascending (lowest first to evict)
         self.tracked.sort(key=lambda x: (x[0], -x[1]))

         freed_mb = 0
         i = 0
         while i < len(self.tracked):
            pri, bytes_used, tensor, desc = self.tracked[i]
            if needed_bytes > 0 and self.used + needed_bytes <= self.budget:
                break  # enough space now

            # Release the tensor reference
            tensor = None
            del tensor

            self.used -= bytes_used
            freed_mb += bytes_used / 1e6
            print(f"Evicted: {desc:<30} | pri {pri:.1f} | {bytes_used / 1e6:6.1f} MB")

            # Remove from list
            del self.tracked[i]

            # Don't increment i because we deleted the current element
            # (list shifts left)

        # Force PyTorch to release any cached memory back to GPU
         torch.cuda.empty_cache()

         print(f"Total freed: {freed_mb:.1f} MB")
         print(f"Remaining tracked: {self.used / 1e6:.1f} MB")
         
    def summary(self):
        print("\nTracked allocations (sorted by priority):")
        for pri, b, t, d in sorted(self.tracked, key=lambda x: x[0]):
            print(f"  pri {pri:.1f} | {b / 1e6:6.1f} MB | {d}")
        print(f"Total tracked: {self.used / 1e6:.1f} MB")