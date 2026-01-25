import torch
from torch import nn
from PriorityAllocator import PriorityTracker   # your file

# Create one tracker per experiment
tracker = PriorityTracker(device='cuda:0', max_vram_fraction=0.7)

class PriorityRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = nn.LSTMCell(input_size, hidden_size)

    def forward(self, input_seq, hidden=None):
     batch, seq_len, _ = input_seq.shape
     input_seq = input_seq.transpose(0, 1)  # [seq_len, batch, input_size]

     print(f"\n=== Forward pass STARTED === seq_len={seq_len}, batch={batch}, hidden_size={self.hidden_size}")

     if hidden is None:
        hx_stacked = tracker.alloc_with_priority(
            (2, batch, self.hidden_size),
            priority=1.0,  # High priority: critical hidden state
            dtype=torch.float32,
            desc="LSTM h+c stacked high pri"
        )
        hx_stacked.zero_()
        h, c = hx_stacked[0], hx_stacked[1]  # 2D slices
        print("Input dtype:", input_seq.dtype)
        print("h dtype:", h.dtype)
        print("c dtype:", c.dtype)
        
     else:
        h, c = hidden
        print("Input dtype:", input_seq.dtype)
        print("h dtype:", h.dtype)
        print("c dtype:", c.dtype)

    # Add a medium-high priority intermediate buffer (simulated gate calculation)
     gate_intermediate = tracker.alloc_with_priority(
        (seq_len, batch, self.hidden_size * 2),
        priority=0.8,  # Medium-high: important but not critical
        dtype=torch.float32,
        desc="gate intermediate medium-high pri"
    )
     # Add a low priority log buffer (simulated disposable)
     log_buffer = tracker.alloc_with_priority(
        (seq_len, batch),
        priority=0.2,  # Low: disposable, evict first
        dtype=torch.float32,
        desc="log buffer low pri"
     )

    # Add a medium-low priority temp calculation tensor (simulated)
     temp_calc = tracker.alloc_with_priority(
        (batch, self.hidden_size),
        priority=0.5,  # Medium-low: temporary, can be evicted if needed
        dtype=torch.float32,
        desc="temp calculation medium-low pri"
     )

    

     outputs = tracker.alloc_with_priority(
        (seq_len, batch, self.hidden_size),
        priority=0.6,  # Medium: output seq
        dtype=torch.float32,
        desc="output seq medium pri"
     )

     for t in range(seq_len):
        h, c = self.rnn_cell(input_seq[t], (h, c))
        outputs[t] = h

        # Simulate using the other buffers (to make it clear they are part of the process)
        gate_intermediate[t] = torch.cat((h * 2.0, h * 2.0), dim=1)  # Dummy operation
        #temp_calc = h + c.mean(dim=1)  # Dummy temp calc
        temp_calc= h+c
        log_buffer[t] = temp_calc.sum(dim=1)  # Dummy log

        if t % 100 == 0 and t > 0:
            print(f"  Processed timestep {t}/{seq_len}... current mem: {torch.cuda.memory_allocated() / 1e6:.1f} MB")

        
     print("=== Forward pass COMPLETED ===")

     tracker.summary()

     return outputs.transpose(0, 1), (h, c)
# def evict_low_priority(self, needed_bytes):
#     if self.used + needed_bytes > self.budget:
#         self.tracked.sort(key=lambda x: x[0])  # low pri first
#         while self.used + needed_bytes > self.budget and self.tracked:
#             pri, b, t, d = self.tracked.pop(0)
#             t = None  # drop reference
#             del t     # encourage GC
#             self.used -= b
#             print(f"Evicted {d} (pri {pri}, {b/1e6:.1f} MB)")

# ------------------ Test / Run the model ------------------
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("No GPU - running on CPU (results will be slower)")
    
    tracker = PriorityTracker(device='cuda:0', max_vram_fraction=0.7)
    model = PriorityRNN(input_size=512, hidden_size=2048).to('cuda' if torch.cuda.is_available() else 'cpu')
    input_seq = torch.randn(64, 512, 512).to(model.rnn_cell.weight_ih.device)  # same device as model

    print("\n=== Starting forward pass ===")
    output, (h_final, c_final) = model(input_seq)

# Show peak during computation
    print("\nPeak during computation:", torch.cuda.max_memory_allocated() / 1e6, "MB")
    
    # ← Here is the good place to evict low-priority stuff
    # (e.g. the output buffer that we don't need anymore after forward)
    tracker.evict_low_priority(needed_bytes=0)   # needed=0 means "free what you can"
    # Show results clearly
    print("\n=== Results ===")
    print("Output shape:          ", output.shape)          # Should be [32, 100, 20]
    print("Final hidden h shape: ", h_final.shape)          # Should be [32, 20]
    print("Final cell c shape:   ", c_final.shape)          # [32, 20]
    print("Sample output[0,0,:5]:", output[0, 0, :5])       # first few values of first sequence
    print("Current allocated after eviction:", torch.cuda.memory_allocated() / 1e6, "MB")
    print("Peak GPU memory used: ", torch.cuda.max_memory_allocated() / 1e6, "MB")

    # Show tracker summary again at end
    tracker.summary()