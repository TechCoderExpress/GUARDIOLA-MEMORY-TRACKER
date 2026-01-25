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
        print(f"\n=== Forward pass STARTED === seq_len={seq_len}, batch={batch}, hidden_size={self.hidden_size}")
        input_seq = input_seq.transpose(0, 1)  # → [seq_len, batch, input_size]

        if hidden is None:
            # Allocate stacked (h and c) as 3D → then slice to get 2D tensors
            hx_stacked = tracker.alloc_with_priority(
                shape=(2, batch, self.hidden_size),      # [2, batch, hidden]
                priority=1.0,
                dtype=torch.float32,
                desc="LSTM h+c stacked high pri"
            )
            hx_stacked.zero_()

            # Slice to get two 2D tensors (batch, hidden)
            h = hx_stacked[0]   # shape: (batch, hidden)
            c = hx_stacked[1]   # shape: (batch, hidden)

        else:
            h, c = hidden  # expect already 2D

        # Output buffer
        outputs = tracker.alloc_with_priority(
            shape=(seq_len, batch, self.hidden_size),
            priority=0.6,
            dtype=torch.float32,
            desc="output seq low pri"
        )

        for t in range(seq_len):
            # Now h and c are 2D → LSTMCell accepts them
            h, c = self.rnn_cell(input_seq[t], (h, c))
            outputs[t] = h   # in-place write
            if t % 100 == 0 and t > 0:
                print(f"  Processed timestep {t}/{seq_len}")
        print("=== Forward pass completed ===")
        
        tracker.summary()

        return outputs.transpose(0, 1), (h, c)  # back to [batch, seq, hidden]
    
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
    
    model = PriorityRNN(input_size=512, hidden_size=1024).to('cuda' if torch.cuda.is_available() else 'cpu')
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