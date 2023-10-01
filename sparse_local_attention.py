import torch
import torch.nn.functional as F
import torch.nn as nn

"""
Copyright (c) 2023 Pritama Nayak

Created Date: Sunday, October 1st 2023, 11:50:11 pm
Author: Pritama Nayak

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS
IS AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.

"""
"""
Sparse Local Attention
----------------------

Sparse local attention focuses ononly a subset of input elements when
processing a particular part of the input. This decreases the computation
cost, especially for long sequences.

Attends only a limited, localized region around a particular position or token
in the input sequence. Good for document processing.

Fixed sparse attention
----------------------

As the name suggests, it will attend only to a fixed-size window around
each position. The window moves along the sequence as the network processes
the input.

Adaptive sparse attention
-------------------------
Its a variation of fixed sparse attention where the attention window size is
dynamically adjusted based on the input sequence. Here I have take the min of
window size and sequence length.

Stride sparse attention
-----------------------

This involves attending to positions in the input sequence with a fixed stride,
skipping certain positions.
"""


class SparseLocalAttention(nn.Module):
    def __init__(
        self,
        window_size=10,
        is_adaptive=False,
        is_stride=False,
        stride=2,
        *args,
        **kwargs
    ) -> None:
        super(SparseLocalAttention, self).__init__(*args, **kwargs)
        self.fix_window_size = window_size
        self.is_adaptive = is_adaptive
        self.is_stride = is_stride
        self.stride = stride

    def forward(self, inputs):
        # let input is of shape (batch, sequence_len, input_dim)
        _, sequence_length, _ = inputs.size()

        self.window_size = self.fix_window_size
        if self.is_adaptive:
            self.window_size = min(sequence_length, self.fix_window_size)

        # Create a mask to enforce local attention window
        attention_mask = torch.arange(sequence_length).unsqueeze(
            0
        )  # attention_mask = (1, sequence_len)

        if self.is_stride:
            attention_mask = (
                (attention_mask % self.stride == 0).float().to(inputs.device)
            )
        else:
            attention_mask = (
                attention_mask
                >= torch.arange(sequence_length).unsqueeze(1)
                - self.window_size
            ) & (
                attention_mask
                <= torch.arange(sequence_length).unsqueeze(1)
                + self.window_size
            )

        attention_mask = attention_mask.float().to(inputs.device)

        # Compute attention scores
        attention_scores = torch.matmul(inputs, inputs.transpose(1, 2))
        attention_scores *= (
            attention_mask - 1e9
        )  # mask out positions outside the window

        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention weights to input sequence
        output = torch.matmul(attention_weights, inputs)

        return output
