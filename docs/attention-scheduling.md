# SM75 attention scheduling gate

The production W8A8/Sol kernel launches one CTA for each 64-token Query block,
batch item, and Query head. CUDA already assigns the next ordinary CTA to the
next available SM. Replacing this grid with a software persistent queue only
helps when the entire Query grid is smaller than the resident CTA capacity and
individual tasks have extreme cost variance.

MiniMax H3 does not meet that condition. A 52,842-token, 56-head call launches
46,256 Query CTAs. Even with two resident 32 KiB CTAs on each of 72 SMs, this is
over 321 scheduling waves. The observed Sol route-density spread (roughly
0.24--0.32) is not large enough to overcome the atomic queue overhead. The
runtime therefore keeps the ordinary Query grid.

Split-K addresses a different problem: a short Query with a very long K/V may
not launch enough Query CTAs to fill the GPU. A naive two-way split for the H3
shape would require about 2.86 GiB of FP32 partial output/max/denominator state,
so it is explicitly rejected for long-Q self-attention. A future bounded
split-K kernel may target unequal short-Q/long-K calls, but it must combine
online-softmax states without allocating full-sequence partial outputs and must
pass an exact-sm75 benchmark before runtime dispatch is added.

Run the static gate with:

```bash
PYTHONPATH=kernel python kernel/scripts/analyze_attention_schedule.py --sm-count 72
```
