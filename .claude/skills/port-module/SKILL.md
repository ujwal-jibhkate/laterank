---
name: port-module
description: How to port code from the smve-lab research repo (~/Projects/smve-lab/src/smve_lab) into laterank. Use whenever moving or rewriting smve-lab code (maxsim, smve, muvera, bm25, inverted_index, metrics, storage, datasets, bge_m3, reranker), including torch-to-numpy conversion.
---

# Porting a module from smve-lab

The source lives in `~/Projects/smve-lab/src/smve_lab/`, with its tests in
`~/Projects/smve-lab/tests/`. Its results under `~/Projects/smve-lab/results/` are the
reference numbers (see the `golden-values` skill).

## Steps

1. **Read the original and its tests in full** before writing anything. Note every constant (for
   example `k1=1.2`, `b=0.75`, the `2**26` chunk size, the 200,000-token sample in `token_mean`) and
   keep them, unless the owner agrees to a change.
2. **Choose the layer** (see CLAUDE.md). Math on arrays goes in `kernels/`, reading and writing
   files goes in `storage/`, and anything that loads a model goes in `components/`, imported
   lazily.
3. **Convert torch to numpy** (kernels/ and storage/ never import torch):

   | torch | numpy |
   |---|---|
   | `torch.topk(x, k, dim=-1)` | `np.argpartition(-x, k-1, axis=-1)[..., :k]`, then sort those k if order matters |
   | `t.index_add_(0, idx, v)` | `np.bincount(idx, weights=v, minlength=n)` (much faster than `np.add.at`) |
   | `torch.unique(keys, return_inverse=True)` | `np.unique(keys, return_inverse=True)` |
   | `torch.repeat_interleave(a, counts)` | `np.repeat(a, counts)` |
   | `F.normalize(x, dim=1)` | `x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)` |
   | `torch.Generator().manual_seed(s)` | the portable SplitMix64 generator in `kernels/` (never `np.random` for anything stored in an index) |

   Keep float32 throughout. Watch for silent upcasts to float64: `np.mean` of float16 data, or
   mixing in Python floats.
4. **Keep the teaching style.** smve-lab modules open with a docstring explaining *what* the
   method does and *why*. Keep and adapt it. Every public function gets type hints and a
   docstring stating array shapes and dtypes.
5. **Test** (in `tests/`, mirroring the package path):
   - a slow, obvious reference loop, compared on small random inputs
     (`np.testing.assert_allclose` with `rtol=1e-5`)
   - a hypothesis property test where there's a natural property (order invariance,
     monotonicity, scaling)
   - edge cases: an empty doc, a one-token doc, k ≥ width, a single query
6. **Any deliberate behavior change** (such as the RNG) gets written down in three places: the
   docstring, the progress log's **Decisions**, and a test that pins the new behavior.
7. **Add a row** to the "Ported from smve-lab" table in `docs/dev/progress.md`:
   `smve-lab file → laterank file | what changed`.
