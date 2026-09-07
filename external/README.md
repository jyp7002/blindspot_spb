# external/ — third-party code, cloned rather than vendored

These are other people's repositories. They are **not** committed here: each
carries its own `.git`, so vendoring would nest a foreign history or leave a
gitlink that nothing can resolve. The exact commits used for every published
number are pinned below — clone these and you have what the paper ran against.

```bash
git clone https://github.com/McGill-NLP/bias-bench.git external/bias-bench
git -C external/bias-bench checkout 4bca39b0cddf6d5a81430116aed7729ad3e14eb0

git clone https://github.com/CharlesYu2000/PCGU-UnlearningBias.git external/PCGU-UnlearningBias
git -C external/PCGU-UnlearningBias checkout 407b6e14db75d5353a486957a3338782eee41ac1
```

| repo | commit | used for |
|---|---|---|
| `bias-bench` (McGill-NLP) | `4bca39b0` (2025-08-18) | **SentenceDebias and INLP baselines** — both are ported from this implementation and fitted on our elicited corpora. Cited as `meade2022`. |
| `PCGU-UnlearningBias` | `407b6e14` (2023-11-07) | **Cited, not reimplemented.** Its reference implementation pins a legacy stack (`torch` 1.4, `transformers` 4.10) that cannot be run alongside ours. Kept for reference only; see the paper's "Baselines cited but not run". |

Licenses: `bias-bench` is MIT. Both are used under their own terms.
