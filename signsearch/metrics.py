import numpy as np


def retrieval_metrics(logits, labels, signers, seed=0, resamples=1000):
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    order = np.argsort(-logits, axis=1, kind="stable")
    ranks = np.argmax(order == labels[:, None], axis=1) + 1
    correct = ranks == 1
    signer_rows = [{"signer": str(s), "clips": int(np.sum(np.asarray(signers) == s)),
                    "top1": float(correct[np.asarray(signers) == s].mean())} for s in sorted(set(signers))]
    signer_values = [s["top1"] for s in signer_rows]
    rng = np.random.default_rng(seed)
    bootstrap = np.array([correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(resamples)])
    return {"top1": float(correct.mean()), "top5": float((ranks <= 5).mean()), "mrr": float((1 / ranks).mean()),
            "clips": len(labels), "signers": len(signer_rows),
            "per_signer": {"min": float(np.min(signer_values)), "median": float(np.median(signer_values)), "max": float(np.max(signer_values)), "full_list": signer_rows},
            "top1_bootstrap_95_ci": [float(v) for v in np.quantile(bootstrap, [.025,.975])],
            "bootstrap_resamples": resamples, "bootstrap_seed": seed}
