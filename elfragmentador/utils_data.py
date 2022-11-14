import logging

import numpy as np
import uniplot


def terminal_plot_similarity(similarities, name=""):
    if all([np.isnan(x) for x in similarities]):
        logging.warning("Skipping because all values are missing")
        return None

    uniplot.histogram(
        similarities,
        title=f"{name} mean:{similarities.mean()}",
    )

    qs = [0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1]
    similarity_quantiles = np.quantile(1 - similarities, qs)
    p90 = similarity_quantiles[2]
    p10 = similarity_quantiles[-3]
    q1 = similarity_quantiles[5]
    med = similarity_quantiles[4]
    q3 = similarity_quantiles[3]
    title = f"Accumulative distribution (y) of the 1 - {name} (x)"
    title += f"\nP90={1-p90:.3f} Q3={1-q3:.3f}"
    title += f" Median={1-med:.3f} Q1={1-q1:.3f} P10={1-p10:.3f}"
    uniplot.plot(xs=similarity_quantiles, ys=qs, lines=True, title=title)
