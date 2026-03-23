#!/usr/bin/env python3
"""Statistics helpers

Authors
 * Artem Ploujnikov 2026
"""

import torch


def descriptive_statistics(
    items: list,
    key: str | None = None,
    result_key: str | None = None
) -> dict:
    """Computes descriptive statistics for the summary

    Arguments
    ---------
    items : list
        a list of dictionaries with metric values for each item
    key : str | None
        The key of the metric for which the statistics will be computed
    result_key : str | None
        The key to use for results

    Returns
    -------
    statistics : dict
        The desccriptive statistics computed
            <result_key>_mean : the arithmetic mean
            <result_key>_std : the standard deviation
            <result_key>_min : the minimum value
            <result_key>_max : the maximum value
            <result_key>_median : the median value
            <result_key>_q1 : the first quartile
            <result_key>_q3 : the third quartile
            <result_key>_iqr : the interquartile ratio
    """
    if not items:
        return {}
    if not result_key:
        result_key = key
    if key is None:
        if isinstance(items[0], dict):
            keys = items[0].keys()
            return {
                stat_key: value
                for key in keys
                for stat_key, value in descriptive_statistics(
                    items, key, key
                ).items()
            }
        else:
            values = torch.tensor(items)
    else:
        values = torch.tensor([item[key] for item in items])
    quantiles = torch.tensor([0.25, 0.5, 0.75])
    q1, median, q3 = values.quantile(quantiles)
    stats = {
        "mean": values.mean(),
        "std": values.std(),
        "min": values.min(),
        "max": values.max(),
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
    }
    return {
        f"{result_key}_{stat_key}": value.item()
        for stat_key, value in stats.items()
    }
