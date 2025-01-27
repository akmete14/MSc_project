import pandas as pd
import numpy as np
from tqdm import tqdm

df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups.csv')
df = df.drop(columns=['Unnamed: 0'])

def balance_data_by_site(
    df,
    site_col='site_id',
    target_col=None,
    lower_pct=None,
    upper_pct=None,
    sample_size=None,
    random_state=None
):
    """
    Balances a DataFrame so that each site has the same number of rows.
    Optionally removes outliers by cutting at lower/upper percentiles of a target column.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing all data (including site_id).
    site_col : str
        Name of the column indicating site IDs.
    target_col : str or None
        If not None, we'll compute quantiles on this column for outlier removal.
    lower_pct : float or None
        Lower percentile for filtering. Example: 0.05 (5% cutoff).
        If None, no lower cutoff is applied.
    upper_pct : float or None
        Upper percentile for filtering. Example: 0.95 (95% cutoff).
        If None, no upper cutoff is applied.
    sample_size : int or None
        The number of samples to keep from each site.
        If None, we use the minimum group size after any outlier filtering.
    random_state : int or None
        Seed for reproducible sampling.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the same number of rows for each site,
        optionally filtered at specified percentiles.
    """

    # Group by site
    grouped = df.groupby(site_col)

    # Prepare a list to hold balanced data for each site
    balanced_chunks = []

    # If sample_size is None, we’ll determine it after filtering
    if sample_size is None:
        # We will compute sample_size as the min group size (post-filtering) across all sites
        # but we can only do that after we know each group’s size.
        # Therefore, let's do a two-pass approach.
        group_sizes = []
        for site_name, group in tqdm(grouped):
            filtered_group = _apply_percentile_filter(group, target_col, lower_pct, upper_pct)
            group_sizes.append(len(filtered_group))
        min_size = min(group_sizes)
        # Use that minimum size for uniform sampling
        sample_size = min_size if min_size > 0 else 0

    # Now process each group again with the final sample_size
    for site_name, group in tqdm(grouped):
        filtered_group = _apply_percentile_filter(group, target_col, lower_pct, upper_pct)
        # Sample from the filtered group, if it has enough rows
        if len(filtered_group) >= sample_size:
            sampled_group = filtered_group.sample(n=sample_size, random_state=random_state)
            balanced_chunks.append(sampled_group)
        else:
            # If not enough data left, you could either discard it or skip it
            # balanced_chunks.append(filtered_group)  # or skip
            pass

    # Concatenate all balanced groups
    balanced_df = pd.concat(balanced_chunks, ignore_index=True)
    return balanced_df


def _apply_percentile_filter(group, target_col, lower_pct, upper_pct):
    """Helper function to filter a group at specified percentiles of target_col."""
    if target_col is not None and (lower_pct is not None or upper_pct is not None):
        # Compute quantiles only if we have a target_col
        if lower_pct is not None:
            lower_bound = group[target_col].quantile(lower_pct)
            group = group[group[target_col] >= lower_bound]
        if upper_pct is not None:
            upper_bound = group[target_col].quantile(upper_pct)
            group = group[group[target_col] <= upper_bound]
    return group


# Suppose df is your DataFrame containing [target, features, site_id, cluster]
df_balanced = balance_data_by_site(
    df,
    site_col='site_id',
    target_col='GPP',      # for percentile filtering
    lower_pct=0.05,           # cut lower 5% outliers
    upper_pct=0.95,           # cut upper 5% outliers
    sample_size=100000,         # keep 1000 rows per site
    random_state=42
)

df_balanced.to_csv('balanced_data.csv', index=False)
