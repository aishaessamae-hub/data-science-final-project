"""
Exploratory Data Analysis (EDA) utilities for Ford GoBike dataset.

This module contains reusable functions for performing statistical analysis,
detecting outliers, and generating insights from the data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats


def get_descriptive_statistics(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Get comprehensive descriptive statistics for numeric columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    columns : list, optional
        Specific columns to analyze. If None, uses all numeric columns.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with descriptive statistics.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    stats_dict = {}
    
    for col in columns:
        if col in df.columns:
            stats_dict[col] = {
                'count': df[col].count(),
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'q25': df[col].quantile(0.25),
                'q75': df[col].quantile(0.75),
                'iqr': df[col].quantile(0.75) - df[col].quantile(0.25),
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis()
            }
    
    return pd.DataFrame(stats_dict).T


def detect_outliers_iqr(
    df: pd.DataFrame,
    column: str,
    multiplier: float = 1.5
) -> Dict[str, Union[int, float, pd.Series]]:
    """
    Detect outliers using IQR (Interquartile Range) method.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Column to analyze.
    multiplier : float, default 1.5
        IQR multiplier for outlier detection.
    
    Returns:
    --------
    dict
        Dictionary containing outlier information:
        - 'lower_bound': Lower outlier bound
        - 'upper_bound': Upper outlier bound
        - 'outliers': Boolean series marking outliers
        - 'n_outliers': Number of outliers
        - 'outlier_percentage': Percentage of outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    n_outliers = outliers.sum()
    outlier_percentage = (n_outliers / len(df)) * 100
    
    return {
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'outliers': outliers,
        'n_outliers': n_outliers,
        'outlier_percentage': outlier_percentage,
        'outlier_values': df[outliers][column].tolist()
    }


def detect_outliers_zscore(
    df: pd.DataFrame,
    column: str,
    threshold: float = 3.0
) -> Dict[str, Union[int, float, pd.Series]]:
    """
    Detect outliers using Z-score method.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Column to analyze.
    threshold : float, default 3.0
        Z-score threshold for outlier detection.
    
    Returns:
    --------
    dict
        Dictionary containing outlier information.
    """
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    outliers_idx = z_scores > threshold
    
    outliers = pd.Series(False, index=df.index)
    outliers.iloc[df[column].dropna().index[outliers_idx]] = True
    
    n_outliers = outliers.sum()
    outlier_percentage = (n_outliers / len(df)) * 100
    
    return {
        'threshold': threshold,
        'outliers': outliers,
        'n_outliers': n_outliers,
        'outlier_percentage': outlier_percentage,
        'outlier_values': df[outliers][column].tolist()
    }


def compute_correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    Compute correlation matrix for numeric columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    columns : list, optional
        Specific columns to include. If None, uses all numeric columns.
    method : str, default 'pearson'
        Correlation method: 'pearson', 'spearman', or 'kendall'.
    
    Returns:
    --------
    pd.DataFrame
        Correlation matrix.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    return df[columns].corr(method=method)


def get_categorical_summary(
    df: pd.DataFrame,
    column: str
) -> Dict:
    """
    Get summary statistics for categorical column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Categorical column name.
    
    Returns:
    --------
    dict
        Dictionary with categorical statistics.
    """
    value_counts = df[column].value_counts()
    value_percentages = df[column].value_counts(normalize=True) * 100
    
    return {
        'unique_values': df[column].nunique(),
        'most_common': value_counts.index[0],
        'most_common_count': value_counts.iloc[0],
        'most_common_percentage': value_percentages.iloc[0],
        'value_counts': value_counts.to_dict(),
        'value_percentages': value_percentages.to_dict(),
        'missing_count': df[column].isna().sum(),
        'missing_percentage': (df[column].isna().sum() / len(df)) * 100
    }


def compare_groups(
    df: pd.DataFrame,
    numeric_col: str,
    group_col: str
) -> pd.DataFrame:
    """
    Compare numeric variable statistics across groups.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    numeric_col : str
        Numeric column to analyze.
    group_col : str
        Categorical column for grouping.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with group statistics.
    """
    stats = df.groupby(group_col)[numeric_col].agg([
        'count',
        'mean',
        'median',
        'std',
        'min',
        'max',
        ('q25', lambda x: x.quantile(0.25)),
        ('q75', lambda x: x.quantile(0.75))
    ]).round(2)
    
    return stats


def find_correlations_above_threshold(
    df: pd.DataFrame,
    threshold: float = 0.7,
    columns: Optional[List[str]] = None
) -> List[Tuple[str, str, float]]:
    """
    Find pairs of variables with correlation above threshold.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    threshold : float, default 0.7
        Minimum absolute correlation value.
    columns : list, optional
        Specific columns to check.
    
    Returns:
    --------
    list of tuples
        List of (var1, var2, correlation) tuples.
    """
    corr_matrix = compute_correlation_matrix(df, columns)
    
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            var1 = corr_matrix.columns[i]
            var2 = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]
            
            if abs(corr_value) >= threshold:
                high_corr.append((var1, var2, corr_value))
    
    high_corr.sort(key=lambda x: abs(x[2]), reverse=True)
    
    return high_corr


def get_missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary of missing values in dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with missing value statistics.
    """
    missing_stats = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isna().sum().values,
        'missing_percentage': (df.isna().sum() / len(df) * 100).values,
        'dtype': df.dtypes.values
    })
    
    missing_stats = missing_stats[missing_stats['missing_count'] > 0].sort_values(
        'missing_percentage', ascending=False
    )
    
    return missing_stats


def analyze_distribution(
    df: pd.DataFrame,
    column: str
) -> Dict:
    """
    Comprehensive distribution analysis for numeric column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Numeric column to analyze.
    
    Returns:
    --------
    dict
        Dictionary with distribution analysis.
    """
    data = df[column].dropna()
    
    if len(data) < 5000:
        shapiro_stat, shapiro_p = stats.shapiro(data)
    else:
        sample = data.sample(n=5000, random_state=42)
        shapiro_stat, shapiro_p = stats.shapiro(sample)
    
    return {
        'count': len(data),
        'mean': data.mean(),
        'median': data.median(),
        'mode': data.mode()[0] if len(data.mode()) > 0 else None,
        'std': data.std(),
        'variance': data.var(),
        'skewness': data.skew(),
        'kurtosis': data.kurtosis(),
        'min': data.min(),
        'max': data.max(),
        'range': data.max() - data.min(),
        'cv': (data.std() / data.mean()) * 100 if data.mean() != 0 else None,
        'shapiro_statistic': shapiro_stat,
        'shapiro_p_value': shapiro_p,
        'is_normal': shapiro_p > 0.05,
        'percentiles': {
            '1%': data.quantile(0.01),
            '5%': data.quantile(0.05),
            '10%': data.quantile(0.10),
            '25%': data.quantile(0.25),
            '50%': data.quantile(0.50),
            '75%': data.quantile(0.75),
            '90%': data.quantile(0.90),
            '95%': data.quantile(0.95),
            '99%': data.quantile(0.99)
        }
    }



def get_time_based_statistics(
    df: pd.DataFrame,
    datetime_col: str,
    value_col: str,
    freq: str = 'D'
) -> pd.DataFrame:
    """
    Get statistics aggregated by time period.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    datetime_col : str
        Datetime column name.
    value_col : str
        Numeric column to aggregate.
    freq : str, default 'D'
        Frequency for aggregation ('D', 'W', 'M', 'Y', etc.).
    
    Returns:
    --------
    pd.DataFrame
        Time-based statistics.
    """
    df_copy = df.copy()
    df_copy[datetime_col] = pd.to_datetime(df_copy[datetime_col])
    df_copy = df_copy.set_index(datetime_col)
    
    stats = df_copy[value_col].resample(freq).agg([
        'count',
        'mean',
        'median',
        'std',
        'min',
        'max'
    ])
    
    return stats


def identify_data_quality_issues(df: pd.DataFrame) -> Dict:
    """
    Identify various data quality issues in the dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    
    Returns:
    --------
    dict
        Dictionary with data quality issues.
    """
    issues = {
        'missing_values': df.isna().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'duplicate_percentage': (df.duplicated().sum() / len(df) * 100),
        'columns_with_single_value': [],
        'columns_with_all_nulls': [],
        'columns_with_high_cardinality': []
    }
    
    for col in df.columns:
        if df[col].nunique() == 1:
            issues['columns_with_single_value'].append(col)
        
        if df[col].isna().all():
            issues['columns_with_all_nulls'].append(col)
        
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            if df[col].nunique() / len(df) > 0.5:
                issues['columns_with_high_cardinality'].append(col)
    
    return issues


def get_percentile_ranges(
    df: pd.DataFrame,
    column: str,
    percentiles: List[int] = [10, 25, 50, 75, 90, 95, 99]
) -> Dict[str, float]:
    """
    Get percentile values for a numeric column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Numeric column name.
    percentiles : list, default [10, 25, 50, 75, 90, 95, 99]
        List of percentile values to compute.
    
    Returns:
    --------
    dict
        Dictionary mapping percentile labels to values.
    """
    percentile_dict = {}
    for p in percentiles:
        percentile_dict[f'P{p}'] = df[column].quantile(p / 100)
    
    return percentile_dict
