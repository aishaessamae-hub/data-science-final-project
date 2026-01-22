"""
General data preprocessing utilities.

This module contains reusable functions for loading, cleaning, and processing
tabular data in pandas DataFrames.
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Union


def load_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load CSV data from the specified filepath.

    Parameters:
    -----------
    filepath : str or Path
        Path to the CSV file to load.

    Returns:
    --------
    pd.DataFrame
        Loaded DataFrame.
    """
    return pd.read_csv(filepath)


def get_data_info(df: pd.DataFrame) -> Dict:
    """
    Get basic information about the DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to inspect.

    Returns:
    --------
    dict
        Dictionary containing shape, columns, and head of DataFrame.
    """
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "head": df.head(),
    }


def get_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get data types of all columns in the DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to check.

    Returns:
    --------
    pd.DataFrame
        DataFrame with column names and their data types.
    """
    return pd.DataFrame({"dtype": df.dtypes})


def fill_missing_categorical(
    df: pd.DataFrame, columns: List[str], fill_value: str = "Unknown"
) -> pd.DataFrame:
    """
    Fill missing values in categorical columns with a specified value.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to process.
    columns : list of str
        List of column names to fill.
    fill_value : str, default 'Unknown'
        Value to use for filling missing values.

    Returns:
    --------
    pd.DataFrame
        DataFrame with missing categorical values filled.
    """
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(fill_value)
    return df_clean


def fill_missing_numeric(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "median",
    fill_value: Optional[float] = None,
) -> pd.DataFrame:
    """
    Fill missing values in numeric columns using specified method.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to process.
    columns : list of str
        List of column names to fill.
    method : str, default 'median'
        Method to use: 'mean', 'median', or 'value'.
        If 'value', fill_value must be provided.
    fill_value : float, optional
        Value to use when method is 'value'.

    Returns:
    --------
    pd.DataFrame
        DataFrame with missing numeric values filled.
    """
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            if method == "mean":
                fill_val = df_clean[col].mean()
            elif method == "median":
                fill_val = df_clean[col].median()
            elif method == "value" and fill_value is not None:
                fill_val = fill_value
            else:
                raise ValueError(
                    f"Invalid method '{method}' or missing fill_value for method 'value'"
                )
            df_clean[col] = df_clean[col].fillna(fill_val)
    return df_clean


def fill_missing_with_mapping(
    df: pd.DataFrame, column_mapping: Dict[str, Union[str, float, int]]
) -> pd.DataFrame:
    """
    Fill missing values in specified columns with custom values.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to process.
    column_mapping : dict
        Dictionary mapping column names to fill values.

    Returns:
    --------
    pd.DataFrame
        DataFrame with missing values filled according to mapping.
    """
    df_clean = df.copy()
    for col, fill_val in column_mapping.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(fill_val)
    return df_clean


def convert_to_category(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Convert specified columns to category type.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to process.
    columns : list of str
        List of column names to convert.

    Returns:
    --------
    pd.DataFrame
        DataFrame with specified columns converted to category type.
    """
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype("category")
    return df_clean


def convert_to_int(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Convert specified columns to integer type.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to process.
    columns : list of str
        List of column names to convert.

    Returns:
    --------
    pd.DataFrame
        DataFrame with specified columns converted to integer type.
    """
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(int)
    return df_clean


def calculate_age_from_year(
    df: pd.DataFrame, birth_year_col: str, age_col: str = "age"
) -> pd.DataFrame:
    """
    Calculate age from birth year column.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to process.
    birth_year_col : str
        Name of the birth year column.
    age_col : str, default 'age'
        Name of the new age column to create.

    Returns:
    --------
    pd.DataFrame
        DataFrame with age column added.
    """
    df_clean = df.copy()
    if birth_year_col in df_clean.columns:
        current_year = datetime.now().year
        df_clean[age_col] = (current_year - df_clean[birth_year_col]).astype(int)
    return df_clean


def convert_duration(
    df: pd.DataFrame,
    duration_sec_col: str,
    minutes_col: Optional[str] = None,
    hours_col: Optional[str] = None,
    decimals: int = 2,
) -> pd.DataFrame:
    """
    Convert duration from seconds to minutes and/or hours.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to process.
    duration_sec_col : str
        Name of the duration in seconds column.
    minutes_col : str, optional
        Name of the new minutes column. If None, not created.
    hours_col : str, optional
        Name of the new hours column. If None, not created.
    decimals : int, default 2
        Number of decimal places for rounding.

    Returns:
    --------
    pd.DataFrame
        DataFrame with duration columns added.
    """
    df_clean = df.copy()
    if duration_sec_col not in df_clean.columns:
        return df_clean

    if minutes_col:
        df_clean[minutes_col] = (df_clean[duration_sec_col] / 60).round(decimals)

    if hours_col:
        df_clean[hours_col] = (df_clean[duration_sec_col] / 3600).round(decimals)

    return df_clean


def flag_extreme_values(
    df: pd.DataFrame,
    column: str,
    flag_col: str,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
) -> pd.DataFrame:
    """
    Flag rows with extreme values in a specified column.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to process.
    column : str
        Name of the column to check.
    flag_col : str
        Name of the new boolean flag column.
    lower_bound : float, optional
        Lower bound for extreme values (values below this are flagged).
    upper_bound : float, optional
        Upper bound for extreme values (values above this are flagged).

    Returns:
    --------
    pd.DataFrame
        DataFrame with flag column added.
    """
    df_clean = df.copy()
    if column not in df_clean.columns:
        return df_clean

    mask = pd.Series(False, index=df_clean.index)
    if lower_bound is not None:
        mask = mask | (df_clean[column] < lower_bound)
    if upper_bound is not None:
        mask = mask | (df_clean[column] > upper_bound)

    df_clean[flag_col] = mask
    return df_clean


def cap_outliers(
    df: pd.DataFrame,
    column: str,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    method: str = "median",
) -> pd.DataFrame:
    """
    Cap outliers in a numeric column to specified bounds.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to process.
    column : str
        Name of the column to process.
    lower_bound : float, optional
        Lower bound for capping.
    upper_bound : float, optional
        Upper bound for capping.
    method : str, default 'median'
        Method to replace outliers: 'median', 'mean', or 'bound'.
        If 'bound', values are capped to the bounds.

    Returns:
    --------
    pd.DataFrame
        DataFrame with outliers capped.
    """
    df_clean = df.copy()
    if column not in df_clean.columns:
        return df_clean

    outlier_mask = pd.Series(False, index=df_clean.index)
    if lower_bound is not None:
        outlier_mask = outlier_mask | (df_clean[column] < lower_bound)
    if upper_bound is not None:
        outlier_mask = outlier_mask | (df_clean[column] > upper_bound)

    if outlier_mask.sum() > 0:
        if method == "median":
            replacement_value = df_clean[column].median()
        elif method == "mean":
            replacement_value = df_clean[column].mean()
        elif method == "bound":
            pass
        else:
            raise ValueError(f"Invalid method: {method}")

        if method == "bound":
            if lower_bound is not None:
                df_clean.loc[df_clean[column] < lower_bound, column] = lower_bound
            if upper_bound is not None:
                df_clean.loc[df_clean[column] > upper_bound, column] = upper_bound
        else:
            df_clean.loc[outlier_mask, column] = replacement_value

    return df_clean


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to process.

    Returns:
    --------
    pd.DataFrame
        DataFrame with duplicates removed.
    """
    return df.drop_duplicates()


def export_data(
    df: pd.DataFrame, output_dir: Union[str, Path], filename: str = "processed_data.csv"
) -> str:
    """
    Export the processed DataFrame to CSV.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to export.
    output_dir : str or Path
        Directory where the CSV file will be saved.
    filename : str, default 'processed_data.csv'
        Name of the output CSV file.

    Returns:
    --------
    str
        Path to the exported file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    df.to_csv(output_path, index=False)

    return str(output_path)
