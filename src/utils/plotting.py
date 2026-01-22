"""
Advanced plotting utilities for Ford GoBike analysis.

This module contains highly reusable Plotly Express functions designed for
both Jupyter notebook analysis and Dash application integration.
All functions return Plotly Figure objects.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Union


DEFAULT_COLOR_DISCRETE = px.colors.qualitative.Set2
DEFAULT_COLOR_CONTINUOUS = px.colors.sequential.Viridis


def plot_categorical_distribution(
    df: pd.DataFrame,
    column: str,
    title: Optional[str] = None,
    color_discrete_sequence: Optional[List[str]] = None,
    show_percentages: bool = True,
    orientation: str = 'v',
    height: int = 500,
    template: str = 'plotly_white'
) -> go.Figure:
    """
    Create a bar chart showing distribution of categorical variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Column name to visualize.
    title : str, optional
        Chart title. Auto-generated if None.
    color_discrete_sequence : list, optional
        Color sequence for bars.
    show_percentages : bool, default True
        Show percentages as annotations.
    orientation : str, default 'v'
        'v' for vertical, 'h' for horizontal.
    height : int, default 500
        Figure height in pixels.
    template : str, default 'plotly_white'
        Plotly theme template.
    
    Returns:
    --------
    go.Figure
        Plotly figure object ready for display or Dash integration.
    """
    if title is None:
        title = f"Distribution of {column.replace('_', ' ').title()}"
    
    if color_discrete_sequence is None:
        color_discrete_sequence = DEFAULT_COLOR_DISCRETE
    
    value_counts = df[column].value_counts().reset_index()
    value_counts.columns = [column, 'count']
    value_counts['percentage'] = (value_counts['count'] / value_counts['count'].sum() * 100).round(2)
    
    if show_percentages:
        value_counts['text_label'] = value_counts.apply(
            lambda row: f"{row['count']:,}<br>({row['percentage']:.1f}%)", axis=1
        )
    else:
        value_counts['text_label'] = value_counts['count'].apply(lambda x: f"{x:,}")
    
    if orientation == 'v':
        fig = px.bar(
            value_counts,
            x=column,
            y='count',
            title=title,
            color=column,
            color_discrete_sequence=color_discrete_sequence,
            text='text_label',
            labels={'count': 'Frequency', column: column.replace('_', ' ').title()}
        )
    else:
        fig = px.bar(
            value_counts,
            x='count',
            y=column,
            title=title,
            color=column,
            color_discrete_sequence=color_discrete_sequence,
            text='text_label',
            orientation='h',
            labels={'count': 'Frequency', column: column.replace('_', ' ').title()}
        )
    
    fig.update_traces(textposition='outside')
    
    if orientation == 'v':
        max_count = value_counts['count'].max()
        fig.update_yaxes(
            range=[0, max_count * 1.15],  
            title_text='Frequency'
        )
    else:
        max_count = value_counts['count'].max()
        fig.update_xaxes(
            range=[0, max_count * 1.15],
            title_text='Frequency'
        )
    
    fig.update_layout(
        height=height,
        template=template,
        showlegend=False,
        hovermode='closest'
    )
    
    return fig


def plot_numeric_distribution(
    df: pd.DataFrame,
    column: str,
    title: Optional[str] = None,
    nbins: int = 50,
    color: str = '#636EFA',
    height: int = 500,
    show_mean: bool = True,
    show_median: bool = True,
    template: str = 'plotly_white'
) -> go.Figure:
    """
    Create a histogram with optional mean/median lines for numeric variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Numeric column name.
    title : str, optional
        Chart title.
    nbins : int, default 50
        Number of bins.
    color : str, default '#636EFA'
        Bar color.
    height : int, default 500
        Figure height.
    show_mean : bool, default True
        Show vertical line for mean.
    show_median : bool, default True
        Show vertical line for median.
    template : str, default 'plotly_white'
        Plotly theme template.
    
    Returns:
    --------
    go.Figure
        Plotly figure object.
    """
    if title is None:
        title = f"Distribution of {column.replace('_', ' ').title()}"
    
    fig = px.histogram(
        df,
        x=column,
        nbins=nbins,
        title=title,
        labels={column: column.replace('_', ' ').title()},
        color_discrete_sequence=[color],
        marginal='box'
)
    
    if show_mean:
        mean_val = df[column].mean()
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_val:.2f}",
            annotation_position="top"
        )
    
    if show_median:
        median_val = df[column].median()
        fig.add_vline(
            x=median_val,
            line_dash="dot",
            line_color="green",
            annotation_text=f"Median: {median_val:.2f}",
            annotation_position="bottom right"
        )
    
    data_min = df[column].min()
    data_max = df[column].max()
    data_range = data_max - data_min
    
    fig.update_xaxes(
        range=[data_min - 0.05 * data_range, data_max + 0.05 * data_range],
        title_text=column.replace('_', ' ').title()
    )
    
    fig.update_yaxes(
        rangemode='tozero',  
        title_text='Frequency'
    )
    
    fig.update_layout(
        height=height,
        template=template,
        showlegend=False
    )
    
    return fig


def plot_boxplot_by_category(
    df: pd.DataFrame,
    numeric_col: str,
    category_col: str,
    title: Optional[str] = None,
    color_discrete_sequence: Optional[List[str]] = None,
    points: str = 'outliers',
    height: int = 500,
    template: str = 'plotly_white'
) -> go.Figure:
    """
    Create box plot showing numeric distribution across categories.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    numeric_col : str
        Numeric column for y-axis.
    category_col : str
        Categorical column for x-axis.
    title : str, optional
        Chart title.
    color_discrete_sequence : list, optional
        Color sequence.
    points : str, default 'outliers'
        'outliers', 'all', or False.
    height : int, default 500
        Figure height.
    template : str, default 'plotly_white'
        Plotly theme.
    
    Returns:
    --------
    go.Figure
        Plotly figure object.
    """
    if title is None:
        title = f"{numeric_col.replace('_', ' ').title()} by {category_col.replace('_', ' ').title()}"
    
    if color_discrete_sequence is None:
        color_discrete_sequence = DEFAULT_COLOR_DISCRETE
    
    fig = px.box(
        df,
        x=category_col,
        y=numeric_col,
        title=title,
        color=category_col,
        color_discrete_sequence=color_discrete_sequence,
        points=points,
        labels={
            numeric_col: numeric_col.replace('_', ' ').title(),
            category_col: category_col.replace('_', ' ').title()
        }
    )
    
    data_min = df[numeric_col].min()
    data_max = df[numeric_col].max()
    data_range = data_max - data_min
    Q1 = df[numeric_col].quantile(0.25)
    Q3 = df[numeric_col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_whisker = Q1 - 1.5 * IQR
    upper_whisker = Q3 + 1.5 * IQR
    
    fig.update_yaxes(
        range=[min(data_min, lower_whisker) - 0.1 * data_range,
               max(data_max, upper_whisker) + 0.1 * data_range],
        title_text=numeric_col.replace('_', ' ').title()
    )
    
    fig.update_layout(
        height=height,
        template=template,
        showlegend=False
    )
    
    return fig


def plot_violin_by_category(
    df: pd.DataFrame,
    numeric_col: str,
    category_col: str,
    title: Optional[str] = None,
    color_discrete_sequence: Optional[List[str]] = None,
    box: bool = True,
    points: str = 'outliers',
    height: int = 500,
    template: str = 'plotly_white'
) -> go.Figure:
    """
    Create violin plot showing detailed distribution across categories.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    numeric_col : str
        Numeric column for y-axis.
    category_col : str
        Categorical column for grouping.
    title : str, optional
        Chart title.
    color_discrete_sequence : list, optional
        Color sequence.
    box : bool, default True
        Show box plot inside violin.
    points : str, default 'outliers'
        'outliers', 'all', or False.
    height : int, default 500
        Figure height.
    template : str, default 'plotly_white'
        Plotly theme.
    
    Returns:
    --------
    go.Figure
        Plotly figure object.
    """
    if title is None:
        title = f"{numeric_col.replace('_', ' ').title()} Distribution by {category_col.replace('_', ' ').title()}"
    
    if color_discrete_sequence is None:
        color_discrete_sequence = DEFAULT_COLOR_DISCRETE
    
    fig = px.violin(
        df,
        x=category_col,
        y=numeric_col,
        title=title,
        color=category_col,
        color_discrete_sequence=color_discrete_sequence,
        box=box,
        points=points,
        labels={
            numeric_col: numeric_col.replace('_', ' ').title(),
            category_col: category_col.replace('_', ' ').title()
        }
    )
    
    data_min = df[numeric_col].min()
    data_max = df[numeric_col].max()
    data_range = data_max - data_min
    
    fig.update_yaxes(
        range=[data_min - 0.1 * data_range, data_max + 0.1 * data_range],
        title_text=numeric_col.replace('_', ' ').title()
    )
    
    fig.update_layout(
        height=height,
        template=template,
        showlegend=False
    )
    
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    title: str = "Correlation Heatmap",
    color_scale: str = 'RdBu_r',
    height: int = 700,
    width: int = 700,
    template: str = 'plotly_white'
) -> go.Figure:
    """
    Create correlation heatmap for numeric columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    columns : list, optional
        Specific columns to include. If None, uses all numeric columns.
    title : str, default 'Correlation Heatmap'
        Chart title.
    color_scale : str, default 'RdBu_r'
        Plotly color scale.
    height : int, default 700
        Figure height.
    width : int, default 700
        Figure width.
    template : str, default 'plotly_white'
        Plotly theme.
    
    Returns:
    --------
    go.Figure
        Plotly figure object.
    """
    if columns is None:
        corr_data = df.select_dtypes(include=[np.number]).corr()
    else:
        corr_data = df[columns].corr()
    
    fig = px.imshow(
        corr_data,
        title=title,
        color_continuous_scale=color_scale,
        aspect='auto',
        text_auto='.2f',
        labels=dict(color="Correlation"),
        zmin=-1,
        zmax=1
    )
    
    fig.update_layout(
        height=height,
        width=width,
        template=template
    )
    
    return fig


def plot_scatter_with_regression(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    size: Optional[str] = None,
    title: Optional[str] = None,
    trendline: bool = True,
    height: int = 600,
    template: str = 'plotly_white'
) -> go.Figure:
    """
    Create scatter plot with optional regression line.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    x : str
        X-axis column.
    y : str
        Y-axis column.
    color : str, optional
        Column for color coding points.
    size : str, optional
        Column for sizing points.
    title : str, optional
        Chart title.
    trendline : bool, default True
        Add OLS regression line.
    height : int, default 600
        Figure height.
    template : str, default 'plotly_white'
        Plotly theme.
    
    Returns:
    --------
    go.Figure
        Plotly figure object.
    """
    if title is None:
        title = f"{y.replace('_', ' ').title()} vs {x.replace('_', ' ').title()}"
    
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        size=size,
        title=title,
        trendline='ols' if trendline else None,
        labels={
            x: x.replace('_', ' ').title(),
            y: y.replace('_', ' ').title()
        },
        color_discrete_sequence=DEFAULT_COLOR_DISCRETE
    )
    
    x_min, x_max = df[x].min(), df[x].max()
    y_min, y_max = df[y].min(), df[y].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    fig.update_xaxes(
        range=[x_min - 0.05 * x_range, x_max + 0.05 * x_range],
        title_text=x.replace('_', ' ').title()
    )
    
    fig.update_yaxes(
        range=[y_min - 0.05 * y_range, y_max + 0.05 * y_range],
        title_text=y.replace('_', ' ').title()
    )
    
    fig.update_layout(
        height=height,
        template=template
    )
    
    return fig


def plot_pie_chart(
    df: pd.DataFrame,
    column: str,
    title: Optional[str] = None,
    hole: float = 0.3,
    height: int = 500,
    template: str = 'plotly_white'
) -> go.Figure:
    """
    Create pie/donut chart for categorical distribution.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Categorical column.
    title : str, optional
        Chart title.
    hole : float, default 0.3
        Size of donut hole (0 = pie, 0.3-0.5 = donut).
    height : int, default 500
        Figure height.
    template : str, default 'plotly_white'
        Plotly theme.
    
    Returns:
    --------
    go.Figure
        Plotly figure object.
    """
    if title is None:
        title = f"Distribution of {column.replace('_', ' ').title()}"
    
    value_counts = df[column].value_counts().reset_index()
    value_counts.columns = [column, 'count']
    
    fig = px.pie(
        value_counts,
        names=column,
        values='count',
        title=title,
        hole=hole,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )
    
    fig.update_layout(
        height=height,
        template=template
    )
    
    return fig


def plot_grouped_bar_chart(
    df: pd.DataFrame,
    category_col: str,
    numeric_col: str,
    aggregation: str = 'mean',
    title: Optional[str] = None,
    color_discrete_sequence: Optional[List[str]] = None,
    height: int = 500,
    template: str = 'plotly_white'
) -> go.Figure:
    """
    Create grouped bar chart with aggregated values.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    category_col : str
        Column to group by.
    numeric_col : str
        Numeric column to aggregate.
    aggregation : str, default 'mean'
        Aggregation function: 'mean', 'median', 'sum', 'count', etc.
    title : str, optional
        Chart title.
    color_discrete_sequence : list, optional
        Color sequence.
    height : int, default 500
        Figure height.
    template : str, default 'plotly_white'
        Plotly theme.
    
    Returns:
    --------
    go.Figure
        Plotly figure object.
    """
    if title is None:
        title = f"{aggregation.title()} {numeric_col.replace('_', ' ').title()} by {category_col.replace('_', ' ').title()}"
    
    if color_discrete_sequence is None:
        color_discrete_sequence = DEFAULT_COLOR_DISCRETE
    
    agg_data = df.groupby(category_col)[numeric_col].agg(aggregation).reset_index()
    agg_data.columns = [category_col, f'{aggregation}_{numeric_col}']
    
    fig = px.bar(
        agg_data,
        x=category_col,
        y=f'{aggregation}_{numeric_col}',
        title=title,
        color=category_col,
        color_discrete_sequence=color_discrete_sequence,
        text=f'{aggregation}_{numeric_col}',
        labels={
            f'{aggregation}_{numeric_col}': f'{aggregation.title()} {numeric_col.replace("_", " ").title()}',
            category_col: category_col.replace('_', ' ').title()
        }
    )
    
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    
    max_val = agg_data[f'{aggregation}_{numeric_col}'].max()
    fig.update_yaxes(
        range=[0, max_val * 1.15],
        title_text=f'{aggregation.title()} {numeric_col.replace("_", " ").title()}'
    )
    
    fig.update_layout(
        height=height,
        template=template,
        showlegend=False
    )
    
    return fig


def plot_stacked_histogram(
    df: pd.DataFrame,
    numeric_col: str,
    category_col: str,
    nbins: int = 30,
    title: Optional[str] = None,
    barmode: str = 'overlay',
    opacity: float = 0.7,
    height: int = 500,
    template: str = 'plotly_white'
) -> go.Figure:
    """
    Create overlaid or stacked histograms by category.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    numeric_col : str
        Numeric column to plot.
    category_col : str
        Categorical column for grouping.
    nbins : int, default 30
        Number of bins.
    title : str, optional
        Chart title.
    barmode : str, default 'overlay'
        'overlay', 'stack', or 'group'.
    opacity : float, default 0.7
        Bar opacity (0-1).
    height : int, default 500
        Figure height.
    template : str, default 'plotly_white'
        Plotly theme.
    
    Returns:
    --------
    go.Figure
        Plotly figure object.
    """
    if title is None:
        title = f"{numeric_col.replace('_', ' ').title()} Distribution by {category_col.replace('_', ' ').title()}"
    
    fig = px.histogram(
        df,
        x=numeric_col,
        color=category_col,
        nbins=nbins,
        title=title,
        barmode=barmode,
        opacity=opacity,
        color_discrete_sequence=DEFAULT_COLOR_DISCRETE,
        labels={
            numeric_col: numeric_col.replace('_', ' ').title(),
            category_col: category_col.replace('_', ' ').title()
        }
    )
    
    data_min = df[numeric_col].min()
    data_max = df[numeric_col].max()
    data_range = data_max - data_min
    
    fig.update_xaxes(
        range=[data_min - 0.05 * data_range, data_max + 0.05 * data_range],
        title_text=numeric_col.replace('_', ' ').title()
    )
    
    fig.update_yaxes(
        rangemode='tozero',
        title_text='Frequency'
    )
    
    fig.update_layout(
        height=height,
        template=template
    )
    
    return fig


def plot_multi_boxplot(
    df: pd.DataFrame,
    numeric_col: str,
    category_cols: List[str],
    title: Optional[str] = None,
    height: int = 600,
    template: str = 'plotly_white'
) -> go.Figure:
    """
    Create multiple box plots in subplots for comparison.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    numeric_col : str
        Numeric column to plot.
    category_cols : list of str
        List of categorical columns for grouping.
    title : str, optional
        Overall chart title.
    height : int, default 600
        Figure height.
    template : str, default 'plotly_white'
        Plotly theme.
    
    Returns:
    --------
    go.Figure
        Plotly figure object with subplots.
    """
    if title is None:
        title = f"{numeric_col.replace('_', ' ').title()} by Different Categories"
    
    n_plots = len(category_cols)
    cols = min(2, n_plots)
    rows = (n_plots + cols - 1) // cols
    
    subplot_titles = [f"by {col.replace('_', ' ').title()}" for col in category_cols]
    
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    colors = DEFAULT_COLOR_DISCRETE
    
    for idx, cat_col in enumerate(category_cols):
        row = idx // cols + 1
        col = idx % cols + 1
        
        for i, category in enumerate(df[cat_col].unique()):
            data = df[df[cat_col] == category][numeric_col]
            
            fig.add_trace(
                go.Box(
                    y=data,
                    name=str(category),
                    marker_color=colors[i % len(colors)],
                    showlegend=(idx == 0)
                ),
                row=row,
                col=col
            )
    
    fig.update_layout(
        title_text=title,
        height=height * rows // 2,
        template=template
    )
    
    return fig


def plot_age_distribution_with_outliers(
    df: pd.DataFrame,
    age_col: str = 'age',
    lower_bound: int = 18,
    upper_bound: int = 80,
    title: str = "Age Distribution with Outlier Detection",
    height: int = 500,
    template: str = 'plotly_white'
) -> go.Figure:
    """
    Plot age distribution highlighting outliers.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    age_col : str, default 'age'
        Age column name.
    lower_bound : int, default 18
        Lower age bound (below = outlier).
    upper_bound : int, default 80
        Upper age bound (above = outlier).
    title : str
        Chart title.
    height : int, default 500
        Figure height.
    template : str, default 'plotly_white'
        Plotly theme.
    
    Returns:
    --------
    go.Figure
        Plotly figure object.
    """
    df_plot = df.copy()
    df_plot['outlier_status'] = 'Normal'
    df_plot.loc[df_plot[age_col] < lower_bound, 'outlier_status'] = 'Below Normal'
    df_plot.loc[df_plot[age_col] > upper_bound, 'outlier_status'] = 'Above Normal'
    
    fig = px.histogram(
        df_plot,
        x=age_col,
        color='outlier_status',
        nbins=60,
        title=title,
        barmode='overlay',
        opacity=0.7,
        color_discrete_map={
            'Normal': '#636EFA',
            'Below Normal': '#EF553B',
            'Above Normal': '#00CC96'
        },
        labels={age_col: 'Age (years)'}
    )
    
    fig.add_vline(x=lower_bound, line_dash="dash", line_color="red", annotation_text=f"Min: {lower_bound}")
    fig.add_vline(x=upper_bound, line_dash="dash", line_color="red", annotation_text=f"Max: {upper_bound}")
    
    age_min = df[age_col].min()
    age_max = df[age_col].max()
    age_range = age_max - age_min
    
    fig.update_xaxes(
        range=[max(0, age_min - 0.05 * age_range), age_max + 0.05 * age_range],
        title_text='Age (years)'
    )
    
    fig.update_yaxes(
        rangemode='tozero',
        title_text='Frequency'
    )
    
    fig.update_layout(
        height=height,
        template=template
    )
    
    return fig
