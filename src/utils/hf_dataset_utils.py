import os 
import math
import logging
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from pathlib import Path
from typing import Dict, Optional

# ============================================================
#                 LOGGING CONFIGURATION
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
#                 VISUALIZATION FUNCTIONS
# ============================================================
def create_distribution_plots( high_freq_df: pd.DataFrame, plots_per_row: int = 3,
                                save_path: Optional[str] = None,
                                filename: str = "numeric_distributions"
                            ) -> None:
    """
    Create distribution plots for all numeric columns in the DataFrame.
    
    Args:
        high_freq_df: DataFrame containing high-frequency data
        plots_per_row: Number of plots to display per row
        save_path: Directory where plot will be saved (if None, only displays)
        filename: Base filename for the saved plot (without extension)
    """
    # Select only numeric columns
    numeric_data = high_freq_df.select_dtypes(include=['number'])
    
    # Filter out columns with only one unique value
    valid_cols = []
    for col in numeric_data.columns:
        n_unique = numeric_data[col].dropna().nunique()
        if n_unique > 1:
            valid_cols.append(col)
        else:
            logger.info(f"Skipping column '{col}' - only {n_unique} unique value(s)")
    
    if not valid_cols:
        logger.warning("No numeric columns with multiple values found for plotting")
        return
    
    # Filter to only valid columns
    numeric_data = numeric_data[valid_cols]
    num_cols = len(valid_cols)
    
    # Calculate grid dimensions
    num_rows = math.ceil(num_cols / plots_per_row)
    
    # Create figure and axes (limit figure size to prevent memory issues)
    max_fig_height = 50  # Limit total figure height
    fig_height = min(5 * num_rows, max_fig_height)
    fig, axes = plt.subplots(nrows=num_rows,
                            ncols=plots_per_row,
                            figsize=(6 * plots_per_row, fig_height)
                        )
    
    # Flatten axes array for easier iteration
    if num_rows == 1 and plots_per_row == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Create color palette
    colors = sns.color_palette("husl", num_cols)
    
    # Plot histograms for each numeric column
    for idx, (ax, col) in enumerate(zip(axes[:num_cols], numeric_data.columns)):
        try:
            # Drop NaN values for this column
            col_data = numeric_data[col].dropna()
            
            if len(col_data) == 0:
                ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{col} Distribution', fontsize=12, pad=10, fontweight='bold')
                continue
            
            # Calculate statistics
            mean_val = col_data.mean()
            median_val = col_data.median()
            std_val = col_data.std()
            min_val = col_data.min()
            max_val = col_data.max()
            
            # Plot histogram with KDE
            sns.histplot(data=col_data, kde=True, ax=ax, color=colors[idx], edgecolor='white', alpha=0.7, stat='density' )    # type: ignore
                    
            # Add mean and median lines
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=1.5, label=f'Median: {median_val:.2f}')
            
            # Customize appearance
            ax.set_title(f'{col} Distribution', fontsize=12, pad=10, fontweight='bold')
            ax.set_xlabel("Value", fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            
            # Add statistics text box
            stats_text = (f"Std: {std_val:.2f}\n"
                        f"Min: {min_val:.2f}\n"
                        f"Max: {max_val:.2f}"
                    )
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'),fontsize=9
                    )
            
            # Customize legend
            ax.legend(fontsize=8, framealpha=0.9, loc='upper left')
            ax.grid(True, alpha=0.3, linestyle='--')
            
        except Exception as e:
            logger.warning(f"Could not plot {col}: {e}")
            ax.text(0.5, 0.5, f'Error plotting\n{col}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{col} Distribution', fontsize=12, pad=10, fontweight='bold')
    
    # Remove empty subplots
    for idx in range(num_cols, len(axes)):
        fig.delaxes(axes[idx])
    
    # Adjust layout
    plt.suptitle("Distribution of Numeric Variables", fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        output_path = os.path.join(save_path, f"{filename}.pdf")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none' )
        logger.info(f"Distribution plot saved to: {output_path}")
    
    plt.close()

# ============================================================
#          DATA LOSS VISUALIZATION FUNCTION
# ============================================================

def plot_data_loss_analysis(df_original: pd.DataFrame,
                            df_after_speed: pd.DataFrame,
                            df_after_power: pd.DataFrame,
                            df_after_cfd: pd.DataFrame,
                            df_final: pd.DataFrame,
                            speed_threshold: Optional[float],
                            power_threshold: Optional[float],
                            cfd_conditions: Optional[Dict[str, float]],
                            save_path: str,
                            imo: str
                        ) -> None:
    """
    Create comprehensive visualizations showing data loss through filtering pipeline.
    
    This function generates a multi-panel visualization showing:
    1. Waterfall chart of data retention through each filtering stage
    2. Pie chart showing final data retention vs loss breakdown
    3. Detailed summary statistics table
    
    Args:
        df_original: DataFrame before any filtering
        df_after_speed: DataFrame after speed filter (before power filter)
        df_after_power: DataFrame after power filter (before CFD conditions)
        df_after_cfd: DataFrame after CFD conditions (before dropna)
        df_final: DataFrame after all filters including dropna
        speed_threshold: Speed threshold used for filtering (None if not applied)
        power_threshold: Power threshold used for filtering (None if not applied)
        cfd_conditions: Dictionary of CFD conditions applied (None if not applied)
        save_path: Directory where plots will be saved
        imo: IMO identifier for filename
        
    Returns:
        None (saves plot to disk)
        
    Side Effects:
        - Creates PNG file: {save_path}/data_loss_analysis_{imo}.png
        - Logs the save location
        
    Example:
        >>> plot_data_loss_analysis(
        ...     df_original=df_raw,
        ...     df_after_speed=df_speed_filtered,
        ...     df_after_power=df_power_filtered,
        ...     df_after_cfd=df_cfd_filtered,
        ...     df_final=df_clean,
        ...     speed_threshold=2.0,
        ...     power_threshold=500.0,
        ...     cfd_conditions={'min_speed': 2.0, 'max_speed': 15.0},
        ...     save_path='./analysis',
        ...     imo='9123456'
        ... )
    """
    
    # ========== Calculate row counts at each stage ==========
    original_count = len(df_original)
    after_speed_count = len(df_after_speed)
    after_power_count = len(df_after_power)
    after_cfd_count = len(df_after_cfd)
    final_count = len(df_final)
    
    # ========== Calculate losses between stages ==========
    speed_loss = original_count - after_speed_count
    power_loss = after_speed_count - after_power_count
    cfd_loss = after_power_count - after_cfd_count
    dropna_loss = after_cfd_count - final_count
    total_loss = original_count - final_count
    
    # ========== Calculate percentages ==========
    speed_loss_pct = (speed_loss / original_count * 100) if original_count > 0 else 0
    power_loss_pct = (power_loss / after_speed_count * 100) if after_speed_count > 0 else 0
    cfd_loss_pct = (cfd_loss / after_power_count * 100) if after_power_count > 0 else 0
    dropna_loss_pct = (dropna_loss / after_cfd_count * 100) if after_cfd_count > 0 else 0
    total_loss_pct = (total_loss / original_count * 100) if original_count > 0 else 0
    
    # ========== Create figure with 2 subplots ==========
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # ========== PLOT 1: Waterfall Chart ==========
    categories = [
        'Original\nData', 
        'After Speed\nFilter', 
        'After Power\nFilter', 
        'After CFD\nConditions',
        'After\ndropna()'
    ]
    values = [original_count, after_speed_count, after_power_count, after_cfd_count, final_count]
    
    # Bar colors for each stage
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e67e22', '#e74c3c']
    x_pos = np.arange(len(categories))
    
    # Create bars
    bars = ax1.bar(x_pos, values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)
    
    # ========== Add loss annotations between stages ==========
    
    # Speed filter loss annotation
    if speed_threshold is not None and speed_loss > 0:
        ax1.plot([0, 1], [original_count, original_count], 'k--', alpha=0.5, linewidth=1)
        ax1.annotate(
            f'−{speed_loss:,}\n({speed_loss_pct:.1f}%)', 
            xy=(0.5, original_count - speed_loss/2),
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95, edgecolor='red', linewidth=2),
            fontsize=10, color='red', fontweight='bold'
        )
    
    # Power filter loss annotation
    if power_threshold is not None and power_loss > 0:
        ax1.plot([1, 2], [after_speed_count, after_speed_count], 'k--', alpha=0.5, linewidth=1)
        ax1.annotate(
            f'−{power_loss:,}\n({power_loss_pct:.1f}%)', 
            xy=(1.5, after_speed_count - power_loss/2),
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95, edgecolor='purple', linewidth=2),
            fontsize=10, color='purple', fontweight='bold'
        )
    
    # CFD conditions loss annotation
    if cfd_conditions is not None and cfd_loss > 0:
        ax1.plot([2, 3], [after_power_count, after_power_count], 'k--', alpha=0.5, linewidth=1)
        ax1.annotate(
            f'−{cfd_loss:,}\n({cfd_loss_pct:.1f}%)', 
            xy=(2.5, after_power_count - cfd_loss/2),
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95, edgecolor='orange', linewidth=2),
            fontsize=10, color='darkorange', fontweight='bold'
        )
    
    # Dropna loss annotation
    if dropna_loss > 0:
        ax1.plot([3, 4], [after_cfd_count, after_cfd_count], 'k--', alpha=0.5, linewidth=1)
        ax1.annotate(
            f'−{dropna_loss:,}\n({dropna_loss_pct:.1f}%)', 
            xy=(3.5, after_cfd_count - dropna_loss/2),
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95, edgecolor='#c0392b', linewidth=2),
            fontsize=10, color='#c0392b', fontweight='bold'
        )
    
    # ========== Add value labels on bars ==========
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        retention_pct = (val / original_count * 100) if original_count > 0 else 0
        ax1.text(
            bar.get_x() + bar.get_width()/2., 
            height,
            f'{val:,}\n({retention_pct:.1f}%)',
            ha='center', va='bottom', 
            fontsize=10, fontweight='bold'
        )
    
    # ========== Configure Plot 1 aesthetics ==========
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Rows', fontsize=13, fontweight='bold')
    
    # Build title with filter conditions
    title = f'Data Filtering Pipeline - IMO {imo}'
    subtitle_parts = []
    if speed_threshold is not None:
        subtitle_parts.append(f'Speed ≥ {speed_threshold} m/s')
    if power_threshold is not None:
        subtitle_parts.append(f'Power ≥ {power_threshold} W')
    if cfd_conditions:
        cfd_parts = []
        if 'min_speed' in cfd_conditions:
            cfd_parts.append(f"Speed: {cfd_conditions['min_speed']}-{cfd_conditions.get('max_speed', '∞')} m/s")
        if 'min_draft' in cfd_conditions:
            cfd_parts.append(f"Draft: {cfd_conditions['min_draft']}-{cfd_conditions.get('max_draft', '∞')} m")
        subtitle_parts.extend(cfd_parts)
    
    if subtitle_parts:
        title += f'\n({" | ".join(subtitle_parts)})'
    
    ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
    ax1.set_ylim(0, original_count * 1.18)
    
    # ========== PLOT 2: Pie Chart ==========
    retention_pct = 100 - total_loss_pct
    
    # Build pie chart data (only non-zero losses)
    sizes = []
    labels = []
    colors_pie = []
    
    # Retained data (always include)
    sizes.append(retention_pct)
    labels.append(f'Retained\n{retention_pct:.1f}%\n({final_count:,} rows)')
    colors_pie.append('#27ae60')
    
    # Speed filter loss
    if speed_loss > 0:
        speed_loss_pct_orig = (speed_loss / original_count * 100)
        sizes.append(speed_loss_pct_orig)
        labels.append(f'Speed Filter\n{speed_loss_pct_orig:.1f}%\n({speed_loss:,} rows)')
        colors_pie.append('#e74c3c')
    
    # Power filter loss
    if power_loss > 0:
        power_loss_pct_orig = (power_loss / original_count * 100)
        sizes.append(power_loss_pct_orig)
        labels.append(f'Power Filter\n{power_loss_pct_orig:.1f}%\n({power_loss:,} rows)')
        colors_pie.append('#9b59b6')
    
    # CFD conditions loss
    if cfd_loss > 0:
        cfd_loss_pct_orig = (cfd_loss / original_count * 100)
        sizes.append(cfd_loss_pct_orig)
        labels.append(f'CFD Conditions\n{cfd_loss_pct_orig:.1f}%\n({cfd_loss:,} rows)')
        colors_pie.append('#e67e22')
    
    # Dropna loss
    if dropna_loss > 0:
        dropna_loss_pct_orig = (dropna_loss / original_count * 100)
        sizes.append(dropna_loss_pct_orig)
        labels.append(f'Missing Data\n{dropna_loss_pct_orig:.1f}%\n({dropna_loss:,} rows)')
        colors_pie.append('#95a5a6')
    
    explode = tuple([0.05] * len(sizes))
    
    # Create pie chart
    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors_pie,autopct='', startangle=90,
                                        explode=explode, shadow=True, textprops={'fontsize': 10, 'fontweight': 'bold'})
    
    ax2.set_title(
        f'Data Retention Analysis - IMO {imo}\n'
        f'Total Loss: {total_loss_pct:.1f}% | Retained: {retention_pct:.1f}%',
        fontsize=14, fontweight='bold', pad=20
    )
    # ========== Save plot ==========
    plot_path = os.path.join(save_path, f"data_loss_analysis_{imo}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Data loss analysis plot saved to: {plot_path}")