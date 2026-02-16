import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import rcParams
import os
from PIL import Image

# Create output directory
os.makedirs('final_thesis_plots', exist_ok=True)
stefano_folder = '/Users/soren/desktop_back_up/_Organized_Results/STEFANO_EXPERIMENTS_FMdataframe.csv'
stefano_folder = '/Users/soren/desktop_back_up/_Organized_Results/STEFANO_FULL_RESULTS_FOR_PLOTTINGdataframe.csv'
stefano_folder = '/Users/soren/desktop_back_up/_Organized_Results/STEFANO_FULL_RESULTS_EXTRA/concated.csv'
jessica_folder = '/Users/soren/desktop_back_up/_Organized_Results/JESSICA_EXPERIMENTS_AFMdataframe.csv'
alr_folder = '/Users/soren/desktop_back_up/_Organized_Results/Original_AutoLR_experimentsdataframe.csv'

# Create sample data (same as before)
np.random.seed(42)
df_line = pd.DataFrame({
    'x': np.arange(0, 10, 0.1),
    'y1': np.sin(np.arange(0, 10, 0.1)) + np.random.normal(0, 0.1, 100),
    'y2': np.cos(np.arange(0, 10, 0.1)) + np.random.normal(0, 0.1, 100),
    'y3': 0.5 * np.sin(np.arange(0, 10, 0.1)) + 0.5 + np.random.normal(0, 0.1, 100),
    'y4': 0.3 * np.cos(np.arange(0, 10, 0.1)) + 0.7 + np.random.normal(0, 0.15, 100),
})

df_violin = pd.DataFrame({
    'category': np.repeat(['A', 'B', 'C', 'D'], 50),
    'value': np.concatenate([
        np.random.normal(5, 1, 50),
        np.random.normal(7, 1.5, 50),
        np.random.normal(6, 0.8, 50),
        np.random.normal(4, 1.2, 50),
    ])
})

pie_data = pd.DataFrame({
    'category': ['Algorithm A', 'Algorithm B', 'Algorithm C', 'Algorithm D'],
    'value': [25, 35, 20, 20]
})

# Define Academic style color palette
ACADEMIC_COLORS = [
    '#007191', 
    '#62C8D3', 
    '#F47A00', 
    '#EF2D56', 
    '#6D597A', 
    '#495867']

print("="*60)
print("CREATING FINAL THESIS PLOTS IN CHOSEN STYLES")
print("="*60)

# ============================================================================
# 1. ACADEMIC STYLE LINE CHART
# ============================================================================
print("\n1. Creating Academic Style Line Chart...")
def make_line_chart(df_line, more_info):

    def set_academic_style():
        """Set academic style with good B&W conversion"""
        plt.style.use('seaborn-v0_8-whitegrid')
        
        rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans'],
            'font.size': 10,
            'axes.titlesize': 11,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 12,
            'lines.linewidth': 1.8,
            'lines.markersize': 6,
            'axes.linewidth': 0.8,
            'grid.linewidth': 0.4,
        })

    set_academic_style()

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ['o', 's', '^', 'D']

    if 'y_conditions' not in more_info:
        more_info['y_conditions'] = [[False] for _ in more_info['y_lines']]
    for y_axis, y_line_label, marker, color, condition in zip(more_info['y_lines'], more_info['y_lines_label'], markers, ACADEMIC_COLORS, more_info['y_conditions']):
        if len(condition) != 1:
            x = df_line[more_info['x_axis']][condition]
            y = df_line[y_axis][condition]
        else:
            x = df_line[more_info['x_axis']]
            y = df_line[y_axis]
        ax.plot(x, y, label=y_line_label, marker=marker, markevery=more_info['mark_freq'], color=color)

    # Labels and titles
    ax.set_xlabel(more_info['x_label'], fontsize=11)
    ax.set_ylabel(more_info['y_label'], fontsize=11)
    ax.set_ylim(more_info['y_lim'] if 'y_lim' in more_info else ax.get_ylim())
    ax.set_title(more_info['title'], fontsize=13, fontweight='bold')

    # Legend and grid
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='black')
    ax.grid(True, alpha=0.3)

    # Consistent tick spacing
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))

    plt.tight_layout()

    # Save in color and B&W
    color_path = f"final_thesis_plots/{more_info['file_name']}.png"
    bw_path = f"final_thesis_plots/{more_info['file_name']}_bw.png"

    fig.savefig(color_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved Academic line chart (color): {color_path}")

    # Save as grayscale
    img = Image.open(color_path).convert('L')
    img.save(bw_path)
    print(f"✓ Saved Academic line chart (B&W): {bw_path}")

    plt.close(fig)

# ============================================================================
# 2. COLORFUL B&W SAFE BOX PLOT
# ============================================================================
print("\n2. Creating Colorful B&W Safe Box Plot...")
def make_boxplot(df_violin, more_info):
    def set_colorful_bw_safe():
        """Colorful style that still works in B&W"""
        plt.style.use('seaborn-v0_8-deep')
        
        rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 10,
            'axes.linewidth': 0.8,
            'grid.linewidth': 0.5,
            'grid.alpha': 0.3,
        })

    set_colorful_bw_safe()

    fig, ax = plt.subplots(figsize=(8, 5))

    # Create box plot with Academic colors
    box_colors = ACADEMIC_COLORS[:4]

    column = more_info['column'] if 'column' in more_info else 'nunique'
    # Create the boxplot manually to control colors
    data_for_plot = []
    for setup in more_info['box_labels']:
        data_for_plot.append(df_violin[df_violin['experiment_name'] == setup][column])
    box_plot = ax.boxplot(data_for_plot,
                        patch_artist=True,
                        labels=more_info['box_labels'],
                        #showmeans=True,
                        #meanline=True,
                        #meanprops=dict(linestyle='--', linewidth=2, color='darkred')
                        )

    ax.set_ylim(more_info['y_lim'] if 'y_lim' in more_info else ax.get_ylim())

    # Apply Academic colors to boxes
    for i, box in enumerate(box_plot['boxes']):
        box.set_facecolor(box_colors[i])
        box.set_alpha(0.7)
        box.set_edgecolor('black')
        box.set_linewidth(1.5)

    # Style the median lines
    for median in box_plot['medians']:
        median.set_color('black')
        median.set_linewidth(2)

    # Style the mean lines
    for mean in box_plot['means']:
        mean.set_color('darkred')
        mean.set_linewidth(2)

    # Style the whiskers and caps
    for whisker in box_plot['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1.5)
        
    for cap in box_plot['caps']:
        cap.set_color('black')
        cap.set_linewidth(1.5)

    # Add individual data points with jitter
    for i, category in enumerate(more_info['box_labels']):
        category_data = df_violin[df_violin['experiment_name'] == category][column]
        # Add jitter to x positions
        x_jitter = np.random.normal(i+1, 0.04, size=len(category_data))
        ax.scatter(x_jitter, category_data, alpha=0.5, s=20, color=box_colors[i], edgecolor='black', linewidth=0.5)

    # Labels and titles
    ax.set_xlabel(more_info['x_label'], fontsize=11)
    ax.set_ylabel(more_info['y_label'], fontsize=11)
    ax.set_title(more_info['title'], fontsize=13, fontweight='bold')

    # Add grid
    ax.grid(True, alpha=0.3, axis='y')

    # Consistent tick spacing
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))

    plt.tight_layout()

    # Save in color and B&W
        # Save in color and B&W
    color_path = f"final_thesis_plots/{more_info['file_name']}.png"
    bw_path = f"final_thesis_plots/{more_info['file_name']}_bw.png"

    fig.savefig(color_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved Colorful B&W box plot (color): {color_path}")

    # Save as grayscale
    img = Image.open(color_path).convert('L')
    img.save(bw_path)
    print(f"✓ Saved Colorful B&W box plot (B&W): {bw_path}")

    plt.close(fig)

# ============================================================================
# 3. SEABORN STYLE PIE CHART (TEXT INSIDE)
# ============================================================================
print("\n3. Creating Seaborn Style Pie Chart (text inside)...")

def make_piechart(pie_data):
    def set_seaborn_style():
        """Seaborn style for pie chart"""
        sns.set_style("white")
        sns.set_context("paper", font_scale=1.1)
        
        rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 10,
            'axes.linewidth': 0,
        })

    set_seaborn_style()

    fig, ax = plt.subplots(figsize=(7, 7))

    # Create Seaborn color palette
    seaborn_colors = sns.color_palette("husl", 4)

    # Create pie chart with text inside
    wedges, texts, autotexts = ax.pie(pie_data['value'], 
                                    labels=pie_data['category'],
                                    autopct='%1.1f%%',
                                    colors=ACADEMIC_COLORS,
                                    startangle=90,
                                    wedgeprops={'edgecolor': 'white', 'linewidth': 2},
                                    textprops={'fontsize': 11, 'fontweight': 'bold'})

    # Customize the text inside
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
        # Add a subtle shadow for better readability
        #autotext.set_path_effects([plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='black', alpha=0.3)])

    # Customize the labels
    for text in texts:
        text.set_fontsize(12)
        text.set_fontweight('bold')

    # Add a title
    ax.set_title('Algorithm Usage Distribution', fontsize=14, fontweight='bold', pad=20)

    # Equal aspect ratio ensures the pie is circular
    ax.axis('equal')

    plt.tight_layout()

    # Save in color and B&W
    color_path = 'final_thesis_plots/seaborn_pie_chart.png'
    bw_path = 'final_thesis_plots/seaborn_pie_chart_bw.png'

    fig.savefig(color_path, dpi=300, bbox_inches='tight', facecolor='white', transparent=False)
    print(f"✓ Saved Seaborn pie chart (color): {color_path}")
    # Save as grayscale
    img = Image.open(color_path).convert('L')
    img.save(bw_path)
    print(f"✓ Saved Colorful B&W box plot (B&W): {bw_path}")

    plt.close(fig)

def make_alr():
    df_line = pd.read_csv(alr_folder)
    df_line = df_line.groupby(['experiment_name', 'generation'])['fitness'].agg(['mean', 'max']).reset_index()
    print('')
    more_info = {
        'x_axis': 'generation',
        'x_label': 'generation',
        'y_lines': ['max', 'mean'],
        'y_lines_label': ['best fitness', 'population fitness'],
        'y_label': 'fitness',
        'title': 'Best and Population fitness for AutoLR',
        'mark_freq': 100,
        'file_name': 'alr_evo'
    }
    make_line_chart(df_line, more_info)

def make_stef_best():
    df_line = pd.read_csv(stefano_folder)
    df_line = df_line.groupby(['experiment_name', 'generation', 'run'])['fitness'].agg(['mean', 'max']).reset_index()
    df_line = df_line.groupby(['experiment_name', 'generation'])['max'].agg(['mean']).reset_index()
    best_dict = {'FM': 0.0, 'FMX':0.0, 'OM':0.0, 'OMX':0.0}
    def fix_max(row):
        if row['max'] < best_dict[row['experiment_name']]:
            row['max'] = best_dict[row['experiment_name']]
        else:
            best_dict[row['experiment_name']] = row['max']
        return row

    #df_line = df_line.apply(fix_max, axis=1)

    more_info = {
        'x_axis': 'generation',
        'x_label': 'generation',
        'y_lines': ['mean', 'mean', 'mean', 'mean'],
        'y_conditions': [df_line['experiment_name'] == 'FMX', 
                        df_line['experiment_name'] == 'FM', 
                        df_line['experiment_name'] == 'OM', 
                        df_line['experiment_name'] == 'OMX'],
        'y_lines_label': ['FMX', 'FM', 'OM', 'OMX'],
        'y_label': 'Best Fitness',
        'y_lim': [0.0, 1.0],
        'title': 'Best Fitness for Facilitated Mutation Experiments',
        'mark_freq': 10,
        'file_name': 'stefano_best_fit'
    }
    make_line_chart(df_line, more_info)\

def make_stef_pop():
    df_line = pd.read_csv(stefano_folder)
    df_line = df_line.groupby(['experiment_name', 'generation'])['fitness'].agg(['mean']).reset_index()
    more_info = {
        'x_axis': 'generation',
        'x_label': 'generation',
        'y_lines': ['mean', 'mean', 'mean', 'mean'],
        'y_conditions': [df_line['experiment_name'] == 'FMX', 
                        df_line['experiment_name'] == 'FM', 
                        df_line['experiment_name'] == 'OM', 
                        df_line['experiment_name'] == 'OMX'],
        'y_lines_label': ['FMX', 'FM', 'OM', 'OMX'],
        'y_label': 'Population Fitness',
        'y_lim': [0.0, 1.0],
        'title': 'Population Fitness for Facilitated Mutation Experiments',
        'mark_freq': 10,
        'file_name': 'stefano_mean_fit'
    }
    make_line_chart(df_line, more_info)

def make_stef_pop_box():
    df_line = pd.read_csv(stefano_folder)
    df_line = df_line.groupby(['experiment_name', 'generation', 'run'])['fitness'].agg(['mean']).reset_index()
    # For each run, get the mean fitness at the final generation
    df_line = df_line[df_line['generation'] == df_line['generation'].max()]

    more_info = {
        'box_labels': ['FMX', 'FM', 'OM', 'OMX'],
        'x_label': 'Mutation Setup',
        'y_label': 'Average Fitness in the Last Generation',
        'title': 'Average Fitness in the Last Generation per Mutation Setup',
        'mark_freq': 100,
        'file_name': 'stefano_pop_box',
        'column': 'mean',
        'y_lim': [0.0, 1.0],
    }
    make_boxplot(df_line, more_info)

def make_stef_uniques():
    #df_box = pd.read_csv('/Users/soren/desktop_back_up/_Organized_Results/STEFANO_FULL_RESULTS_FOR_PLOTTINGdataframe.csv')
    df_box = pd.read_csv(stefano_folder)
    df_box = df_box[df_box['fitness'] > 0.5]
    df_box = df_box.groupby(['experiment_name', 'run_number'])['smart_phenotype'].agg([pd.Series.nunique]).reset_index()
    #df_line = df_line.groupby(['experiment_name', 'generation'])['fitness'].agg(['mean']).reset_index()
    more_info = {
        'box_labels': ['FMX', 'FM', 'OM', 'OMX'],
        'x_label': 'Mutation Setup',
        'y_label': 'Number of Unique Viable Behaviors',
        'title': 'Number of Unique Viable Behaviors per Mutation Setup',
        'mark_freq': 100,
        'file_name': 'stefano_uniques',
        'y_lim': [0, 3200]
    }
    make_boxplot(df_box, more_info)

def make_stef_evals():
    df_box = pd.read_csv(stefano_folder)
    def is_evaluated(row):
        if 'grad' in row['smart_phenotype']:
            return True 
        return False
    df_box = df_box[df_box.apply(is_evaluated, axis=1)]
    df_box = df_box.groupby(['experiment_name', 'run_number'])['smart_phenotype'].agg([pd.Series.nunique]).reset_index()
    #df_line = df_line.groupby(['experiment_name', 'generation'])['fitness'].agg(['mean']).reset_index()
    more_info = {
        'box_labels': ['FMX', 'FM', 'OM', 'OMX'],
        'x_label': 'Mutation Setup',
        'y_label': 'Number of Evaluations',
        'title': 'Number of Evaluations per Mutation Setup',
        'mark_freq': 100,
        'file_name': 'stefano_comp_costs',
        'y_lim': [0, 16100]
    }
    make_boxplot(df_box, more_info)

def make_jessica_pop():
    df_line = pd.read_csv(jessica_folder)
    df_line = df_line.groupby(['experiment_name', 'generation'])['fitness'].agg(['mean']).reset_index()
    more_info = {
        'x_axis': 'generation',
        'x_label': 'generation',
        'y_lines': ['mean', 'mean', 'mean', 'mean'],
        'y_conditions': [df_line['experiment_name'] == 'extended_grammar_adaptive', 
                        df_line['experiment_name'] == 'extended_grammar_standard', 
                        df_line['experiment_name'] == 'old_grammar_adaptive', 
                        df_line['experiment_name'] == 'old_grammar_standard'],
        'y_lines_label': ['AFM + FGG', 'OM + FGG', 'AFM', 'OM'],
        'y_label': 'Population Fitness',
        'title': 'Population Fitness for Adaptive Facilitated Mutation Experiments',
        'mark_freq': 10,
        'file_name': 'jessica_mean_fit'
    }
    make_line_chart(df_line, more_info)

def make_jessica_best():
    df_line = pd.read_csv(jessica_folder)
    df_line = df_line.groupby(['experiment_name', 'generation'])['fitness'].agg(['max']).reset_index()
    more_info = {
        'x_axis': 'generation',
        'x_label': 'generation',
        'y_lines': ['max', 'max', 'max', 'max'],
        'y_conditions': [df_line['experiment_name'] == 'extended_grammar_adaptive', 
                        df_line['experiment_name'] == 'extended_grammar_standard', 
                        df_line['experiment_name'] == 'old_grammar_adaptive', 
                        df_line['experiment_name'] == 'old_grammar_standard'],
        'y_lines_label': ['AFM + FGG', 'OM + FGG', 'AFM', 'OM'],
        'y_label': 'Best Fitness',
        'title': 'Best Fitness for Adaptive Facilitated Mutation Experiments',
        'mark_freq': 10,
        'file_name': 'jessica_max_fit'
    }
    make_line_chart(df_line, more_info)

#make_stef_best()
make_stef_pop_box()
#make_stef_uniques()
make_stef_evals()