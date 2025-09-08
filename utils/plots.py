import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fig_size = (14, 8)
sns.set_theme(style="ticks")

def get_mutation_df(df):
    df_mutation = df[df['operation'] == 'mutation']
    df_mutation['parent_1'] = df_mutation['parents'].apply(lambda x: int(x[0]))
    def determine_mutation_type(row):
        # Determine crossover type for this row.
        # "Circular" if archive_id is different from both parents but "source" is "archive"
        # "Redundant" if archive_id is the same as one of the parents
        # "Destructive" if archive_id is different from both parents and "source" is not "archive"
        
        p1_smart_phenotype = df[df['genetic_id'] == row['parent_1']]['smart_phenotype'].head(1).item()
        if row['smart_phenotype'] == p1_smart_phenotype:
            return "Redundant"
        elif 'source' in row and row['source'] == 'archive':
            return "Circular"
        else:
            return "Destructive"
    df_mutation['mutation_type'] = df_mutation.apply(determine_mutation_type, axis=1)

    return df_mutation

def is_mutation_df(df):
    if 'mutation_type' in df.columns:
        return True
    else:
        return False

def is_crossover_df(df):
    if 'crossover_type' in df.columns:
        return True
    else:
        return False
    
def get_crossover_df(df):
    df_crossover = df[df['operation'] == 'crossover']
    df_crossover['parent_1'] = df_crossover['parents'].apply(lambda x: int(x[0]))
    df_crossover['parent_2'] = df_crossover['parents'].apply(lambda x: int(x[1]))
    def determine_crossover_type(row):
        # Determine crossover type for this row.
        # "Circular" if archive_id is different from both parents but "source" is "archive"
        # "Redundant" if archive_id is the same as one of the parents
        # "Destructive" if archive_id is different from both parents and "source" is not "archive"

        p1_smart_phenotype = df[df['genetic_id'] == row['parent_1']]['smart_phenotype'].head(1).item()
        p2_smart_phenotype = df[df['genetic_id'] == row['parent_2']]['smart_phenotype'].head(1).item()
        if row['smart_phenotype'] == p1_smart_phenotype or row['smart_phenotype'] == p2_smart_phenotype:
            return "Redundant"
        elif 'source' in row and row['source'] == 'archive':
            return "Circular"
        else:
            return "Destructive"
    df_crossover['crossover_type'] = df_crossover.apply(determine_crossover_type, axis=1)

    return df_crossover

def plot_grouped_stacked_histogram(df, archive_min_freq=5, generation_group_size=3, max_archive_groups=15):
    """
    Create a stacked histogram with grouped generations and behavior_ids.
    
    Parameters:
    df : pandas.DataFrame with columns 'generation' and 'behavior_id'
    archive_min_freq : minimum total occurrences to show individual behavior_id
    generation_group_size : number of generations to group together
    max_archive_groups : maximum number of behavior_id groups to show (including 'Others')
    """
    
    df['generation'] = df['generation'].astype(int)

    # Count occurrences of each behavior_id in each generation
    count_df = df.groupby(['generation', 'behavior_id']).size().reset_index(name='Count')
    
    # Group generations
    count_df['generation_Group'] = (count_df['generation'] - 1) // generation_group_size
    count_df['generation_Group_Label'] = count_df.apply(
        lambda row: f"G{row['generation_Group']*generation_group_size + 1}-{row['generation_Group']*generation_group_size + generation_group_size}",
        axis=1
    )
    
    # Identify frequent behavior_ids (across all data)
    total_counts = count_df.groupby('behavior_id')['Count'].sum()
    frequent_ids = total_counts[total_counts >= archive_min_freq].index.tolist()
    
    # If we have too many frequent IDs, keep only the top ones
    if len(frequent_ids) > max_archive_groups - 1:  # -1 for 'Others'
        top_ids = total_counts.nlargest(max_archive_groups - 1).index.tolist()
        frequent_ids = top_ids
    
    # Create behavior_id groups
    count_df['Archive_Group'] = count_df['behavior_id'].apply(
        lambda x: f"ID_{x}" if x in frequent_ids else 'Others'
    )
    
    # Aggregate by generation groups and archive groups
    grouped_df = count_df.groupby(['generation_Group', 'generation_Group_Label', 'Archive_Group'])['Count'].sum().reset_index()
    
    # Pivot for stacked bar plot
    final_pivot = grouped_df.pivot_table(
        index=['generation_Group', 'generation_Group_Label'],
        columns='Archive_Group',
        values='Count',
        fill_value=0
    )
    
    # Sort behavior_id groups (put Others last)
    archive_columns = sorted([col for col in final_pivot.columns if col != 'Others'])
    if 'Others' in final_pivot.columns:
        archive_columns.append('Others')
    final_pivot = final_pivot[archive_columns]
    
    # Create the plot
    plt.figure(figsize=fig_size)
    
    # Create stacked bar plot
    ax = final_pivot.plot(kind='bar', stacked=True, colormap='tab20', width=0.8)
    
    plt.title(f'behavior_id Frequency by generation Groups\n'
              f'(generations grouped by {generation_group_size}, '
              f'behavior_ids with < {archive_min_freq} occurrences grouped)')
    plt.xlabel('generation Groups')
    plt.ylabel('Frequency Count')
    
    # Set x-axis labels to generation group ranges
    x_positions = np.arange(len(final_pivot))
    plt.xticks(x_positions, final_pivot.index.get_level_values('generation_Group_Label'), rotation=45)
    
    # Move legend outside if there are many groups
    if len(archive_columns) > 8:
        plt.legend(title='behavior_id Groups', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.legend(title='behavior_id Groups')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"Original generations: {df['generation'].min()} to {df['generation'].max()}")
    print(f"generation groups: {len(final_pivot)}")
    print(f"Total unique behavior_ids: {len(total_counts)}")
    print(f"behavior_id groups shown: {len(archive_columns)}")
    print(f"behavior_ids in 'Others' group: {len(total_counts) - len(frequent_ids)}")
    
    return final_pivot, grouped_df

def plot_percentile_grouped_stacked_histogram(df, archive_percentile=90, generation_group_size=3, max_archive_groups=15):
    """
    Group behavior_ids by percentile instead of fixed frequency.
    """
    df['generation'] = df['generation'].astype(int)
    
    count_df = df.groupby(['generation', 'behavior_id']).size().reset_index(name='Count')
    
    # Group generations
    count_df['generation_Group'] = (count_df['generation'] - 1) // generation_group_size
    count_df['generation_Group_Label'] = count_df.apply(
        lambda row: f"G{row['generation_Group']*generation_group_size + 1}-{row['generation_Group']*generation_group_size + generation_group_size}",
        axis=1
    )
    
    # Identify behavior_ids above percentile threshold
    total_counts = count_df.groupby('behavior_id')['Count'].sum()
    threshold = np.percentile(total_counts, archive_percentile)
    frequent_ids = total_counts[total_counts >= threshold].index.tolist()
    
    # Limit number of groups
    if len(frequent_ids) > max_archive_groups - 1:
        top_ids = total_counts.nlargest(max_archive_groups - 1).index.tolist()
        frequent_ids = top_ids
    
    count_df['Archive_Group'] = count_df['behavior_id'].apply(
        lambda x: f"ID_{x}" if x in frequent_ids else 'Others'
    )
    
    # Aggregate and pivot
    grouped_df = count_df.groupby(['generation_Group', 'generation_Group_Label', 'Archive_Group'])['Count'].sum().reset_index()
    final_pivot = grouped_df.pivot_table(
        index=['generation_Group', 'generation_Group_Label'],
        columns='Archive_Group',
        values='Count',
        fill_value=0
    )
    
    # Sort columns
    archive_columns = sorted([col for col in final_pivot.columns if col != 'Others'])
    if 'Others' in final_pivot.columns:
        archive_columns.append('Others')
    final_pivot = final_pivot[archive_columns]
    
    # Plot
    plt.figure(figsize=fig_size)
    ax = final_pivot.plot(kind='bar', stacked=True, colormap='tab20', width=0.8)
    
    plt.title(f'behavior_id Frequency by generation Groups\n'
              f'(Top {100-archive_percentile}% behavior_ids shown individually)')
    plt.xlabel('generation Groups')
    plt.ylabel('Frequency Count')
    plt.xticks(np.arange(len(final_pivot)), final_pivot.index.get_level_values('generation_Group_Label'), rotation=45)
    
    if len(archive_columns) > 8:
        plt.legend(title='behavior_id Groups', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.legend(title='behavior_id Groups')
    
    plt.tight_layout()
    plt.show()
    
    return final_pivot

def plot_fitness_and_unique_runs(df):
    # Assuming you have a DataFrame 'df' with columns 'generation', 'fitness', 'experiment_name', and 'run_number'

    # Step 1: Group by 'experiment_name', 'run_number', and 'generation', and compute the mean and max fitness for each group
    fitness_data = df.groupby(['experiment_name', 'run_number', 'generation'])['fitness'].agg(['mean', 'max']).reset_index()

    # Step 2: Group by 'generation' and compute the overall mean and max fitness for each generation
    generation_stats = fitness_data.groupby('generation').agg({'mean': 'mean', 'max': 'mean', 'run_number': 'nunique'}).reset_index()
    generation_stats.rename(columns={'run_number': 'Unique Runs'}, inplace=True)

    # Step 3: Calculate highest and lowest fitness across all runs per generation from the max fitness data
    highest_fitness = fitness_data.groupby('generation')['max'].max().reset_index()
    lowest_fitness = fitness_data.groupby('generation')['max'].min().reset_index()

    # Step 4: Plot the average fitness, best fitness, and the shadow for the range between highest and lowest fitness
    fig, ax1 = plt.subplots(figsize=(fig_size[0], fig_size[1]))

    # Plot average fitness on the first y-axis
    ax1.plot(generation_stats['generation'], generation_stats['mean'], marker='o', linestyle='-', color='b', label='Average Fitness')
    ax1.plot(generation_stats['generation'], generation_stats['max'], marker='s', linestyle='-', color='g', label='Best Fitness')
    ax1.fill_between(generation_stats['generation'], lowest_fitness['max'], highest_fitness['max'], alpha=0.3, color='g')

    ax1.set_xlabel('generation')
    ax1.set_ylabel('fitness')
    ax1.set_title('Fitness Across generations')

    # Create a secondary y-axis for the histogram
    ax2 = ax1.twinx()

    # Plot transparent histogram for unique runs on the secondary y-axis
    ax2.hist(generation_stats['generation'], bins=len(generation_stats), alpha=0.3, color='gray', edgecolor='black',
            weights=generation_stats['Unique Runs'], label='Unique Runs')
    ax2.set_ylabel('Unique Runs')

    # Combine the legends from both y-axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2)

    plt.grid(True)
    plt.show()

def plot_average_population_fitness_per_experiment(df):
    experiment_names = df['experiment_name'].unique()
    df_mean_fitness = df.groupby(['experiment_name', 'run_number']).agg({'generation': 'max'}).reset_index()

    last_gen_stats = {'experiment_name': [], "Fitness": [], "generation": []}
    for i, data in df_mean_fitness.iterrows():
        #print(data)
        last_gen_stats['experiment_name'].append(data['experiment_name'])
        last_gen_stats['fitness'].append(np.mean(df[(df['generation'] == data['generation']) & (df['experiment_name'] == data['experiment_name']) & (df['run_number'] == data['run_number'])]['fitness']))
        last_gen_stats['generation'].append(data['generation'])

    df_mean_fitness = pd.DataFrame(last_gen_stats)
    # Step 1: Create a box plot
    plt.figure(figsize=(fig_size[0], fig_size[1]))  # Adjust the figure size as needed

    # Use seaborn to create a box plot, specifying 'x' as 'experiment_name' and 'y' as 'fitness'
    sns.boxplot(data=df_mean_fitness, x='experiment_name', y='fitness')

    # Customize the plot
    plt.title('Population Fitness per Experiment')
    plt.xlabel('Experiment Name')
    plt.ylabel('Population Fitness')

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45)

    # Show the plot
    #plt.tight_layout()
    #plt.show()

    # Set custom y-axis limits (adjust these values as needed)
    #plt.ylim(0.8, 0.93)  # Example: set the y-axis limits from 0 to 100

    # Show the plot
    plt.tight_layout()
    plt.show()
    for i, experiment_name in enumerate(experiment_names):
        df_i = df_mean_fitness[df_mean_fitness['experiment_name'] == experiment_name]
        print(f"Experiment: {experiment_name} {np.mean(df_i['fitness'])}")

def plot_max_fitness_per_experiment(df, y_lim=None):
    experiment_names = df['experiment_name'].unique()
    # Assuming you have a DataFrame 'df' with columns 'experiment_name', 'run_number', and 'fitness'
    df_max_fitness = df.groupby(['experiment_name', 'run_number']).agg({'fitness': 'max'}).reset_index()
    print(df_max_fitness.head(5))
    # Step 1: Create a box plot
    plt.figure(figsize=(fig_size[0], fig_size[1]))  # Adjust the figure size as needed

    # Use seaborn to create a box plot, specifying 'x' as 'experiment_name' and 'y' as 'fitness'
    sns.boxplot(data=df_max_fitness, x='experiment_name', y='fitness')

    # Customize the plot
    plt.title('Max Fitness per Experiment')
    plt.xlabel('Experiment Name')
    plt.ylabel('Max Fitness')

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45)

    # Show the plot
    #plt.tight_layout()
    #plt.show()

    if y_lim == None:
        y_min = df_max_fitness['fitness'].min()
        y_max = df_max_fitness['fitness'].max()
    else:
        y_min = y_lim[0]
        y_max = y_lim[1]
    # Set custom y-axis limits (adjust these values as needed)
    plt.ylim(y_lim[0], y_lim[1])  # Example: set the y-axis limits from 0 to 100

    # Show the plot
    plt.tight_layout()
    plt.show()
    for i, experiment_name in enumerate(experiment_names):
        df_i = df_max_fitness[df_max_fitness['experiment_name'] == experiment_name]
        print(f"Experiment: {experiment_name} Mean: {np.mean(df_i['fitness'])} Max: {np.max(df_i['fitness'])}")

def plot_unique_evaluations_per_experiment(df, min_fitness=0.0):
    # Assuming you have a DataFrame 'df' with columns 'experiment_name', 'run_number', and 'fitness'

    # min fitness should be problem dependent above random guesses
    df_unique_evals = df[df['fitness'] >= min_fitness].groupby(['experiment_name', 'run_number']).agg({'smart_phenotype': 'nunique'}).reset_index()
    print(df_unique_evals.head(5))
    # Step 1: Create a box plot
    plt.figure(figsize=(fig_size[0], fig_size[1]))  # Adjust the figure size as needed

    # Use seaborn to create a box plot, specifying 'x' as 'experiment_name' and 'y' as 'fitness'
    sns.boxplot(data=df_unique_evals, x='experiment_name', y='smart_phenotype')

    # Customize the plot
    plt.title('Unique Evaluations per Experiment')
    plt.xlabel('Experiment Name')
    plt.ylabel('Unique Evaluations')

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45)

    # Set custom y-axis limits (adjust these values as needed)
    #plt.ylim(0.8, 0.93)  # Example: set the y-axis limits from 0 to 100

    # Show the plot
    plt.tight_layout()
    plt.show()
    #for i, experiment_name in enumerate(experiment_names):
        #df_i = df_max_fitness[df_max_fitness['experiment_name'] == experiment_name]
        #print(f"{experiment_name} {np.mean(df_i['fitness'])}")

def plot_operations_per_generation(df, vertical_markers=None):
    sns.set_theme(style="ticks")
    df['generation'] = df['generation'].astype('category')
    f, ax = plt.subplots(figsize=fig_size)
    sns.despine(f)
    sns.histplot(
        data=df,
        x="generation",
        hue='operation',
        multiple="stack",
        edgecolor=".3",
        linewidth=.5,
        ax=ax
    )
    plt.xlabel('generation')
    plt.ylabel('Count')
    plt.title('Histogram of Operations by generation')
    # Add vertical markers
    if vertical_markers is not None:
        for gen in vertical_markers:
            plt.axvline(x=gen, color='red', linestyle='--', alpha=0.7)
    plt.show()

def plot_sources_per_generation(df, vertical_markers=None):
    sns.set_theme(style="ticks")
    df['generation'] = df['generation'].astype('category')
    f, ax = plt.subplots(figsize=fig_size)
    sns.despine(f)
    sns.histplot(
        data=df,
        x="generation",
        hue='source',
        multiple="stack",
        edgecolor=".3",
        linewidth=.5,
        ax=ax
    )
    plt.xlabel('generation')
    plt.ylabel('Count')
    plt.title('Histogram of Sources by generation')
    # Add vertical markers
    if vertical_markers is not None:
        for gen in vertical_markers:
            plt.axvline(x=gen, color='red', linestyle='--', alpha=0.7)
    plt.show()

def plot_operation_and_sources_per_generation(df, vertical_markers=None):

    # Ensure 'generation' is categorical for proper ordering on x-axis
    df['generation'] = df['generation'].astype('category')

    f, ax = plt.subplots(figsize=fig_size)
    sns.despine(f)

    # Plot histogram with stacked bars
    sns.histplot(
        data=df,
        x="generation",
        hue='Source+Operation',
        multiple="stack",
        edgecolor=".3",
        linewidth=.5,
    )

    # Add labels and title
    plt.xlabel('generation')
    plt.ylabel('Count')
    plt.title('Histogram of Source+Operation by generation')
    # Add vertical markers
    if vertical_markers is not None:
        for gen in vertical_markers:
            plt.axvline(x=gen, color='red', linestyle='--', alpha=0.7)
    # Show the plot
    plt.show()

def print_sources_per_operation(df):
    # Get counts of each source per operation
    #df2 = df[df['generation'].astype(int) > 29]
    df2 = df
    count_df = df2.groupby(['operation', 'source']).size().unstack(fill_value=0)

    # Calculate percentages
    percentage_df = count_df.div(count_df.sum(axis=1), axis=0) * 100

    # Combine counts and percentages into a multi-level DataFrame
    result = pd.concat({
        'Count': count_df,
        'Percentage': percentage_df.round(1)  # Round to 1 decimal place
    }, axis=1)

    # Create a combined string representation
    combined_df = count_df.astype(str) + " (" + percentage_df.round(1).astype(str) + "%)"

    print("\nCombined count and percentage display:")
    print(combined_df)
    print(df.head())

def plot_crossover_type(df, vertical_markers=None):
    if not is_crossover_df(df):
        df = get_crossover_df(df)

    sns.set_theme(style="ticks")

    # Ensure 'generation' is categorical for proper ordering on x-axis
    df['generation'] = df['generation'].astype('category')

    f, ax = plt.subplots(figsize=fig_size)
    sns.despine(f)

    # Plot histogram with stacked bars
    sns.histplot(
        data=df,
        x="generation",
        hue='crossover_type',
        multiple="stack",
        edgecolor=".3",
        linewidth=.5,
    )

    # Add labels and title
    plt.xlabel('generation')
    plt.ylabel('Count')
    plt.title('Histogram of Crossover Type by generation')
    # Add vertical markers
    if vertical_markers is not None:
        for gen in vertical_markers:
            plt.axvline(x=gen, color='red', linestyle='--', alpha=0.7)
    # Show the plot
    plt.show()

def plot_mutation_type(df, vertical_markers=None):
    if not is_mutation_df(df):
        df = get_mutation_df(df)

    sns.set_theme(style="ticks")

    # Ensure 'generation' is categorical for proper ordering on x-axis
    df['generation'] = df['generation'].astype('category')

    f, ax = plt.subplots(figsize=fig_size)
    sns.despine(f)

    # Plot histogram with stacked bars
    sns.histplot(
        data=df,
        x="generation",
        hue='mutation_type',
        multiple="stack",
        edgecolor=".3",
        linewidth=.5,
    )

    # Add labels and title
    plt.xlabel('generation')
    plt.ylabel('Count')
    plt.title('Histogram of Mutation Type by generation')
    # Add vertical markers
    if vertical_markers is not None:
        for gen in vertical_markers:
            plt.axvline(x=gen, color='red', linestyle='--', alpha=0.7)
    # Show the plot
    plt.show()

def plot_fitness_difference_after_crossover(df, df_crossover, type=None):
    if df_crossover is None:
        df_crossover = get_crossover_df(df)

    if type is not None:
        df_crossover = df_crossover[df_crossover['crossover_type'] == type]

    # Calculate parent fitness averages and difference for each crossover
    def get_crossover_fitness_diff(df, df_crossover):
        parent_avg_fitness = []
        fitness_diff = []

        for _, row in df_crossover.iterrows():
            # Get parent fitness values
            p1_fit = df[df['genetic_id'] == row['parent_1']]['fitness']
            p2_fit = df[df['genetic_id'] == row['parent_2']]['fitness']
            # Use .head(1) to avoid duplicates, .item() to get value
            p1_fit = p1_fit.head(1).item() if not p1_fit.empty else np.nan
            p2_fit = p2_fit.head(1).item() if not p2_fit.empty else np.nan
            avg_fit = np.nanmean([p1_fit, p2_fit])
            parent_avg_fitness.append(avg_fit)
            fitness_diff.append(row['fitness'] - avg_fit)

        df_crossover = df_crossover.copy()
        df_crossover['parent_avg_fitness'] = parent_avg_fitness
        df_crossover['fitness_diff'] = fitness_diff
        return df_crossover

    df_crossover = get_crossover_fitness_diff(df, df_crossover)
    # Visualization
    mean_diff = np.nanmean(df_crossover['fitness_diff'])
    std_diff = np.nanstd(df_crossover['fitness_diff'])

    plt.figure(figsize=(10, 6))
    plt.hist(df_crossover['fitness_diff'].dropna(), bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(mean_diff, color='red', linestyle='--', label=f'Mean: {mean_diff:.4f}')
    plt.axvline(mean_diff + std_diff, color='green', linestyle=':', label=f'+1 Std: {mean_diff + std_diff:.4f}')
    plt.axvline(mean_diff - std_diff, color='green', linestyle=':', label=f'-1 Std: {mean_diff - std_diff:.4f}')
    if type is not None:
        plt.title(f'Distribution of Fitness Difference After Crossover Type: {type}\nMean fitness difference: {mean_diff:.4f}\nStd fitness difference: {std_diff:.4f}')
    else:
        plt.title(f'Distribution of Fitness Difference After Crossover \nMean fitness difference: {mean_diff:.4f}\nStd fitness difference: {std_diff:.4f}')
    plt.xlabel('Fitness Difference')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_fitness_difference_after_mutation(df, df_mutation, type=None):
    if df_mutation is None:
        df_mutation = get_mutation_df(df)

    if type is not None:
        df_mutation = df_mutation[df_mutation['mutation_type'] == type]

    # Calculate parent fitness averages and difference for each mutation
    def get_mutation_fitness_diff(df, df_mutation):
        parent_avg_fitness = []
        fitness_diff = []

        for _, row in df_mutation.iterrows():
            # Get parent fitness values
            p1_fit = df[df['genetic_id'] == row['parent_1']]['fitness']
            # Use .head(1) to avoid duplicates, .item() to get value
            p1_fit = p1_fit.head(1).item() if not p1_fit.empty else np.nan
            avg_fit = np.nanmean([p1_fit])
            parent_avg_fitness.append(avg_fit)
            fitness_diff.append(row['fitness'] - avg_fit)

        df_mutation = df_mutation.copy()
        df_mutation['parent_avg_fitness'] = parent_avg_fitness
        df_mutation['fitness_diff'] = fitness_diff
        return df_mutation

    df_mutation = get_mutation_fitness_diff(df, df_mutation)
    
    # Visualization
    mean_diff = np.nanmean(df_mutation['fitness_diff'])
    std_diff = np.nanstd(df_mutation['fitness_diff'])

    plt.figure(figsize=(10, 6))
    plt.hist(df_mutation['fitness_diff'].dropna(), bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(mean_diff, color='red', linestyle='--', label=f'Mean: {mean_diff:.4f}')
    plt.axvline(mean_diff + std_diff, color='green', linestyle=':', label=f'+1 Std: {mean_diff + std_diff:.4f}')
    plt.axvline(mean_diff - std_diff, color='green', linestyle=':', label=f'-1 Std: {mean_diff - std_diff:.4f}')
    if type is not None:
        plt.title(f'Distribution of Fitness Difference After Mutation Type: {type}\nMean fitness difference: {mean_diff:.4f}\nStd fitness difference: {std_diff:.4f}')
    else:
        plt.title(f'Distribution of Fitness Difference After Mutation \nMean fitness difference: {mean_diff:.4f}\nStd fitness difference: {std_diff:.4f}')
    plt.xlabel('Fitness Difference')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.show()