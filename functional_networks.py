#functional_networks.py

# Analysis from:
# Network-level changes in the brain underlie fear memory strength
# Josue Haubrich & Karim Nader eLife, 2023 https://doi.org/10.7554/eLife.88172.3

# Conversion of the R code from:
# - [https://github.com/johaubrich/Networks/](https://github.com/johaubrich/Networks/)

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import pearsonr, spearmanr, kruskal, wilcoxon
from statsmodels.stats.multitest import multipletests
from sklearn.utils import resample
import statsmodels.api as sm
from scipy.spatial.distance import jaccard
import community as community_louvain
from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib import gridspec

def corr_matrix(df, p_adjust_method='none', corr_type='pearson'):
    """
    Input: DataFrame with headers as titles (brain regions)
           Whether or not to apply a p-value adjustment method
           (e.g., 'fdr', 'bonferroni' etc.).
           Whether to use 'pearson' or 'spearman' correlation.
           NOTE: assume undirected graph!
    
    Output: List of two DataFrames corresponding to 1) all pairwise correlations
            and 2) all associated un-adjusted p-values
    """
    def compute_corr(df, corr_type):
        corr_func = pearsonr if corr_type == 'pearson' else spearmanr
        corr_matrix = np.zeros((df.shape[1], df.shape[1]))
        p_matrix = np.zeros((df.shape[1], df.shape[1]))
        for i in range(df.shape[1]):
            for j in range(i, df.shape[1]):
                corr, p = corr_func(df.iloc[:, i], df.iloc[:, j])
                corr_matrix[i, j] = corr_matrix[j, i] = corr
                p_matrix[i, j] = p_matrix[j, i] = p
        return corr_matrix, p_matrix

    corr, p_vals = compute_corr(df, corr_type)
    df_corr = pd.DataFrame(corr, columns=df.columns, index=df.columns)
    df_pvalue = pd.DataFrame(p_vals, columns=df.columns, index=df.columns)
    
    if p_adjust_method != 'none':
        upper_indices = np.triu_indices_from(p_vals, k=1)
        p_vals_upper = p_vals[upper_indices]
        _, p_vals_adj, _, _ = multipletests(p_vals_upper, method=p_adjust_method)
        p_vals[upper_indices] = p_vals_adj
        p_vals[np.tril_indices_from(p_vals)] = np.nan
        df_pvalue = pd.DataFrame(p_vals, columns=df.columns, index=df.columns)
    
    return {"corr": df_corr, "pvalue": df_pvalue}


def corr_matrix_threshold(df, negs=False, thresh=0.01, thresh_param='p', p_adjust_method='bonferroni', corr_type='pearson'):
    """
    Input: DataFrame with headers as titles (brain regions) of counts etc.
           threshold and threshold parameter (p, r, or cost), whether or not to keep negative correlations.
           and the p-value adjustment method if using p-value as threshold parameter.
           and whether to use pearson or spearman correlation
    
    Output: DataFrame of correlations thresholded at p < threshold
    
    NOTE: Removes diagonals
    """
    results = corr_matrix(df, p_adjust_method=p_adjust_method, corr_type=corr_type)
    df_corr = results['corr']
    df_pvalue = results['pvalue']
    
    np.fill_diagonal(df_corr.values, 0)
    np.fill_diagonal(df_pvalue.values, np.nan)
    
    df_pvalue.replace([np.inf, -np.inf, np.nan], 1, inplace=True)
    df_corr.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    
    if not negs:
        df_corr[df_corr < 0] = 0
    
    if thresh_param.lower() == 'p':
        df_corr[df_pvalue >= thresh] = 0
    elif thresh_param.lower() == 'r':
        df_corr[np.abs(df_corr) <= thresh] = 0
    elif thresh_param.lower() == 'cost':
        r_threshold = np.quantile(np.abs(df_corr.values), 1 - thresh)
        df_corr[np.abs(df_corr) <= r_threshold] = 0
    else:
        raise ValueError("Invalid thresholding parameter")
    
    return df_corr


def calculate_nodal_efficiency(G, weighted=False, normalized=False):
    # Invert weights for shortest path calculation as in R function
    if weighted:
        G = G.copy()
        for u, v, d in G.edges(data=True):
            d['weight'] = 1 / abs(d['weight']) if 'weight' in d else 1.0

    def efficiency_func(G):
        nodes = G.nodes()
        efficiency = {}
        for n in nodes:
            if weighted:
                lengths = dict(nx.single_source_dijkstra_path_length(G, n, weight='weight'))
            else:
                lengths = dict(nx.single_source_shortest_path_length(G, n))
            
            # Handling unconnected nodes, as in R's shortest.paths function
            inv_lengths = {k: 1/v for k, v in lengths.items() if v != 0}
            total_nodes = len(nodes) - 1  # excluding the node itself
            # Handle unconnected nodes
            if len(inv_lengths) < total_nodes:
                missing_nodes = total_nodes - len(inv_lengths)
                inv_lengths.update({k: 0 for k in nodes if k not in inv_lengths})
            
            efficiency[n] = sum(inv_lengths.values()) / total_nodes
            
            # print(f"Node {n}:")
            # print("  Shortest path lengths:", lengths)
            # print("  Inverted lengths:", inv_lengths)
            # print("  Efficiency:", efficiency[n])
        
        return efficiency

    eff = efficiency_func(G)
    
    # Handle non-finite values (set them to zero)
    for k, v in eff.items():
        if not np.isfinite(v):
            eff[k] = 0

    if normalized:
        max_eff = max(eff.values(), default=1)
        eff = {k: v/max_eff for k, v in eff.items()}
    
    return pd.Series(eff)

def get_centrality_measures(G, weighted=False, nodal_efficiency_calc=False, normalized=False, min_max_normalization=False):
    if G.number_of_edges() == 0:
        zeros = np.zeros(G.number_of_nodes())
        return pd.DataFrame({"degree": zeros, "betweenness": zeros, "eigenvector": zeros, 
                             "closeness": zeros, "transitivity": zeros}, index=G.nodes)
    
    degree = {node: val for node, val in G.degree()}  # Get the integer degree centrality
    
    G_pos = G.copy()
    for u, v, d in G_pos.edges(data=True):
        d['weight_abs'] = abs(d['weight']) if 'weight' in d else 1.0
        d['weight'] = 1.0 / d['weight_abs'] if weighted else 1.0
    
    eigenvector = nx.eigenvector_centrality_numpy(G, weight='weight' if weighted else None)
    
    between = nx.betweenness_centrality(G_pos, normalized=normalized, weight='weight' if weighted else None)

    close = nx.closeness_centrality(G_pos, distance='weight' if weighted else None, wf_improved=False)
    
    trans = nx.clustering(G, weight='weight' if weighted else None)
    
    df = pd.DataFrame({"degree": degree, "betweenness": between, "eigenvector": eigenvector, 
                       "closeness": close, "transitivity": trans})
    
    if nodal_efficiency_calc:
        df["efficiency"] = calculate_nodal_efficiency(G, weighted=weighted, normalized=normalized)
    
    if min_max_normalization:
        df = (df - df.min()) / (df.max() - df.min())
        df["degree.betweenness"] = df["degree"] + df["betweenness"]
        df["transitivity"] = trans  # Do not normalize transitivity because it is already normalized
    
    return df



def global_efficiency(G, weighted=True):
    if G.number_of_edges() == 0:
        return 0
    lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight' if weighted else None))
    inv_lengths = {k: {kk: 1/vv for kk, vv in v.items() if vv != 0} for k, v in lengths.items()}
    eff = {k: np.mean(list(v.values())) for k, v in inv_lengths.items()}
    return np.mean(list(eff.values()))

def watts_strogatz_model(G, iterations=1000, trans_match_iter=100):
    n = len(G)
    k = int(np.mean([d for _, d in G.degree()]))
    p = nx.transitivity(G)
    
    TR = []
    GE = []
    deg_dist = np.zeros((iterations, n))
    clust_dist = np.zeros((iterations, n))
    
    for i in range(iterations):
        W = nx.watts_strogatz_graph(n, k, p)
        if W.number_of_edges() > 0:
            TR.append(nx.transitivity(W))
            GE.append(global_efficiency(W, weighted=False))
            deg_dist[i, :] = sorted([d for n, d in W.degree()])
            clust_dist[i, :] = sorted(nx.clustering(W).values())
        else:
            TR.append(0)
            GE.append(0)
            deg_dist[i, :] = np.zeros(n)
            clust_dist[i, :] = np.zeros(n)
    
    df_all_out = pd.DataFrame({"degree": deg_dist.flatten(), "transitivity": clust_dist.flatten()})
    
    TR_out = [np.mean(TR), np.std(TR)]
    GE_out = [np.mean(GE), np.std(GE)]
    
    df_out = pd.DataFrame({"Global.efficiency": GE_out, "Transitivity": TR_out})
    
    return df_out, np.mean(deg_dist, axis=0), np.mean(clust_dist, axis=0), df_all_out


def nodal_efficiency_ci(G, group, network_type, num_samples):
    efficiency = calculate_nodal_efficiency(G, weighted=True)
    mean_eff = efficiency.mean()
    sd_eff = efficiency.std()
    ci_eff = 1.96 * (sd_eff / np.sqrt(num_samples))
    return {"Group": group, "Network": network_type, "Mean": mean_eff, "SD": sd_eff, "CI": ci_eff}

def conf_int_globaleff(group1, group2, n1, n2):
    mean1 = group1['Mean']
    mean2 = group2['Mean']
    sd1 = group1['SD']
    sd2 = group2['SD']
    se = np.sqrt(sd1**2/n1 + sd2**2/n2)
    diffmeans = mean2 - mean1
    confint = diffmeans + np.array([-1, 1]) * 1.96 * se

    result = {
        "Group1": group1['Group'],
        "Group2": group2['Group'],
        "Mean_Group1": mean1,
        "Mean_Group2": mean2,
        "Mean_diff": diffmeans,
        "CI95diff_low": confint[0],
        "CI95diff_up": confint[1]
    }
    return pd.DataFrame([result])




### Visualization functions


def vis_corr_matrix(df, ax=None, cbar_ax=None, cbar=False):
    """
    Visualize the correlation matrix.
    
    Parameters:
    df (DataFrame): The input data frame (a correlation matrix).
    ax (matplotlib.axes._subplots.AxesSubplot, optional): The axis on which to plot the heatmap. Defaults to None.
    cbar_ax (matplotlib.axes._subplots.AxesSubplot, optional): The axis on which to plot the colorbar. Defaults to None.
    cbar (bool, optional): Whether to include the colorbar. Defaults to False.
    
    Returns:
    ax (matplotlib.axes._subplots.AxesSubplot): The axis with the heatmap.
    """
    # Calculate the correlation matrix
    corr_matrix = df.copy()
    
    # Set the diagonal to NaN
    np.fill_diagonal(corr_matrix.values, np.nan)
    
    # Reverse the order of rows
    corr_matrix = corr_matrix.iloc[::-1]

    # Plot the heatmap
    sns.heatmap(corr_matrix, annot=False, cmap='bwr', vmin=-1, vmax=1, center=0, 
                cbar=cbar, cbar_ax=cbar_ax, cbar_kws={'label': 'R value'}, annot_kws={"size": 10},
                square=True, ax=ax)
    
    # Set the title and adjust the aspect
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    return ax

def plot_multi_corr_matrices(mat1, mat2, mat3, titles=['Network NS', 'Network 2S', 'Network 10S']):
    """
    Plot three correlation matrices side by side with a single colorbar.
    
    Parameters:
    mat1 (DataFrame): The first correlation matrix.
    mat2 (DataFrame): The second correlation matrix.
    mat3 (DataFrame): The third correlation matrix.
    titles (list): List of titles for each subplot.
    """
    # Create a multi-subplot figure with a single colorbar
    fig = plt.figure(figsize=(24, 8))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.3)

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    cbar_ax = plt.subplot(gs[3])

    vis_corr_matrix(mat1, ax=ax1, cbar=False)
    ax1.set_title(titles[0])

    vis_corr_matrix(mat2, ax=ax2, cbar=False)
    ax2.set_title(titles[1])

    vis_corr_matrix(mat3, ax=ax3, cbar_ax=cbar_ax, cbar=True)
    ax3.set_title(titles[2])

    # Adjust the colorbar to match the height of the heatmaps
    fig.subplots_adjust(right=0.85)
    cbar_ax.set_position([0.86, 0.2, 0.02, 0.7])

    plt.show()





def plot_circular_networks(corr_matrices, titles, colors):
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    for ax, corr_matrix, title, color in zip(axes, corr_matrices, titles, colors):
        G = nx.Graph()
        
        # Create edges with weights based on the correlation matrix
        for i, row in enumerate(corr_matrix.index):
            for j, col in enumerate(corr_matrix.columns):
                if i < j:
                    weight = corr_matrix.at[row, col]
                    if weight != 0:  # Only add edges for non-zero weights
                        G.add_edge(row, col, weight=weight)
        
        pos = nx.circular_layout(G)
        
        edges = G.edges(data=True)
        weights = [max(abs(d['weight']), 0.1) * 10 for (_, _, d) in edges]  # Ensure non-zero weights, increased scaling factor
        edge_colors = [d['weight'] for (_, _, d) in edges]
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=color, edgecolors='black', ax=ax)

        # Draw edges with colormap
        norm = mcolors.Normalize(vmin=-1, vmax=1)
        edge_cmap = cm.ScalarMappable(norm=norm, cmap='viridis')
        edge_colors = [edge_cmap.to_rgba(w) for w in edge_colors]
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, edge_color=edge_colors, ax=ax)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_color='black', ax=ax)
        
        ax.set_title(title)
        ax.axis('off')

    # Add a colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(edge_cmap, cax=cbar_ax)
    cbar.set_label('R value')
    
    # Add custom legends for edge weights
    weight_legend_elements = [
        Line2D([0], [0], color='black', lw=0.25*10, label='Weight 0.25'),
        Line2D([0], [0], color='black', lw=0.50*10, label='Weight 0.50'),
        Line2D([0], [0], color='black', lw=0.75*10, label='Weight 0.75'),
        Line2D([0], [0], color='black', lw=1.00*10, label='Weight 1.00')
    ]
    fig.legend(handles=weight_legend_elements, loc='lower right')

    plt.show()



    
def plot_network(G, title, node_color, edge_color, ax, layout='spring'):
    """
    Plot the network on a given axis.
    
    Parameters:
    G (networkx.Graph): The network graph to plot.
    title (str): The title of the plot.
    node_color (str): Color of the nodes.
    edge_color (str): Color of the edges.
    ax (matplotlib.axes._subplots.AxesSubplot): The axis on which to plot the network.
    layout (str): The layout for the network (default is 'spring').
    """
    # Choose the layout
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=42)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    elif layout == 'shell':
        pos = nx.shell_layout(G)
    elif layout == 'stress':
        pos = nx.spring_layout(G, k=None, iterations=100, seed=42)
    else:
        raise ValueError(f"Unsupported layout: {layout}")
    
    # Community detection
    partition = community_louvain.best_partition(G)
    
    # Draw communities using convex hulls
    communities = {}
    for node, community in partition.items():
        communities.setdefault(community, []).append(node)
    
    for nodes in communities.values():
        points = np.array([pos[node] for node in nodes])
        if len(points) > 2:  # Need at least 3 points to form a convex hull
            hull = ConvexHull(points)
            polygon = plt.Polygon(points[hull.vertices], alpha=0.3, color='tan')
            ax.add_patch(polygon)

    # Compute the node sizes based on degree
    degrees = dict(G.degree())
    node_size = [100 + 200 * degrees[n] for n in G.nodes()]
    
    # Compute the edge widths based on weights
    edges = G.edges(data=True)
    edge_widths = [d['weight'] * 5 for (_, _, d) in edges]

    # Draw the network
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_color, alpha=1, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, alpha=0.9, edgecolors='black', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='black', ax=ax)
    
    ax.set_title(title)
    ax.axis('off')

def plot_multi_networks(graphs, titles, colors, layout):
    """
    Plot multiple network graphs in a multi-panel figure.
    
    Parameters:
    graphs (list): List of networkx.Graph objects.
    titles (list): List of titles for each subplot.
    colors (list): List of node colors for each subplot.
    layout (str): The layout for the networks.
    """
    fig, axes = plt.subplots(1, len(graphs), figsize=(24, 8))

    for i, (G, title, color) in enumerate(zip(graphs, titles, colors)):
        plot_network(G, title, color, 'black', ax=axes[i], layout=layout)

    # Create custom legends for edge weights
    weight_legend_elements = [
        Line2D([0], [0], color='black', lw=3, label='Weight 0.6'),
        Line2D([0], [0], color='black', lw=5, label='Weight 0.8'),
        Line2D([0], [0], color='black', lw=7, label='Weight 1.0')
    ]

    # Create custom legends for node degree
    degree_legend_elements = [
        Line2D([0], [0], linestyle='none', lw=1, marker='o', markerfacecolor='none', color='grey', markersize=np.sqrt(100), label='Degree 1'),
        Line2D([0], [0], linestyle='none', lw=1, marker='o', markerfacecolor='none', color='grey', markersize=np.sqrt(300), label='Degree 3'),
        Line2D([0], [0], linestyle='none', lw=1, marker='o', markerfacecolor='none', color='grey', markersize=np.sqrt(500), label='Degree 5')
    ]

    # Combine the legends
    legend_elements = weight_legend_elements + degree_legend_elements

    # Add legends
    # axes[-1].legend(handles=legend_elements, loc='lower right')
    axes[-1].legend(handles=legend_elements, loc='lower right', handleheight=2, labelspacing=1.5, borderaxespad=2)

    plt.tight_layout()
    plt.show()




def create_subplot_figure(cent_ns, cent_2s, cent_10s, palette):
    # Merge all centrality measures into a single DataFrame
    merged_centrality = pd.concat([cent_ns, cent_2s, cent_10s])

    # Convert the DataFrame to a long format for easier plotting and analysis
    merged_centrality_long = pd.melt(merged_centrality, id_vars=['Structure', 'Group'], 
                                     var_name='Centrality', value_name='Value')
    
    # Define centrality measures to plot
    centrality_measures = ['degree', 'betweenness', 'efficiency']
    y_labels = ['Degree', 'Betweenness', 'Nodal Efficiency']
    
    # Create the figure
    fig, axes = plt.subplots(3, 4, figsize=(24, 18), sharey='row')
    
    for i, (centrality, y_label) in enumerate(zip(centrality_measures, y_labels)):
        # Filter the data for the current centrality measure
        cent_data = merged_centrality_long[merged_centrality_long['Centrality'] == centrality]
        
        # Plot for each group
        for j, group in enumerate(['NS', '2S', '10S']):
            ax = axes[i, j]
            group_data = cent_data[cent_data['Group'] == group]
            group_data = group_data.sort_values(by='Value', ascending=False)
            sns.barplot(x='Structure', y='Value', data=group_data, ax=ax, color=palette[group])
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            if j == 0:
                ax.set_ylabel(y_label, fontsize=16)
            else:
                ax.set_ylabel('')
            ax.set_xlabel('')  # Remove 'Group' label
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
        
        # Plot comparison between groups
        ax = axes[i, 3]
        sns.barplot(x='Group', y='Value', data=cent_data, ax=ax, palette=palette, capsize=.2)
        ax.set_ylabel(y_label, fontsize=16)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        
        # Perform statistical tests and add annotations
        ns_data = cent_data[cent_data['Group'] == 'NS']['Value']
        s2_data = cent_data[cent_data['Group'] == '2S']['Value']
        s10_data = cent_data[cent_data['Group'] == '10S']['Value']
        
        kruskal_result = kruskal(ns_data, s2_data, s10_data)
        ax.text(1, max(cent_data['Value']) + 0.1, f'Kruskal-Wallis p = {kruskal_result.pvalue:.3f}', ha='center')
        
        wilcoxon_result_ns_2s = wilcoxon(ns_data, s2_data)
        wilcoxon_result_ns_10s = wilcoxon(ns_data, s10_data)
        wilcoxon_result_2s_10s = wilcoxon(s2_data, s10_data)
        # ax.text(1, max(cent_data['Value']), f'NS-2S p = {wilcoxon_result_ns_2s.pvalue:.3f}', ha='center')
        # ax.text(1, max(cent_data['Value']) - 0.1, f'NS-10S p = {wilcoxon_result_ns_10s.pvalue:.3f}', ha='center')
        # ax.text(1, max(cent_data['Value']) - 0.2, f'2S-10S p = {wilcoxon_result_2s_10s.pvalue:.3f}', ha='center')
    
    plt.tight_layout()
    plt.show()
