#!/usr/bin/env python3
"""
Watts-Strogatz Model Visualization
=================================

Recreates the classic visualization showing the transition from regular lattice
to small-world to random network as β increases from 0 to 1.

Shows three networks:
- Regular lattice (β = 0)
- Small-world (β = 0.1) 
- Random network (β = 1.0)

Author: Network Analysis Course
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math
from matplotlib.patches import Arc, Wedge
import warnings
warnings.filterwarnings('ignore')

class WattsStrogatzVisualizer:
    """Visualize the Watts-Strogatz model transition."""
    
    def __init__(self, n=20, k=4):
        """
        Initialize the visualizer.
        
        Parameters:
        -----------
        n : int
            Number of nodes (default: 20)
        k : int
            Each node is connected to k nearest neighbors in ring topology (default: 4)
        """
        self.n = n
        self.k = k
        
    def create_watts_strogatz_graph(self, beta, seed=42):
        """
        Create a Watts-Strogatz graph with given rewiring probability.
        
        Parameters:
        -----------
        beta : float
            Rewiring probability (0 = regular, 1 = random)
        seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        networkx.Graph
            The generated Watts-Strogatz graph
        """
        return nx.watts_strogatz_graph(self.n, self.k, beta, seed=seed)
    
    def get_circular_positions(self, scale=1.0):
        """Get positions for nodes arranged in a circle."""
        positions = {}
        for i in range(self.n):
            angle = 2 * math.pi * i / self.n
            x = scale* math.cos(angle)
            y = scale *math.sin(angle)
            positions[i] = (x, y)
        return positions
    
    def draw_node_neighborhood_highlight(self, ax, positions, target_node=0, highlight_color='hotpink', alpha=0.3):
        """
        Draw a pink transparent arc highlighting a node and its neighbors.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axes to draw on
        positions : dict
            Node positions
        target_node : int
            The node to highlight (default: 0)
        highlight_color : str
            Color for the highlight (default: 'hotpink')
        alpha : float
            Transparency level (default: 0.3)
        """
        # Calculate node radius from node_size (150)
        # node_size in matplotlib is area, so radius = sqrt(node_size/π)
        node_radius = math.sqrt(150 / math.pi) / 100  # Scale to match coordinate system
        
        # Nodes are positioned at radius 1.4, so:
        circle_radius = 1.4
        # Make arc deeper by extending both inward and outward
        depth_extension = node_radius *0.5 # Make arc 50% deeper
        inner_radius = circle_radius - node_radius - depth_extension  # Extend inward
        outer_radius = circle_radius + node_radius + depth_extension  # Extend outward
        
        # Find neighbors in the regular lattice (k=4 means 2 neighbors on each side)
        neighbors = []
        for i in range(1, self.k//2 + 1):
            left_neighbor = (target_node - i) % self.n
            right_neighbor = (target_node + i) % self.n
            neighbors.extend([left_neighbor, right_neighbor])
        
        # Include the target node itself
        highlighted_nodes = [target_node] + neighbors
        
        # Calculate the angular span for the highlight
        angles = []
        for node in highlighted_nodes:
            angle = 2 * math.pi * node / self.n
            angles.append(angle)
        
        # Sort angles and handle wraparound
        angles.sort()
        
        # Calculate angular span and handle potential wraparound
        min_angle = min(angles)
        max_angle = max(angles)
        
        # Check if we need to handle wraparound (when neighbors span across 0/2π boundary)
        angular_span = max_angle - min_angle
        needs_wraparound = angular_span > math.pi  # If span > π, we're wrapping around
        
        # Extend to align with outer edge of last neighbor nodes and add extra width
        angular_node_span = math.asin(node_radius / circle_radius)
        extra_width = 0.02  # Add extra angular padding to make arc wider
        
        if needs_wraparound:
            # Handle wraparound case: draw two separate arcs
            # Find the gap in the middle and split there
            angles_sorted = sorted(angles)
            max_gap = 0
            split_point = 0
            
            for i in range(len(angles_sorted)):
                next_i = (i + 1) % len(angles_sorted)
                if next_i == 0:  # Wraparound gap
                    gap = (2 * math.pi - angles_sorted[i]) + angles_sorted[0]
                else:
                    gap = angles_sorted[next_i] - angles_sorted[i]
                
                if gap > max_gap:
                    max_gap = gap
                    split_point = i
            
            # Split the angles into two groups
            if split_point == len(angles_sorted) - 1:  # Split at wraparound
                group1 = [a for a in angles_sorted if a <= math.pi]
                group2 = [a for a in angles_sorted if a > math.pi]
            else:
                split_angle = (angles_sorted[split_point] + angles_sorted[split_point + 1]) / 2
                group1 = [a for a in angles_sorted if a <= split_angle]
                group2 = [a for a in angles_sorted if a > split_angle]
            
            # Draw first arc
            if group1:
                min1, max1 = min(group1) - angular_node_span - extra_width, max(group1) + angular_node_span + extra_width
                wedge1_outer = Wedge((0, 0), outer_radius, 
                                   math.degrees(min1), math.degrees(max1),
                                   facecolor=highlight_color, alpha=alpha, 
                                   edgecolor='none', zorder=0)
                wedge1_inner = Wedge((0, 0), inner_radius, 
                                   math.degrees(min1), math.degrees(max1),
                                   facecolor='white', alpha=1.0, 
                                   edgecolor='none', zorder=1)
                ax.add_patch(wedge1_outer)
                ax.add_patch(wedge1_inner)
            
            # Draw second arc
            if group2:
                min2, max2 = min(group2) - angular_node_span - extra_width, max(group2) + angular_node_span + extra_width
                wedge2_outer = Wedge((0, 0), outer_radius, 
                                   math.degrees(min2), math.degrees(max2),
                                   facecolor=highlight_color, alpha=alpha, 
                                   edgecolor='none', zorder=0)
                wedge2_inner = Wedge((0, 0), inner_radius, 
                                   math.degrees(min2), math.degrees(max2),
                                   facecolor='white', alpha=1.0, 
                                   edgecolor='none', zorder=1)
                ax.add_patch(wedge2_outer)
                ax.add_patch(wedge2_inner)
        else:
            # Normal case: single arc
            min_angle -= angular_node_span + extra_width
            max_angle += angular_node_span + extra_width
            
            # Create arc ring (outer radius - inner radius)
            wedge_outer = Wedge((0, 0), outer_radius, 
                              math.degrees(min_angle), math.degrees(max_angle),
                              facecolor=highlight_color, alpha=alpha, 
                              edgecolor='none', zorder=0)
            wedge_inner = Wedge((0, 0), inner_radius, 
                              math.degrees(min_angle), math.degrees(max_angle),
                              facecolor='white', alpha=1.0, 
                              edgecolor='none', zorder=1)
            ax.add_patch(wedge_outer)
            ax.add_patch(wedge_inner)
    
    def classify_edges(self, graph, positions):
        """
        Classify edges as local (short-range) or long-range connections.
        
        Parameters:
        -----------
        graph : networkx.Graph
            The graph to analyze
        positions : dict
            Node positions
            
        Returns:
        --------
        tuple
            (local_edges, long_range_edges)
        """
        local_edges = []
        long_range_edges = []
        
        for edge in graph.edges():
            u, v = edge
            # Calculate the shortest distance around the circle
            diff = abs(u - v)
            circular_distance = min(diff, self.n - diff)
            
            # Edges to immediate neighbors (distance <= k/2) are local
            if circular_distance <= self.k // 2:
                local_edges.append(edge)
            else:
                long_range_edges.append(edge)
                
        return local_edges, long_range_edges
    
    def draw_curved_edges(self, ax, positions, edges, color, width, base_curvature):
        """Draw curved edges using matplotlib Path with inward curves scaled by distance"""
        from matplotlib.path import Path
        from matplotlib.patches import PathPatch
        
        for edge in edges:
            node1, node2 = edge
            x1, y1 = positions[node1]
            x2, y2 = positions[node2]
            
            # Calculate the midpoint
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            # Calculate distance between nodes
            dx = x2 - x1
            dy = y2 - y1
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance > 0:
                # Calculate node sequence distance (how many steps apart in the circle)
                # For a circle of n nodes, the sequence distance is the minimum of:
                # - direct distance: abs(node2 - node1)
                # - wraparound distance: n - abs(node2 - node1)
                n_nodes = len(positions)
                direct_distance = abs(node2 - node1)
                wraparound_distance = n_nodes - direct_distance
                sequence_distance = min(direct_distance, wraparound_distance)
                
                # Scale curvature with sequence distance
                # Neighbors (distance=1) get zero curvature, max distance gets full curvature
                max_sequence_distance = n_nodes // 2  # Maximum possible sequence distance
                if sequence_distance <= 1:
                    scaled_curvature = 0  # Zero curvature for neighbors
                else:
                    distance_factor = (sequence_distance - 1) / (max_sequence_distance - 1)
                    scaled_curvature = base_curvature * distance_factor
                
                # Direction toward center of circle (inward)
                # Center is at (0, 0), so direction from midpoint to center
                center_direction_x = -mid_x
                center_direction_y = -mid_y
                center_distance = math.sqrt(center_direction_x**2 + center_direction_y**2)
                
                if center_distance > 0:
                    # Normalize the direction toward center
                    center_direction_x /= center_distance
                    center_direction_y /= center_distance
                    
                    # Control point for the curve (toward center)
                    control_x = mid_x + scaled_curvature * center_direction_x
                    control_y = mid_y + scaled_curvature * center_direction_y
                    
                    # Create quadratic Bezier curve
                    verts = [(x1, y1), (control_x, control_y), (x2, y2)]
                    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
                    
                    path = Path(verts, codes)
                    patch = PathPatch(path, facecolor='none', edgecolor=color, 
                                    linewidth=width, alpha=0.8)
                    ax.add_patch(patch)

    def _avg_shortest_path_length_connected(self, G):
        """Average shortest path length on the largest connected component (exact)."""
        try:
            if nx.is_connected(G):
                return nx.average_shortest_path_length(G)
            largest_cc = max(nx.connected_components(G), key=len)
            H = G.subgraph(largest_cc).copy()
            return nx.average_shortest_path_length(H)
        except Exception:
            return float('inf')

    def _largest_connected_subgraph(self, G):
        """Return the largest connected component subgraph."""
        try:
            if G.number_of_nodes() == 0:
                return G
            if nx.is_connected(G):
                return G
            largest_cc = max(nx.connected_components(G), key=len)
            return G.subgraph(largest_cc).copy()
        except Exception:
            return G

    def _approx_average_shortest_path_length_connected(self, G, sample_size=300, seed=None):
        """Fast approximation of average shortest path length on largest component.

        Samples up to `sample_size` source nodes and averages single-source
        shortest-path lengths to all reachable nodes. This is typically much
        faster than computing all-pairs for larger n, while tracking trends.
        """
        H = self._largest_connected_subgraph(G)
        n = H.number_of_nodes()
        if n <= 1:
            return float('inf')

        rng = np.random.default_rng(seed)
        nodes = list(H.nodes())
        m = min(sample_size, n)
        # Choose unique sample sources without replacement when possible
        sources = list(rng.choice(nodes, size=m, replace=False))

        total_dist = 0.0
        pair_count = 0
        for s in sources:
            lengths = nx.single_source_shortest_path_length(H, s)
            # exclude self distance (0)
            if len(lengths) > 1:
                total_dist += sum(d for v, d in lengths.items() if v != s)
                pair_count += (len(lengths) - 1)

        return (total_dist / pair_count) if pair_count > 0 else float('inf')

    def compute_ws_metrics(self, beta_values, trials=5, seed=123, metrics_n=None, approx=True, sample_size=300):
        """Compute clustering C and path length L across betas, averaged over trials.

        Parameters:
        - beta_values: iterable of rewiring probabilities
        - trials: number of random graphs per beta to average
        - seed: RNG seed for reproducibility
        - metrics_n: if provided, use this node count for metrics graphs
        - approx: use fast path-length approximation when True
        - sample_size: number of sources for approximation
        """
        rng = np.random.default_rng(seed)
        C = []
        L = []
        n_local = metrics_n or self.n
        for beta in beta_values:
            c_vals = []
            l_vals = []
            for _ in range(trials):
                g = nx.watts_strogatz_graph(n_local, self.k, beta, seed=int(rng.integers(0, 1_000_000)))
                c_vals.append(nx.average_clustering(g))
                if approx:
                    l_vals.append(self._approx_average_shortest_path_length_connected(g, sample_size=sample_size, seed=int(rng.integers(0, 1_000_000))))
                else:
                    l_vals.append(self._avg_shortest_path_length_connected(g))
            C.append(float(np.mean(c_vals)))
            L.append(float(np.mean(l_vals)))
        return np.array(C), np.array(L)

    def draw_network(self, ax, graph, title, beta_value, show_metrics=True, node_scale=1.3):
        """
        Draw a single network with proper styling.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axes to draw on
        graph : networkx.Graph
            The graph to draw
        title : str
            Title for the subplot
        beta_value : float
            Beta value for this network
        show_metrics : bool
            Whether to show network metrics
        """
        positions = self.get_circular_positions(node_scale)
        local_edges, long_range_edges = self.classify_edges(graph, positions)
        
        # Clear the axes
        ax.clear()
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add pink highlight for top node and neighbors in Regular network (β=0)
        if beta_value == 0.0:
            top_node = self.n // 4  # Node at the top of the circle (90 degrees)
            self.draw_node_neighborhood_highlight(ax, positions, target_node=top_node)
        
        # Draw curved edges manually
        self.draw_curved_edges(ax, positions, local_edges, 'gray', 1.5, 4.5)
        
        # Draw long-range edges with different colors and curves
        if long_range_edges:
            if beta_value == 0:
                edge_color = 'lightgray'  # No long-range edges in regular lattice
            elif beta_value < 1:
                edge_color = 'turquoise'  # Small-world connections
            else:
                edge_color = 'turquoise'  # Random connections
                
            self.draw_curved_edges(ax, positions, long_range_edges, edge_color, 2.0, 0.5)
        
        # Draw nodes (bigger)
        nx.draw_networkx_nodes(graph, positions, ax=ax,
                             node_color='black', node_size=150)
        
        # Highlight the target node in red if this is the regular network (β=0)
        if beta_value == 0.0:
            top_node = self.n // 4  # Node at the top of the circle (90 degrees)
            nx.draw_networkx_nodes(graph, positions, nodelist=[top_node], ax=ax,
                                 node_color='red', node_size=150)
        
        # Set title (moved down)
        ax.set_title(title, fontsize=20, fontweight='bold', pad=10, y=1.0)
        
        # Add beta value annotation (moved down)
        ax.text(0, -1.8, f'β = {beta_value}', fontsize=18, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add network metrics if requested (repositioned)
        if show_metrics:
            clustering = nx.average_clustering(graph)
            try:
                path_length = nx.average_shortest_path_length(graph)
            except:
                path_length = float('inf')
            
            metrics_text = f'C = {clustering:.3f}\nL = {path_length:.2f}'
            ax.text(1.4, 1.0, metrics_text, fontsize=15, ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
        
        # Set axis limits to make networks bigger
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.2, 1.5)
    
    def create_transition_visualization(self):
        """Create the complete Watts-Strogatz transition visualization."""
        # Create the three networks
        beta_values = [0.0, 0.05, 1.0]
        titles = ['Regular', 'Small-world', 'Erdős-Rényi']
        
        # Create figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Watts-Strogatz Model: Network Transition', 
                    fontsize=24, fontweight='bold', y=1.05)
        
        graphs = []
        for i, (beta, title) in enumerate(zip(beta_values, titles)):
            # Generate graph
            graph = self.create_watts_strogatz_graph(beta, seed=42)
            graphs.append(graph)
            
            # Draw network
            self.draw_network(axes[i], graph, title, beta, show_metrics=True)
        
        # Add annotations for clustering and path length
        fig.text(0.5, 0.02, 'increasing connection probability\nredistribution from short-range to long-range', 
                ha='center', fontsize=18, style='italic')
        
        # Add arrow showing the transition
        from matplotlib.patches import FancyArrowPatch
        arrow_y = 0.12
        arrow = FancyArrowPatch((0.15, arrow_y), (0.85, arrow_y),
                               arrowstyle='->', mutation_scale=20, 
                               color='black', linewidth=2,
                               transform=fig.transFigure)
        fig.patches.append(arrow)
        
        # # Add clustering and path length indicators
        # # Pink arc for clustering (ρS)
        # fig.text(0.17, 0.85, 'ρS = 1', fontsize=12, color='magenta', 
        #         ha='center', weight='bold', transform=fig.transFigure)
        # fig.text(0.5, 0.85, 'ρS < 1', fontsize=12, color='magenta', 
        #         ha='center', weight='bold', transform=fig.transFigure)
        # fig.text(0.83, 0.85, 'ρS = ρL', fontsize=12, color='magenta', 
        #         ha='center', weight='bold', transform=fig.transFigure)
        
        # # Turquoise arc for path length (ρL)
        # fig.text(0.17, 0.15, 'ρL = 0', fontsize=12, color='turquoise', 
        #         ha='center', weight='bold', transform=fig.transFigure)
        # fig.text(0.5, 0.15, 'ρL > 0', fontsize=12, color='turquoise', 
        #         ha='center', weight='bold', transform=fig.transFigure)
        # fig.text(0.83, 0.15, '', fontsize=12, color='turquoise', 
        #         ha='center', weight='bold', transform=fig.transFigure)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, bottom=0.05)
        plt.savefig('watts_strogatz_transition.png', dpi=300, bbox_inches='tight')
        print("✓ Watts-Strogatz transition visualization saved as 'watts_strogatz_transition.png'")
        
        return fig, graphs
    
    def create_detailed_analysis(self, graphs):
        """Create detailed analysis of the three networks."""
        beta_values = [0.0, 0.1, 1.0]
        network_names = ['Regular', 'Small-world', 'Random']
        
        print("\n" + "="*60)
        print("WATTS-STROGATZ MODEL ANALYSIS")
        print("="*60)
        
        print(f"{'Network':<12} {'β':<6} {'Clustering':<12} {'Path Length':<12} {'Edges':<8}")
        print("-" * 60)
        
        for i, (graph, beta, name) in enumerate(zip(graphs, beta_values, network_names)):
            clustering = nx.average_clustering(graph)
            try:
                path_length = nx.average_shortest_path_length(graph)
            except:
                path_length = float('inf')
            edges = graph.number_of_edges()
            
            print(f"{name:<12} {beta:<6} {clustering:<12.4f} {path_length:<12.2f} {edges:<8}")
        
        print("\n" + "="*60)
        print("KEY INSIGHTS")
        print("="*60)
        print("• Regular (β=0): High clustering, long paths - like a social circle")
        print("• Small-world (β=0.1): High clustering, short paths - 'six degrees'")
        print("• Random (β=1): Low clustering, short paths - like Erdős-Rényi")
        print("• Small-world networks combine benefits of both extremes!")
        
        # Calculate small-world coefficient
        regular_clustering = nx.average_clustering(graphs[0])
        sw_clustering = nx.average_clustering(graphs[1])
        random_clustering = nx.average_clustering(graphs[2])
        
        try:
            regular_path = nx.average_shortest_path_length(graphs[0])
            sw_path = nx.average_shortest_path_length(graphs[1])
            random_path = nx.average_shortest_path_length(graphs[2])
            
            # Small-world coefficient: (C/C_random) / (L/L_random)
            if random_clustering > 0 and random_path > 0:
                sw_coefficient = (sw_clustering / random_clustering) / (sw_path / random_path)
                print(f"\n• Small-world coefficient (β=0.1): {sw_coefficient:.2f}")
                if sw_coefficient > 1:
                    print("  → This confirms small-world properties!")
        except:
            pass

    def create_classic_ws_figure(self):
        """Create figure matching the classic WS diagram with normalized curves."""
        # Top row networks
        beta_top = [0.0, 0.05, 1.0]
        titles = ['Regular', 'Small-world', 'Random']
        graphs = [self.create_watts_strogatz_graph(b, seed=42) for b in beta_top]

        # Metrics over log-spaced beta (use larger n and fast path-length)
        beta_vals = np.logspace(-4, 0, 35)
        metrics_n = 300
        C_vals, L_vals = self.compute_ws_metrics(beta_vals, trials=80, seed=123,
                                                metrics_n=metrics_n, approx=True,
                                                sample_size=300)
        # Baseline (beta=0) with the same metrics_n
        G0 = nx.watts_strogatz_graph(metrics_n, self.k, 0.0, seed=42)
        C0 = nx.average_clustering(G0)
        L0 = self._approx_average_shortest_path_length_connected(G0, sample_size=300, seed=123)
        C_norm = C_vals / C0 if C0 > 0 else C_vals
        L_norm = L_vals / L0 if L0 > 0 else L_vals

        # Figure composition
        fig = plt.figure(figsize=(12, 9))
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1.3], hspace=0.35)
        axes_top = [fig.add_subplot(gs[0, i]) for i in range(3)]
        ax_bottom = fig.add_subplot(gs[1, :])

        # Draw networks
        for ax, G, title, beta in zip(axes_top, graphs, titles, beta_top):
            self.draw_network(ax, G, title, beta, show_metrics=False, node_scale=1.25)
            #ax.text(0, -2.70, f'p = {beta}', fontsize=12, ha='center')

        # Arrow and label
        from matplotlib.patches import FancyArrowPatch
        arrow = FancyArrowPatch((0.18, 0.58), (0.82, 0.58), arrowstyle='->', mutation_scale=20,
                                color='black', linewidth=1.8, transform=fig.transFigure)
        fig.patches.append(arrow)
        fig.text(0.5, 0.56, 'increasing randomness', ha='center', fontsize=14)

        # Bottom plot
        ax_bottom.set_xscale('log')
        ax_bottom.plot(beta_vals, L_norm, 'o', color='black', markersize=5, label=r'$L_\beta/L_0$')
        ax_bottom.plot(beta_vals, C_norm, 'o', markerfacecolor='none', markeredgecolor='black',
                       markersize=5, label=r'$C_\beta/C_0$')
        ax_bottom.set_xlabel(r'Rewiring Probability ($\beta$)', fontsize=16)
        ax_bottom.set_ylabel('Clustering, Path Length', fontsize=16)
        ax_bottom.tick_params(axis='both', which='both', labelsize=14)
        ax_bottom.set_ylim(0, max(1.05, float(np.nanmax([np.nanmax(L_norm), np.nanmax(C_norm)]))))
        ax_bottom.grid(True, which='both', linestyle=':', alpha=0.4)
        ax_bottom.legend(loc='upper right', frameon=False, fontsize=14)

        plt.tight_layout()
        plt.savefig('watts_strogatz_classic.png', dpi=300, bbox_inches='tight')
        plt.savefig('watts_strogatz_classic.pdf', dpi=300, bbox_inches='tight')
        print("\u2713 Watts-Strogatz classic figure saved as 'watts_strogatz_classic.png' and '.pdf'")
        return fig

def main():
    """Main function to create Watts-Strogatz visualization."""
    print("Watts-Strogatz Model Visualization")
    print("="*40)
    
    # Initialize visualizer (node count can be tuned for aesthetics)
    visualizer = WattsStrogatzVisualizer(n=24, k=4)
    
    # Create the classic WS diagram (networks + normalized curves)
    visualizer.create_classic_ws_figure()

    # Create the transition visualization panel
    fig, graphs = visualizer.create_transition_visualization()
    
    # Create detailed analysis
    visualizer.create_detailed_analysis(graphs)
    
    print(f"\n" + "="*40)
    print("VISUALIZATION COMPLETE!")
    print("="*40)
    print("Generated file: watts_strogatz_transition.png")
    print("\nThis visualization demonstrates:")
    print("  • The smooth transition from regular to random networks")
    print("  • How small-world networks emerge at intermediate β values")
    print("  • The balance between clustering and path length")
    print("  • Why small-world networks are important for modeling real systems")

if __name__ == "__main__":
    main()
