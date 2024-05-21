import matplotlib.pyplot as plt
import numpy as np

# Function to plot percentage difference between Pareto curve and multiple other curves
def plot_percentage_differences(pareto_curve, other_curves, labels):
    # Ensure each curve in other_curves has the same length as pareto_curve
    for curve in other_curves:
        if len(pareto_curve) != len(curve):
            raise ValueError("All curves must have the same length as the Pareto curve")

    plt.figure(figsize=(10, 6))
    
    # Define line styles and markers to differentiate curves
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', '*', 'x', 'd', '+']
    colors = plt.cm.viridis(np.linspace(0, 1, len(other_curves)))

    # Plot percentage differences for each other curve
    for i, other_curve in enumerate(other_curves):
        # Calculate the percentage difference
        percentage_difference = [(other_curve[j] - pareto_curve[j]) / pareto_curve[j] * 100 for j in range(len(pareto_curve))]
        
        # Select a line style and marker
        line_style = line_styles[i % len(line_styles)]
        marker = markers[i % len(markers)]
        color = colors[i]
        
        # Plot the percentage difference with transparency
        plt.plot(percentage_difference, linestyle=line_style, marker=marker, color=color, alpha=0.7, label=f'Percentage Difference - {labels[i]}')

    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)  # Horizontal line at zero
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid lines
    plt.title("Percentage Difference between Pareto Curve and Other Curves")
    plt.xlabel("Index")
    plt.ylabel("Percentage Difference (%)")
    plt.legend()
    plt.show()