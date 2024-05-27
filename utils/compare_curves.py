import matplotlib.pyplot as plt
import numpy as np

def plot_percentage_differences(pareto_curve, other_curves, labels):
    # Ensure each curve in other_curves has the same length as pareto_curve
    for curve in other_curves:
        if len(pareto_curve) != len(curve):
            raise ValueError("All curves must have the same length as the Pareto curve")

    plt.figure(figsize=(12, 8))
    
    # Define line styles and markers to differentiate curves
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', '*', 'x', 'd', '+']
    colors = plt.cm.viridis(np.linspace(0, 1, len(other_curves)))

    # Plot percentage differences for each other curve
    for i, other_curve in enumerate(other_curves):
        # Calculate the percentage difference
        percentage_difference = [(other_curve[j] - pareto_curve[j]) / pareto_curve[j] * 100 for j in range(len(pareto_curve))]
        
        # Select a line style, marker, and color
        line_style = line_styles[i % len(line_styles)]
        marker = markers[i % len(markers)]
        color = colors[i]
        
        # Create the step plot
        plt.step(range(len(pareto_curve)), percentage_difference, where='post', linestyle=line_style, linewidth=2.5, color=color, alpha=0.7, label=f'Percentage Difference - {labels[i]}')
        
        # Add markers to the steps
        plt.plot(range(len(pareto_curve)), percentage_difference, linestyle='None', marker=marker, color=color)

    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)  # Horizontal line at zero
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid lines
    plt.title("Percentage Difference between Pareto Curve and Other Curves")
    plt.xlabel("Period")
    plt.ylabel("Percentage Difference (%)")
    plt.legend()
    plt.show()
