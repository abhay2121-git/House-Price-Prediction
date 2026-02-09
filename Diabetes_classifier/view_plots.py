"""
Quick plot viewer - Display all generated plots interactively
"""

import matplotlib.pyplot as plt
import os
from PIL import Image

def view_all_plots():
    """
    Display all generated plots in a simple viewer
    """
    figures_dir = 'outputs/figures'
    
    if not os.path.exists(figures_dir):
        print(f"Error: {figures_dir} directory not found!")
        return
    
    plot_files = [f for f in os.listdir(figures_dir) if f.endswith('.png')]
    
    if not plot_files:
        print("No plot files found!")
        return
    
    print(f"Found {len(plot_files)} plots:")
    for i, file in enumerate(plot_files, 1):
        print(f"  {i}. {file}")
    
    print("\nOpening plots...")
    
    for plot_file in plot_files:
        filepath = os.path.join(figures_dir, plot_file)
        try:
            # Open and display the image
            img = Image.open(filepath)
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.title(plot_file.replace('.png', '').replace('_', ' ').title())
            plt.axis('off')
            plt.show()
            print(f"Displayed: {plot_file}")
        except Exception as e:
            print(f"Error displaying {plot_file}: {e}")

if __name__ == "__main__":
    view_all_plots()
