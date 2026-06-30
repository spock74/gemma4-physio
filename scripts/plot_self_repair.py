import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    results_path = "results/self_repair_sweep.json"
    if not os.path.exists(results_path):
        print(f"File not found: {results_path}")
        return
        
    with open(results_path, "r") as f:
        data = json.load(f)
        
    survival_by_layer = data["survival_by_layer"]
    
    # Extract data for plotting
    layers = sorted([int(k) for k in survival_by_layer.keys()])
    survival_rates = [survival_by_layer[str(layer)] * 100.0 for layer in layers]
    
    # Plot Configuration
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Plot curve
    sns.lineplot(
        x=layers, 
        y=survival_rates, 
        marker='o', 
        markersize=10, 
        linewidth=3, 
        color='#2C3E50',
        ax=ax
    )
    
    # Aesthetics
    ax.set_title('Residual Stream Self-Repair Under 90° SPPS Perturbation', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Intervention Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Text Semantic Survival (%)', fontsize=14, fontweight='bold')
    
    # Add annotations and highlighted regions
    ax.axvspan(10, 26, color='red', alpha=0.1, label='Vulnerability Zone')
    ax.axvspan(0, 8, color='green', alpha=0.1, label='Early Self-Repair Zone')
    ax.axvspan(28, 33, color='blue', alpha=0.1, label='Late Robustness Zone')
    
    # Highlight specific points
    min_survival = min(survival_rates)
    min_layer = layers[survival_rates.index(min_survival)]
    
    ax.annotate(f'Maximum Degradation\n(Layer {min_layer}, {min_survival:.1f}%)', 
                xy=(min_layer, min_survival), 
                xytext=(min_layer - 5, min_survival - 3),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=11)
                
    ax.set_ylim(min(80, min(survival_rates)-5), 105)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', fontsize=11)
    
    plt.tight_layout()
    
    # Save Outputs
    os.makedirs("docs/figures", exist_ok=True)
    os.makedirs("docs/in-tex/figures", exist_ok=True)
    
    plt.savefig("docs/figures/fig9_self_repair.png", format='png', bbox_inches='tight')
    plt.savefig("docs/figures/fig9_self_repair.pdf", format='pdf', bbox_inches='tight')
    
    # Also save to the original figures folder if it exists
    if os.path.exists("figures"):
        plt.savefig("figures/fig9_self_repair.png", format='png', bbox_inches='tight')
        plt.savefig("figures/fig9_self_repair.pdf", format='pdf', bbox_inches='tight')
        
    print("Plotting complete! Saved as docs/figures/fig9_self_repair.png and .pdf")

if __name__ == "__main__":
    main()
