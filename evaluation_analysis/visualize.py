import json
import os
import matplotlib.pyplot as plt

def plot_evaluation_results(report_path="outputs/evaluation/baseline_comparison.json"):
    """
    Plots the final results: Swift (Multiple Exits) vs Baselines.
    """
    if not os.path.exists(report_path):
        print(f"Report not found at {report_path}. Run baseline_evaluation.py first.")
        return

    with open(report_path, 'r') as f:
        data = json.load(f)

    swift_results = data.get('swift_results', [])
    baselines = data.get('baselines', {})

    # Extract Swift RD-C points
    swift_psnr = [r['avg_psnr'] for r in swift_results]
    swift_bpp = [r['avg_bpp'] for r in swift_results]
    labels = [f"L{r['config']['quality_level']} {r['config']['exit_at']}" for r in swift_results]

    plt.figure(figsize=(10, 6))

    # 1. Plot Swift Curve
    plt.plot(swift_bpp, swift_psnr, 'o-', label='Swift (Neural Layered)', markersize=8, linewidth=2)
    for i, txt in enumerate(labels):
        plt.annotate(txt, (swift_bpp[i], swift_psnr[i]), xytext=(5, 5), textcoords='offset points')

    # 2. Plot Baselines as horizontal/points
    colors = ['r', 'g', 'm']
    for i, (name, metrics) in enumerate(baselines.items()):
        plt.axhline(y=metrics['psnr'], color=colors[i % len(colors)], linestyle='--', alpha=0.6, label=name)
        # Also plot a point if bpp is known
        if 'bpp' in metrics:
            plt.plot(metrics['bpp'], metrics['psnr'], 's', color=colors[i % len(colors)])

    plt.title('Rate-Distortion Comparison (NSDI \'22 Swift)')
    plt.xlabel('Bitrate (bpp)')
    plt.ylabel('Quality (PSNR dB)')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()

    # Save Graph
    output_img = "outputs/evaluation/rd_curve_comparison.png"
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    plt.savefig(output_img, dpi=300)
    print(f"Visualization saved to {output_img}")
    plt.show()

if __name__ == "__main__":
    plot_evaluation_results()
