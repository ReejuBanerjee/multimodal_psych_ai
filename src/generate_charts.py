# Author: Reeju Banerjee & Shourya
# Registration Number: RA2511003010548
# Project: Explainable Multimodal AI Framework - Presentation Assets

import matplotlib.pyplot as plt

def generate_presentation_charts():
    plt.style.use('dark_background')
    accent = '#deff9a'
    
    # --- CHART 1: CLASSIFICATION METRICS ---
    # UPDATE THESE NUMBERS with your actual evaluate.py output!
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [96.50, 96.72, 95.81, 96.25] 
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=accent, alpha=0.9)
    plt.ylim(85, 100)
    plt.title('Final Classification Performance', fontsize=16, pad=20)
    plt.ylabel('Score (%)')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                 f'{height}%', ha='center', va='bottom', fontweight='bold', color=accent)
                 
    plt.savefig('../classification_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- CHART 2: REGRESSION R2 SCORE ---
    # UPDATE THIS NUMBER with your actual R2 score
    r2_score = 98.60 
    
    plt.figure(figsize=(8, 8))
    plt.pie([r2_score, 100 - r2_score], labels=['Explained Variance', 'Error Margin'], 
            autopct='%1.2f%%', colors=[accent, '#333333'], startangle=140,
            textprops={'color': 'white', 'fontsize': 14, 'fontweight': 'bold'})
    plt.title(f'Continuous Severity Precision ($R^2$)', fontsize=18)
    
    plt.savefig('../regression_chart.png', dpi=300, bbox_inches='tight')
    print("[SUCCESS] Charts generated and saved to the project root folder!")

if __name__ == "__main__":
    generate_presentation_charts()