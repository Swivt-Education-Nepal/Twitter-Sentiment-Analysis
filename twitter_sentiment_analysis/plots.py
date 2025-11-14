# twitter_sentiment_analysis/plots.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
import os
warnings.filterwarnings('ignore')

# Set the style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TrainingVisualizer:
    """Class to create visualizations from training metrics"""
    
    def __init__(self, metrics_path="reports/train_val_metrics_5class.csv"):
        self.metrics_path = metrics_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load training metrics from CSV"""
        try:
            self.df = pd.read_csv(self.metrics_path)
            print(f"✅ Loaded metrics data with {len(self.df)} epochs")
            print(f"📊 Columns: {list(self.df.columns)}")
        except FileNotFoundError:
            print(f"❌ Metrics file not found: {self.metrics_path}")
            print("Please run training first to generate metrics CSV")
            return False
        return True
    
    def plot_loss_curves(self, save_path="reports/figures/loss_curves.png"):
        """Plot training and validation loss curves"""
        if self.df is None:
            return
        
        plt.figure(figsize=(12, 6))
        
        plt.plot(self.df['epoch'], self.df['train_loss'], 
                label='Training Loss', linewidth=2.5, alpha=0.8)
        plt.plot(self.df['epoch'], self.df['val_loss'], 
                label='Validation Loss', linewidth=2.5, alpha=0.8)
        
        # Find minimum validation loss point
        min_val_loss_idx = self.df['val_loss'].idxmin()
        min_val_loss_epoch = self.df.loc[min_val_loss_idx, 'epoch']
        min_val_loss = self.df.loc[min_val_loss_idx, 'val_loss']
        
        plt.axvline(x=min_val_loss_epoch, color='red', linestyle='--', alpha=0.7, 
                   label=f'Best Epoch: {min_val_loss_epoch}')
        plt.scatter(min_val_loss_epoch, min_val_loss, color='red', s=100, zorder=5)
        
        plt.xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel('Loss', fontsize=12, fontweight='bold')
        plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        Path("reports/figures").mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Loss curves saved to: {save_path}")
    
    def plot_accuracy_curves(self, save_path="reports/figures/accuracy_curves.png"):
        """Plot training and validation accuracy curves"""
        if self.df is None:
            return
        
        plt.figure(figsize=(12, 6))
        
        plt.plot(self.df['epoch'], self.df['train_acc'], 
                label='Training Accuracy', linewidth=2.5, alpha=0.8)
        plt.plot(self.df['epoch'], self.df['val_acc'], 
                label='Validation Accuracy', linewidth=2.5, alpha=0.8)
        
        # Find maximum validation accuracy point
        max_val_acc_idx = self.df['val_acc'].idxmax()
        max_val_acc_epoch = self.df.loc[max_val_acc_idx, 'epoch']
        max_val_acc = self.df.loc[max_val_acc_idx, 'val_acc']
        
        plt.axvline(x=max_val_acc_epoch, color='green', linestyle='--', alpha=0.7,
                   label=f'Best Epoch: {max_val_acc_epoch}')
        plt.scatter(max_val_acc_epoch, max_val_acc, color='green', s=100, zorder=5)
        
        plt.xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
        plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Set y-axis to percentage format
        plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
        
        # Save the plot
        Path("reports/figures").mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Accuracy curves saved to: {save_path}")
    
    def plot_combined_metrics(self, save_path="reports/figures/combined_metrics.png"):
        """Plot combined loss and accuracy in subplots"""
        if self.df is None:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Loss
        ax1.plot(self.df['epoch'], self.df['train_loss'], 
                label='Training Loss', linewidth=2.5, alpha=0.8)
        ax1.plot(self.df['epoch'], self.df['val_loss'], 
                label='Validation Loss', linewidth=2.5, alpha=0.8)
        ax1.set_xlabel('Epoch', fontweight='bold')
        ax1.set_ylabel('Loss', fontweight='bold')
        ax1.set_title('Loss Curves', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy
        ax2.plot(self.df['epoch'], self.df['train_acc'], 
                label='Training Accuracy', linewidth=2.5, alpha=0.8)
        ax2.plot(self.df['epoch'], self.df['val_acc'], 
                label='Validation Accuracy', linewidth=2.5, alpha=0.8)
        ax2.set_xlabel('Epoch', fontweight='bold')
        ax2.set_ylabel('Accuracy', fontweight='bold')
        ax2.set_title('Accuracy Curves', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Set y-axis to percentage format for accuracy
        ax2.set_yticklabels(['{:.0f}%'.format(x*100) for x in ax2.get_yticks()])
        
        plt.tight_layout()
        
        # Save the plot
        Path("reports/figures").mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Combined metrics saved to: {save_path}")
    
    def plot_epoch_times(self, save_path="reports/figures/epoch_times.png"):
        """Plot epoch training times"""
        if self.df is None:
            return
        
        plt.figure(figsize=(10, 6))
        
        plt.bar(self.df['epoch'], self.df['epoch_time_sec'], 
               alpha=0.7, color='skyblue', edgecolor='navy')
        
        avg_time = self.df['epoch_time_sec'].mean()
        plt.axhline(y=avg_time, color='red', linestyle='--', 
                   label=f'Average: {avg_time:.1f}s')
        
        plt.xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel('Time (seconds)', fontsize=12, fontweight='bold')
        plt.title('Epoch Training Times', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        Path("reports/figures").mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Epoch times saved to: {save_path}")
        print(f"⏱️  Average epoch time: {avg_time:.2f} seconds")
    
    def plot_training_summary(self, save_path="reports/figures/training_summary.png"):
        """Create a comprehensive training summary plot"""
        if self.df is None:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Loss curves
        ax1.plot(self.df['epoch'], self.df['train_loss'], label='Training Loss', linewidth=2)
        ax1.plot(self.df['epoch'], self.df['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_title('Loss Curves', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy curves
        ax2.plot(self.df['epoch'], self.df['train_acc'], label='Training Accuracy', linewidth=2)
        ax2.plot(self.df['epoch'], self.df['val_acc'], label='Validation Accuracy', linewidth=2)
        ax2.set_title('Accuracy Curves', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yticklabels(['{:.0f}%'.format(x*100) for x in ax2.get_yticks()])
        
        # Plot 3: Epoch times
        ax3.bar(self.df['epoch'], self.df['epoch_time_sec'], alpha=0.7, color='lightgreen')
        ax3.set_title('Epoch Training Times', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Time (seconds)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Final metrics comparison
        final_train_acc = self.df['train_acc'].iloc[-1]
        final_val_acc = self.df['val_acc'].iloc[-1]
        final_train_loss = self.df['train_loss'].iloc[-1]
        final_val_loss = self.df['val_loss'].iloc[-1]
        
        metrics = ['Train Acc', 'Val Acc', 'Train Loss', 'Val Loss']
        values = [final_train_acc, final_val_acc, final_train_loss, final_val_loss]
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightsalmon']
        
        bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
        ax4.set_title('Final Epoch Metrics', fontweight='bold')
        ax4.set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            if 'Acc' in bar.get_label():
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            else:
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save the plot
        Path("reports/figures").mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Training summary saved to: {save_path}")
    
    def print_training_summary(self):
        """Print a textual summary of training results"""
        if self.df is None:
            return
        
        best_val_loss_idx = self.df['val_loss'].idxmin()
        best_val_acc_idx = self.df['val_acc'].idxmax()
        
        print("\n" + "="*60)
        print("🏆 TRAINING SUMMARY")
        print("="*60)
        
        print(f"📈 Total Epochs Trained: {len(self.df)}")
        print(f"⏱️  Total Training Time: {self.df['epoch_time_sec'].sum():.1f} seconds")
        print(f"🕒 Average Epoch Time: {self.df['epoch_time_sec'].mean():.1f} seconds")
        
        print("\n🎯 BEST VALIDATION PERFORMANCE:")
        print(f"   • Best Validation Loss: {self.df.loc[best_val_loss_idx, 'val_loss']:.4f} "
              f"(Epoch {self.df.loc[best_val_loss_idx, 'epoch']})")
        print(f"   • Best Validation Accuracy: {self.df.loc[best_val_acc_idx, 'val_acc']:.4f} "
              f"(Epoch {self.df.loc[best_val_acc_idx, 'epoch']})")
        
        print("\n📊 FINAL EPOCH PERFORMANCE:")
        print(f"   • Final Training Loss: {self.df['train_loss'].iloc[-1]:.4f}")
        print(f"   • Final Validation Loss: {self.df['val_loss'].iloc[-1]:.4f}")
        print(f"   • Final Training Accuracy: {self.df['train_acc'].iloc[-1]:.4f} "
              f"({self.df['train_acc'].iloc[-1]*100:.2f}%)")
        print(f"   • Final Validation Accuracy: {self.df['val_acc'].iloc[-1]:.4f} "
              f"({self.df['val_acc'].iloc[-1]*100:.2f}%)")
        
        print("\n📉 OVERFITTING ANALYSIS:")
        train_val_gap = self.df['train_acc'].iloc[-1] - self.df['val_acc'].iloc[-1]
        if train_val_gap > 0.1:
            status = "⚠️  HIGH (Possible overfitting)"
        elif train_val_gap > 0.05:
            status = "ℹ️   MODERATE"
        else:
            status = "✅ LOW"
        print(f"   • Train-Val Accuracy Gap: {train_val_gap:.4f} - {status}")
    
    def create_all_plots(self):
        """Create all available plots"""
        if self.df is None:
            print("❌ No data loaded. Cannot create plots.")
            return
        
        print("🎨 Creating training visualizations...")
        
        self.plot_loss_curves()
        self.plot_accuracy_curves()
        self.plot_combined_metrics()
        self.plot_epoch_times()
        self.plot_training_summary()
        self.print_training_summary()
        
        print("\n✅ All plots created and saved to 'reports/figures/'")

# Convenience functions
def quick_plot(metrics_path="reports/train_val_metrics_5class.csv"):
    """Quick function to create all plots"""
    visualizer = TrainingVisualizer(metrics_path)
    visualizer.create_all_plots()

def plot_from_csv(csv_path):
    """Plot from a specific CSV file path"""
    visualizer = TrainingVisualizer(csv_path)
    visualizer.create_all_plots()

if __name__ == "__main__":
    # When run directly, create all plots
    quick_plot()