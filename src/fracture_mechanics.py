"""
3-Point Bending Fracture Mechanics Analysis
Calculate K1 (Stress Intensity Factor) and visualize specimen geometry
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.lines import Line2D
import json

class ThreePointBendingAnalysis:
    """Analyze 3-point bending specimen with crack using fracture mechanics"""
    
    # Defect classification thresholds
    NO_DEFECT_THRESHOLD = 5  # mm - No defect if crack < 5mm
    MINOR_DEFECT_THRESHOLD = 15  # mm - Minor defect 5-15mm
    MODERATE_DEFECT_THRESHOLD = 30  # mm - Moderate defect 15-30mm
    SEVERE_DEFECT_THRESHOLD = 50  # mm - Severe defect 30-50mm
    CRITICAL_DEFECT_THRESHOLD = 50  # mm - Critical if > 50mm
    
    def __init__(self, width=100, thickness=10, span=400, crack_depth=30, load=1000):
        """
        Initialize specimen parameters.
        
        Args:
            width: Specimen width W (mm)
            thickness: Specimen thickness B (mm)
            span: Support span L (mm)
            crack_depth: Crack depth a (mm)
            load: Applied load P (N)
        """
        self.W = width  # Width
        self.B = thickness  # Thickness
        self.L = span  # Span (distance between supports)
        self.a = crack_depth  # Crack depth
        self.P = load  # Applied load
    
    def classify_defect(self):
        """
        Classify defect severity based on crack depth.
        
        Returns:
            dict: Classification results with status, risk level, and description
        """
        if self.a < self.NO_DEFECT_THRESHOLD:
            return {
                "has_defect": False,
                "classification": "NO DEFECT",
                "severity_level": 0,
                "risk": "SAFE",
                "color": "green",
                "description": "No crack detected - Specimen is healthy"
            }
        elif self.a < self.MINOR_DEFECT_THRESHOLD:
            return {
                "has_defect": True,
                "classification": "MINOR DEFECT",
                "severity_level": 1,
                "risk": "LOW",
                "color": "yellow",
                "description": "Small crack detected - Monitor closely"
            }
        elif self.a < self.MODERATE_DEFECT_THRESHOLD:
            return {
                "has_defect": True,
                "classification": "MODERATE DEFECT",
                "severity_level": 2,
                "risk": "MEDIUM",
                "color": "orange",
                "description": "Significant crack - Inspection recommended"
            }
        elif self.a < self.SEVERE_DEFECT_THRESHOLD:
            return {
                "has_defect": True,
                "classification": "SEVERE DEFECT",
                "severity_level": 3,
                "risk": "HIGH",
                "color": "red",
                "description": "Large crack - Immediate action required"
            }
        else:
            return {
                "has_defect": True,
                "classification": "CRITICAL DEFECT",
                "severity_level": 4,
                "risk": "CRITICAL",
                "color": "darkred",
                "description": "Critical crack - Specimen failure imminent"
            }
        
    def calculate_K1_3pb(self):
        """
        Calculate K1 for 3-point bending using ASTM E1820 formula.
        
        K1 = (P*L)/(B*W^1.5) * f(a/W)
        
        where f(a/W) is the geometric function:
        f(a/W) = 1.93*(a/W)^0.5 - 3.07*(a/W) + 14.27*(a/W)^2 - 25.11*(a/W)^3 + 25.80*(a/W)^4
        
        Returns:
            K1: Stress intensity factor (MPa√m)
        """
        # If no defect, K1 is 0 (no crack tip stress concentration)
        defect_info = self.classify_defect()
        if not defect_info['has_defect']:
            return 0.0, 0.0, 0.0
        
        alpha = self.a / self.W  # Relative crack depth
        
        # Geometric function for 3PB (ASTM E1820)
        f_alpha = (1.93 * np.sqrt(alpha) 
                   - 3.07 * alpha 
                   + 14.27 * alpha**2 
                   - 25.11 * alpha**3 
                   + 25.80 * alpha**4)
        
        # Convert load from N to kN for calculation
        P_kN = self.P / 1000
        
        # K1 calculation
        # K1 = (P*L)/(B*W^1.5) * f(a/W)
        # Result in MPa√m
        K1 = (P_kN * self.L) / (self.B * self.W**1.5) * f_alpha
        
        return K1, alpha, f_alpha
    
    def calculate_max_stress(self):
        """Calculate maximum bending stress at crack location."""
        # σ_max = 3*P*L / (2*B*W^2)
        sigma_max = (3 * self.P * self.L) / (2 * self.B * self.W**2)
        return sigma_max
    
    def visualize(self, save_path=None):
        """Create detailed visualization of 3-point bending specimen with crack."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Get defect classification
        defect_info = self.classify_defect()
        
        # ===== LEFT PLOT: Full Setup =====
        ax1.set_xlim(-50, self.L + 50)
        ax1.set_ylim(-30, 100)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('Distance (mm)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Height (mm)', fontsize=11, fontweight='bold')
        ax1.set_title('3-Point Bending Test Setup', fontsize=13, fontweight='bold')
        
        # Draw specimen (uncracked outline)
        specimen_x = self.L / 2
        specimen = Rectangle((specimen_x - self.W/2, self.W/2), self.W, self.B, 
                             fill=False, edgecolor='black', linewidth=2)
        ax1.add_patch(specimen)
        
        # Draw crack only if defect exists
        if defect_info['has_defect']:
            crack_x = specimen_x
            crack_y_top = self.W/2 + self.B
            crack_y_bottom = self.W/2 + self.B - self.a
            ax1.plot([crack_x, crack_x], [crack_y_bottom, crack_y_top], 
                    color=defect_info['color'], linewidth=4, label=f'Crack: {self.a}mm')
        
        # Draw support points
        support_left = 0
        support_right = self.L
        ax1.plot(support_left, 0, 'g^', markersize=15, label='Support (Left)')
        ax1.plot(support_right, 0, 'g^', markersize=15, label='Support (Right)')
        
        # Draw load point
        ax1.arrow(specimen_x, 80, 0, -10, head_width=10, head_length=5, fc='blue', ec='blue')
        ax1.text(specimen_x + 15, 75, f'P = {self.P}N', fontsize=10, fontweight='bold', color='blue')
        
        # Draw supports
        support_width = 15
        ax1.plot([support_left - support_width, support_left + support_width], [-5, -5], 'g-', linewidth=4)
        ax1.plot([support_right - support_width, support_right + support_width], [-5, -5], 'g-', linewidth=4)
        
        # Dimensions
        ax1.annotate('', xy=(0, -15), xytext=(self.L, -15),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
        ax1.text(self.L/2, -22, f'L = {self.L}mm', ha='center', fontsize=10, fontweight='bold')
        
        # Add specimen dimension labels
        ax1.text(specimen_x - self.W/2 - 15, self.W/2 + self.B/2, f'B={self.B}mm', 
                fontsize=9, rotation=90, va='center')
        ax1.text(specimen_x, self.W/2 - 10, f'W={self.W}mm', fontsize=9, ha='center')
        
        # Add defect status box
        status_text = f"STATUS: {defect_info['classification']}\nRISK: {defect_info['risk']}"
        ax1.text(0.02, 0.98, status_text, transform=ax1.transAxes,
                fontsize=11, verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=defect_info['color'], alpha=0.7, pad=0.8),
                color='white' if defect_info['color'] in ['red', 'darkred'] else 'black')
        
        ax1.legend(loc='upper left', fontsize=10)
        
        # ===== RIGHT PLOT: Crack Detail & K1 Analysis =====
        ax2.set_xlim(-10, 60)
        ax2.set_ylim(-10, 60)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('Width (mm)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Height (mm)', fontsize=11, fontweight='bold')
        ax2.set_title('Crack Detail & K1 Analysis', fontsize=13, fontweight='bold')
        
        # Draw specimen cross-section
        specimen_detail = Rectangle((0, 0), self.W, self.B, 
                                   fill=True, facecolor='lightblue', 
                                   edgecolor='black', linewidth=2, alpha=0.5)
        ax2.add_patch(specimen_detail)
        
        # Draw crack only if defect exists
        if defect_info['has_defect']:
            crack_x_detail = self.W / 2
            ax2.plot([crack_x_detail, crack_x_detail], [self.B, self.B - self.a], 
                    color=defect_info['color'], linewidth=4, label='Crack')
            ax2.plot([crack_x_detail - 1, crack_x_detail + 1], [self.B, self.B], 
                    color=defect_info['color'], linewidth=2)  # Crack tip
            
            # Dimension lines
            ax2.plot([self.W/2, self.W/2], [self.B + 3, self.B - self.a - 3], 'k--', alpha=0.5)
            ax2.annotate('', xy=(self.W/2 + 8, self.B), xytext=(self.W/2 + 8, self.B - self.a),
                        arrowprops=dict(arrowstyle='<->', color=defect_info['color'], lw=2))
            ax2.text(self.W/2 + 15, self.B - self.a/2, f'a = {self.a}mm', 
                    fontsize=11, fontweight='bold', color=defect_info['color'])
        else:
            # No crack - show healthy specimen
            ax2.text(self.W/2, self.B/2, '✓ HEALTHY\nNO CRACK', 
                    fontsize=14, fontweight='bold', color='green',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7, pad=1))
        
        # Width dimension
        ax2.annotate('', xy=(0, -3), xytext=(self.W, -3),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
        ax2.text(self.W/2, -6, f'W = {self.W}mm', ha='center', fontsize=10, fontweight='bold')
        
        # Height dimension
        ax2.annotate('', xy=(-3, 0), xytext=(-3, self.B),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
        ax2.text(-6, self.B/2, f'B = {self.B}mm', ha='right', va='center', fontsize=10, fontweight='bold')
        
        # Calculate and display K1
        K1, alpha, f_alpha = self.calculate_K1_3pb()
        sigma_max = self.calculate_max_stress()
        
        # Add K1 calculation info
        info_text = (f"K1 Calculation Results:\n"
                    f"━━━━━━━━━━━━━━━━━━━━━\n"
                    f"Relative crack depth: α = a/W = {alpha:.3f}\n"
                    f"Geometric function: f(α) = {f_alpha:.4f}\n"
                    f"Stress intensity factor:\n"
                    f"K₁ = {K1:.2f} MPa√m\n"
                    f"━━━━━━━━━━━━━━━━━━━━━\n"
                    f"Max bending stress: σ = {sigma_max:.2f} MPa\n"
                    f"Defect: {defect_info['classification']}")
        
        ax2.text(0.5, 0.5, info_text, transform=ax2.transAxes,
                fontsize=11, verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=1),
                family='monospace', fontweight='bold')
        
        ax2.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")
        
        plt.show()
        
        return K1, sigma_max
    
    def generate_report(self):
        """Generate comprehensive analysis report."""
        K1, alpha, f_alpha = self.calculate_K1_3pb()
        sigma_max = self.calculate_max_stress()
        defect_info = self.classify_defect()
        
        report = {
            "title": "3-Point Bending Fracture Mechanics Analysis",
            "specimen_geometry": {
                "width_W_mm": self.W,
                "thickness_B_mm": self.B,
                "span_L_mm": self.L,
                "crack_depth_a_mm": self.a,
            },
            "loading": {
                "applied_load_N": self.P,
            },
            "defect_classification": {
                "has_defect": defect_info['has_defect'],
                "classification": defect_info['classification'],
                "severity_level": defect_info['severity_level'],
                "risk_level": defect_info['risk'],
                "description": defect_info['description'],
            },
            "normalized_parameters": {
                "relative_crack_depth_alpha": alpha,
                "geometric_function_f_alpha": f_alpha,
            },
            "results": {
                "stress_intensity_factor_K1_MPa_sqrt_m": K1,
                "maximum_bending_stress_sigma_MPa": sigma_max,
                "fracture_toughness_required_MPa_sqrt_m": K1,
            },
            "formula": {
                "K1_formula": "K1 = (P*L)/(B*W^1.5) * f(a/W)",
                "f_alpha_formula": "f(α) = 1.93*√α - 3.07*α + 14.27*α² - 25.11*α³ + 25.80*α⁴",
                "sigma_max_formula": "σ_max = 3*P*L / (2*B*W²)",
            },
            "interpretation": {
                "K1_meaning": "Stress Intensity Factor for Mode I (opening) crack",
                "higher_K1_means": "Higher stress concentration at crack tip",
                "fracture_occurs_when": "K1 > K1C (material fracture toughness)",
            }
        }
        
        return report


def main():
    """Run analysis with varying crack sizes on each run."""
    import os
    
    print("\n" + "="*60)
    print("3-POINT BENDING FRACTURE MECHANICS ANALYSIS")
    print("="*60)
    
    # Generate varying crack size (cycles through different sizes)
    counter_file = "crack_size_counter.txt"
    
    if os.path.exists(counter_file):
        with open(counter_file, 'r') as f:
            run_count = int(f.read().strip())
    else:
        run_count = 0
    
    # Define sample crack sizes for different scenarios
    crack_sizes = [
        2,    # No defect
        8,    # Minor defect
        20,   # Moderate defect
        40,   # Severe defect
        60,   # Critical defect
    ]
    
    # Cycle through crack sizes
    crack_depth = crack_sizes[run_count % len(crack_sizes)]
    
    # Update counter
    with open(counter_file, 'w') as f:
        f.write(str((run_count + 1) % len(crack_sizes)))
    
    print(f"\n📊 Run #{run_count + 1} - Simulating Specimen with Crack Depth: {crack_depth}mm")
    
    # Create analysis instance with variable crack depth
    analyzer = ThreePointBendingAnalysis(
        width=100,        # W = 100 mm (specimen width)
        thickness=10,     # B = 10 mm (specimen thickness)
        span=400,         # L = 400 mm (support span)
        crack_depth=crack_depth,  # Variable crack depth
        load=1000         # P = 1000 N (applied load)
    )
    
    # Get defect classification
    defect_info = analyzer.classify_defect()
    
    print(f"\nSpecimen Geometry:")
    print(f"  Width (W):        {analyzer.W} mm")
    print(f"  Thickness (B):    {analyzer.B} mm")
    print(f"  Span (L):         {analyzer.L} mm")
    print(f"  Crack depth (a):  {analyzer.a} mm")
    print(f"\nLoading Condition:")
    print(f"  Applied Load (P): {analyzer.P} N")
    
    # Calculate K1
    K1, alpha, f_alpha = analyzer.calculate_K1_3pb()
    sigma_max = analyzer.calculate_max_stress()
    
    print(f"\nCalculation Parameters:")
    print(f"  Relative crack depth: α = a/W = {alpha:.3f}")
    print(f"  Geometric function:   f(α) = {f_alpha:.4f}")
    
    print(f"\n{'─'*60}")
    print(f"STRESS INTENSITY FACTOR (K1): {K1:.2f} MPa√m")
    print(f"{'─'*60}")
    print(f"Maximum bending stress:     {sigma_max:.2f} MPa")
    
    # Display defect classification
    print(f"\n{'═'*60}")
    print(f"DEFECT DETECTION REPORT")
    print(f"{'═'*60}")
    print(f"Status:        {defect_info['classification']}")
    print(f"Risk Level:    {defect_info['risk']}")
    print(f"Has Defect:    {'YES ⚠️' if defect_info['has_defect'] else 'NO ✓'}")
    print(f"Description:   {defect_info['description']}")
    print(f"{'═'*60}")
    
    # Visualization
    print(f"\nGenerating visualization...")
    K1_viz, sigma_viz = analyzer.visualize(save_path="3pb_analysis.png")
    
    # Generate report
    report = analyzer.generate_report()
    
    # Save report
    with open("3pb_fracture_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✓ Report saved to 3pb_fracture_report.json")
    
    print("\n" + "="*60)
    print("Next run will test a different crack size automatically!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
