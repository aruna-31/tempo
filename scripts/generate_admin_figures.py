"""
Generate publication-quality Figure 6 for TEMPO Research Paper:
Admin Intelligence Panel: Architecture, Pedagogical Proxemics, and Longitudinal 3-Class Telemetry.
"""
import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("storage/research/journal_figures", exist_ok=True)
output_path = "storage/research/journal_figures/fig6_admin_panel_architecture_and_telemetry.png"

fig = plt.figure(figsize=(14, 10), dpi=300)
fig.patch.set_facecolor('#ffffff')

# Create a 2x2 grid of subplots
gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

# --- Subplot 1: Pedagogical Proxemics Zone Breakdown across 3 Observed Classes ---
ax1 = fig.add_subplot(gs[0, 0])
classes = ['Class 1 (Theory)', 'Class 2 (Problem Solving)', 'Class 3 (Group Review)']
lectern = [47.7, 28.1, 22.1]
aisle = [22.4, 35.4, 43.7]
board = [20.1, 28.1, 24.9]
desk = [9.8, 8.4, 9.3]

x = np.arange(len(classes))
width = 0.55

p1 = ax1.bar(x, lectern, width, label='Didactic Lectern (Podium)', color='#2B6CB0')
p2 = ax1.bar(x, board, width, bottom=lectern, label='Board Exposition (Slides)', color='#4299E1')
p3 = ax1.bar(x, aisle, width, bottom=np.array(lectern)+np.array(board), label='Aisle Circulation (Walking)', color='#48BB78')
p4 = ax1.bar(x, desk, width, bottom=np.array(lectern)+np.array(board)+np.array(aisle), label='Desk Consultation (<1m)', color='#ECC94B')

ax1.set_ylabel('Pedagogical Proxemics Distribution (%)', fontsize=10, fontweight='bold', color='#1A365D')
ax1.set_title('(A) Longitudinal Spatial Proxemics across 3 Classes', fontsize=11, fontweight='bold', color='#1A365D', pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(classes, fontsize=9, fontweight='bold')
ax1.set_ylim(0, 105)
ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)
ax1.grid(axis='y', linestyle='--', alpha=0.4)

# Value annotations
for i in range(len(classes)):
    ax1.text(i, lectern[i]/2, f"{lectern[i]}%", ha='center', va='center', color='white', fontweight='bold', fontsize=8)
    ax1.text(i, lectern[i] + board[i]/2, f"{board[i]}%", ha='center', va='center', color='white', fontweight='bold', fontsize=8)
    ax1.text(i, lectern[i] + board[i] + aisle[i]/2, f"{aisle[i]}%", ha='center', va='center', color='white', fontweight='bold', fontsize=8)

# --- Subplot 2: Mobility Index vs. Student Focus Correlation ---
ax2 = fig.add_subplot(gs[0, 1])
class_nums = [1, 2, 3]
mobility = [56.0, 72.9, 78.9]
student_focus = [79.8, 84.0, 87.8]
entropy = [0.542, 0.638, 0.724]

ax2_twin = ax2.twinx()

line1 = ax2.plot(class_nums, mobility, 'o-', color='#319795', linewidth=2.5, markersize=8, label='Faculty Mobility Index')
line2 = ax2.plot(class_nums, student_focus, 's--', color='#D69E2E', linewidth=2.5, markersize=8, label='Student Focus (S_att %)')
line3 = ax2_twin.plot(class_nums, entropy, '^-.', color='#9F7AEA', linewidth=2.2, markersize=8, label='Shannon Entropy (H)')

ax2.set_xlabel('Class Observation Window (Sequential Classes)', fontsize=10, fontweight='bold', color='#1A365D')
ax2.set_ylabel('Score / Percentage (%)', fontsize=10, fontweight='bold', color='#1A365D')
ax2_twin.set_ylabel('Normalized Shannon Entropy (H)', fontsize=10, fontweight='bold', color='#6B46C1')
ax2.set_title('(B) Mobility, Focus & Behavioral Entropy Co-Evolution', fontsize=11, fontweight='bold', color='#1A365D', pad=10)
ax2.set_xticks(class_nums)
ax2.set_xticklabels(['Class 1\n(Theory)', 'Class 2\n(Problems)', 'Class 3\n(Review)'], fontsize=9)
ax2.set_ylim(50, 95)
ax2_twin.set_ylim(0.4, 0.85)

lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, loc='lower right', fontsize=8, framealpha=0.9)
ax2.grid(True, linestyle='--', alpha=0.4)

# --- Subplot 3: Timetable Topological Ingestion Pipeline Matrix ---
ax3 = fig.add_subplot(gs[1, 0])
stages = ['64-Page aSc PDF\nIngestion', 'pdfplumber BBox\nParsing', '3-Tier Token\nExtraction', 'PostgreSQL\nRelational Link']
counts = [64, 741, 133, 50]  # PDF Pages, Slots, Subjects, Sections
labels_sub = ['Pages Ingested', 'Schedule Slots', 'Subjects Mapped', 'Sections Linked']
colors = ['#2C5282', '#3182CE', '#63B3ED', '#90CDF4']

y_pos = np.arange(len(stages))
bars = ax3.barh(y_pos, counts, color=colors, height=0.55, edgecolor='#1A365D', linewidth=1)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(stages, fontsize=9, fontweight='bold')
ax3.set_xlabel('Count of Ingested Academic Entities', fontsize=10, fontweight='bold', color='#1A365D')
ax3.set_title('(C) Topological Timetable Ingestion Efficiency', fontsize=11, fontweight='bold', color='#1A365D', pad=10)
ax3.grid(axis='x', linestyle='--', alpha=0.4)

for bar, cnt, lbl in zip(bars, counts, labels_sub):
    ax3.text(bar.get_width() + 15, bar.get_y() + bar.get_height()/2, f"{cnt} ({lbl})", 
             va='center', ha='left', fontsize=8.5, fontweight='bold', color='#2D3748')
ax3.set_xlim(0, 920)

# --- Subplot 4: FIAC Interaction Category Distribution & Delivery Rate ---
ax4 = fig.add_subplot(gs[1, 1])
fiac_categories = [
    'Category 5\n(Direct Lecturing)',
    'Categories 4 & 8\n(Question / Response)',
    'Categories 3 & 9\n(Student Facilitation)'
]
fiac_shares = [28.6, 38.2, 33.2]
fiac_colors = ['#E53E3E', '#DD6B20', '#38A169']

wedges, texts, autotexts = ax4.pie(
    fiac_shares, labels=fiac_categories, autopct='%1.1f%%',
    startangle=140, colors=fiac_colors,
    textprops={'fontsize': 9, 'fontweight': 'bold'},
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
for at in autotexts:
    at.set_color('white')
    at.set_fontsize(9.5)
ax4.set_title('(D) Flanders Interaction (FIAC) Synthesis across 3 Classes\nActive Teaching Delivery Rate = 97.4% (Idling: 2.6%, Phone: 0.0%)', 
              fontsize=10, fontweight='bold', color='#1A365D', pad=10)

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"SUCCESS: Generated Figure 6 at {output_path}")
