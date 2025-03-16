import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Define project tasks, start weeks, durations, and assignees
tasks = {
    "Data Processing": [
        ("Setup ScanNet++ dataset", 1, 3, "Patrick, Fangchen"),
    ],
    "Gaussian Splatting Implementation": [
        ("Implement base GS model", 2, 4, "Patrick"),
        ("Optimize GS for real-time rendering", 3, 5, "Patrick"),
        ("Test GS performance", 4, 6, "Patrick")
    ],
    "NeRF Refinement Model": [
        ("Train NeRF refinement model", 2, 4, "Fangchen"),
        ("Optimize NeRF inference speed", 4, 6, "Fangchen")
    ],
    "Evaluation & Presentation": [
        ("Implement evaluation metrics", 5, 6, "Patrick"),
        ("Run baseline comparisons", 5, 6, "Patrick"),
        ("Fast and high-fidelity merged rendering", 5, 6, "Fangchen, Patrick"),
        ("Prepare poster & recorded presentation", 5, 7, "Patrick, Fangchen"),
        ("Write and submit project report", 6, 7, "Patrick, Fangchen")
    ]
}

# Define colors for different task categories
colors = {
    "Data Processing": "deepskyblue",
    "Gaussian Splatting Implementation": "royalblue",
    "NeRF Refinement Model": "purple",
    "Evaluation & Presentation": "goldenrod"
}

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each task on the timeline
y_pos = 0
task_patches = []
for category, task_list in tasks.items():
    for task, start, end, assignee in task_list:
        ax.barh(y_pos, end - start, left=start, color=colors[category], height=0.6, align="center")
        ax.text(start + (end - start) / 2, y_pos, f"{task}\n({assignee})", 
                ha='center', va='center', fontsize=8, color='black', bbox=dict(facecolor='black', alpha=0))
        y_pos += 1
    y_pos += 1  # Add spacing between categories

# Set labels and grid
ax.set_yticks([])  # Hide y-axis labels (tasks shown inside bars)
ax.set_xticks(range(1, 8))
ax.set_xticklabels(["Week 1", "Week 2", "Week 3", "Week 4", "Week 5", "Week 6", "Week 7"])
ax.set_xlabel("Project Timeline")
ax.set_title("Project Gantt Chart for NeRF-GS Implementation with Assignees")
ax.grid(axis='x', linestyle='--', alpha=0.6)

# Add legend
legend_patches = [mpatches.Patch(color=colors[cat], label=cat) for cat in colors]
ax.legend(handles=legend_patches, loc="upper left")

# Save the updated Gantt chart
gantt_chart_assigned_path = "gantt_chart_assigned.png"
plt.savefig(gantt_chart_assigned_path, bbox_inches="tight")

# Display the updated Gantt chart
plt.show()
