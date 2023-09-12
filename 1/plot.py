# pl: 65.  53.2
# 0.1: 66 54.2
# 0.01: 65.5   54
# 0.001: 68.6 57.8
# 0.0001: 67.1  56.9
# 1.015   1.0076 1.055 1.032

# pl: 65.  53.2
# 0.1: 79.7 70.6
# 0.01: 81.6   72.9
# 0.001: 81.2 71.5
# 0.0001: 67.1  56.9
# 1.015   1.0076 1.055 1.032



# pl: 85.8 77.0
# 0.1: 88.9 81.7 = 1.036
# 0.01: 87.9 = 1.024
# 0.001: 88.1 = 1.038
# 0.0001: 89.85 = 1.016

# pl: 85.4 = 1.04
# 0.1: 88.9 = 1.03
# 0.01: 88.0 = 
# 0.001: 89.2 = 1.045
# 0.0001: 85.9 = 1.005
import matplotlib.pyplot as plt

# Data
x_values = [0,0.1, 0.2, 0.3, 0.4]  # Custom x-axis values
y_values = [1.0076, 1.015, 1.055, 1.032,1]

# y_values = [1.027, 1.03, 1.045, 1.005,1]

# y_values = [1.036, 1.029, 1.04, 1.021,1]

# Custom tick labels
x_labels = [r'$\gamma_1$', r'$\gamma_2$', r'$\gamma_3$', r'$\gamma_4$', r'$\gamma_5$']

# Create a line plot
plt.plot(x_values, y_values, marker='o',linewidth = 2)

# Set custom x-axis tick labels
plt.xticks(x_values, x_labels,  fontsize=16)
plt.yticks(ticks=[1.0076, 1.015, 1.055, 1.032,1], labels=['{:.3f}'.format(val) for val in y_values], fontsize=16)

# Add labels and title
plt.xlabel('Gamma Values',  fontsize=16)
plt.ylabel('Y Values',  fontsize=16)
plt.title('Line Plot of Y Values vs. Gamma Values')

# Show the plot
plt.show()










