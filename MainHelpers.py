import math
import matplotlib.pyplot as plt


def plot_x_steps(x_steps, function):
    lowest_x = math.inf
    highest_x = -math.inf
    y_steps = []
    for x_step in x_steps:
        y_steps.append(function.evaluate(x_step))
        if x_step < lowest_x:
            lowest_x = x_step

        if x_step > highest_x:
            highest_x = x_step

    start_x = lowest_x - 0.25 * (highest_x - lowest_x)
    end_x = highest_x + 0.25 * (highest_x - lowest_x)

    sample_points = 1000
    sample_difference = (end_x - start_x) / sample_points
    samples = []
    for i in range(sample_points):
        sample_x = start_x + (i * sample_difference)
        sample_y = function.evaluate(sample_x)
        samples.append((sample_x, sample_y))

    sample_point_x, sample_point_y = zip(*samples)
    plt.plot(sample_point_x, sample_point_y, '-g')
    plt.plot(x_steps, y_steps, '--ro')
    plt.plot(x_steps[0], y_steps[0], marker='*', markersize=20, color='blue')
    plt.title(f'Convergence in ({len(x_steps)} steps)')
    plt.show()


def plot_function(function, low, high, x_plot_points):
    sample_points = 1000
    sample_difference = (high - low) / sample_points
    samples = []
    for i in range(sample_points):
        sample_x = low + (i * sample_difference)
        sample_y = function.evaluate(sample_x)
        samples.append((sample_x, sample_y))

    y_plot_points = []
    for x_plot_point in x_plot_points:
        y_plot_points.append(function.evaluate(x_plot_point))

    sample_point_x, sample_point_y = zip(*samples)
    plt.plot(sample_point_x, sample_point_y, '-g')
    plt.plot(x_plot_points, y_plot_points, marker='o', color='r')
    plt.show()


