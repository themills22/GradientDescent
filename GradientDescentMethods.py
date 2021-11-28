import math


def simple_gradient_descent(learning_rate, num_epochs, cutoff_difference, function, start_point):
    current_epoch = 0
    difference = math.inf
    x = start_point
    x_steps = [x]
    while current_epoch < num_epochs and difference > cutoff_difference:
        gradient = function.gradient(x)
        new_x = x - (learning_rate * gradient)
        difference = abs(x - new_x)
        x = new_x
        x_steps.append(x)
        current_epoch += 1

    return x_steps


def hessian_gradient_descent(num_epochs, cutoff_difference, function, start_point):
    current_epoch = 0
    difference = math.inf
    x = start_point
    x_steps = [x]
    while current_epoch < num_epochs and difference > cutoff_difference:
        gradient = function.gradient(x)
        hessian = function.hessian(x)
        new_x = x - (gradient / hessian)
        difference = abs(x - new_x)
        x = new_x
        x_steps.append(x)
        current_epoch += 1

    return x_steps


def momentum_gradient_descent(learning_rate, momentum_rate, num_epochs, cutoff_difference, function, start_point):
    current_epoch = 0
    difference = math.inf
    x = start_point
    previous_difference = 0
    x_steps = [x]
    while current_epoch < num_epochs and abs(difference) > cutoff_difference:
        gradient = function.gradient(x)
        new_x = x - ((learning_rate * gradient) + (momentum_rate * previous_difference))
        difference = x - new_x
        previous_difference = difference
        x = new_x
        x_steps.append(x)
        current_epoch += 1

    return x_steps


def nesterov_gradient_descent(learning_rate, momentum_rate, num_epochs, cutoff_difference, function, start_point):
    current_epoch = 0
    difference = math.inf
    x = start_point
    momentum = 0
    x_steps = [x]
    while current_epoch < num_epochs and abs(difference) > cutoff_difference:
        gradient = function.gradient(x - (momentum_rate * momentum))
        momentum = ((learning_rate * gradient) + (momentum_rate * momentum))
        new_x = x - momentum
        difference = x - new_x
        x = new_x
        x_steps.append(x)
        current_epoch += 1

    return x_steps


def adagrad_gradient_descent(learning_rate, num_epochs, cutoff_difference, function, start_point):
    current_epoch = 0
    difference = math.inf
    epsilon = 1 * (10 ** -8)
    update_divider = 0
    x = start_point
    x_steps = [x]
    while current_epoch < num_epochs and abs(difference) > cutoff_difference:
        gradient = function.gradient(x)
        if current_epoch < 10:
            new_x = x - (learning_rate * gradient)
        else:
            new_x = x - ((learning_rate / math.sqrt(update_divider + epsilon)) * gradient)
        update_divider += (gradient ** 2)
        difference = x - new_x
        x = new_x
        x_steps.append(x)
        current_epoch += 1

    return x_steps


def adam_gradient_descent(learning_rate, num_epochs, cutoff_difference, function, start_point, mean_rate,
                          variance_rate):
    current_epoch = 0
    difference = math.inf
    epsilon = 1 * (10 ** -8)
    mean_estimate = 0
    variance_estimate = 0
    x = start_point
    x_steps = [x]
    while current_epoch < num_epochs and abs(difference) > cutoff_difference:
        current_epoch += 1
        gradient = function.gradient(x)
        mean_estimate = (mean_rate * mean_estimate) + ((1 - mean_rate) * gradient)
        variance_estimate = (variance_rate * variance_estimate) + ((1 - variance_rate) * (gradient ** 2))
        if current_epoch < 10:
            new_x = x - (learning_rate * gradient)
        else:
            mean_bias = mean_estimate / (1 - (mean_rate ** current_epoch))
            variance_bias = variance_estimate / (1 - (variance_rate ** current_epoch))
            new_x = x - ((learning_rate * mean_bias) / (math.sqrt(variance_bias) + epsilon))

        difference = x - new_x
        x = new_x
        x_steps.append(x)

    return x_steps
