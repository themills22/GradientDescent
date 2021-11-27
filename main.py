import GradientDescentMethods as Gradient
import MainHelpers
import PolynomialFunction as PolyFunc

# if __name__ == '__main__':
#     polynomial_function = PolyFunc.PolynomialFunction([1, 0, 0])
#     MainHelpers.plot_function(polynomial_function, -3, 3, [-2])

if __name__ == '__main__':
    learning_rate = float(input('Enter learning rate: '))
    polynomials = []
    polynomial = input('Enter polynomial: ')
    while polynomial != '':
        polynomials.append(float(polynomial))
        polynomial = input('Enter polynomial: ')

    num_epochs = 10000
    # cutoff_difference = 0.0001
    cutoff_difference = 1 * (10 ** -7)
    start_point = float(input('Enter start x coordinate: '))

    polynomial_function = PolyFunc.PolynomialFunction(polynomials)
    # x_steps = Gradient.simple_gradient_descent(learning_rate, num_epochs, cutoff_difference, polynomial_function,
    #                                            start_point)
    # MainHelpers.plot_x_steps(x_steps, polynomial_function)

    # x_steps = Gradient.hessian_gradient_descent(num_epochs, cutoff_difference, polynomial_function, start_point)
    # MainHelpers.plot_x_steps(x_steps, polynomial_function)
    #
    # momentum_rate = float(input('Enter momentum rate: '))
    # x_steps = Gradient.momentum_gradient_descent(learning_rate, momentum_rate, num_epochs, cutoff_difference,
    #                                              polynomial_function, start_point)
    # MainHelpers.plot_x_steps(x_steps, polynomial_function)
    #
    # x_steps = Gradient.nesterov_gradient_descent(learning_rate, momentum_rate, num_epochs, cutoff_difference,
    #                                              polynomial_function, start_point)
    # MainHelpers.plot_x_steps(x_steps, polynomial_function)

    x_steps = Gradient.adagrad_gradient_descent(learning_rate, num_epochs, cutoff_difference, polynomial_function,
                                                start_point)
    MainHelpers.plot_x_steps(x_steps, polynomial_function)

    x_steps = Gradient.adam_gradient_descent(learning_rate, num_epochs, cutoff_difference, polynomial_function,
                                             start_point, 0.999, 0.9)
    MainHelpers.plot_x_steps(x_steps, polynomial_function)
