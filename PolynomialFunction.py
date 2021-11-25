class PolynomialFunction:
    def __init__(self, polynomials):
        self.__polynomials = polynomials
        self.__num_polynomials = len(self.__polynomials)

    def evaluate(self, x):
        result = 0
        highest_index = self.__num_polynomials - 1
        power = highest_index
        while power >= 0:
            polynomial = self.__polynomials[highest_index - power]
            result += polynomial * (x ** power)
            power -= 1

        return result

    def gradient(self, x):
        result = 0
        highest_index = self.__num_polynomials - 1
        power = highest_index
        while power >= 1:
            polynomial = self.__polynomials[highest_index - power]
            multiplicative = power * polynomial
            result += multiplicative * (x ** (power - 1))
            power -= 1

        return result

    def hessian(self, x):
        result = 0
        highest_index = self.__num_polynomials - 1
        power = highest_index
        while power >= 2:
            polynomial = self.__polynomials[highest_index - power]
            multiplicative = power * (power - 1) * polynomial
            result += multiplicative * (x ** (power - 2))
            power -= 1

        return result
