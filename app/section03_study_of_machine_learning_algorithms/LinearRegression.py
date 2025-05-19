# %%
from numpy import cov, var, sqrt, std, mean, array

# %%


class LinearRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.__correlation_coefficient = self.__correlation()
        self.__inclination = self.__inclination()
        self.__intercept = self.__intercept()

    def __correlation(self):
        convariation = cov(self.x, self.y, bias=True)[0][1]
        variance_x = var(self.x)
        variance_y = var(self.y)
        return convariation / sqrt(variance_x * variance_y)

    def __inclination(self):
        stdx = std(self.x)
        stdy = std(self.y)
        return self.__correlation_coefficient * (stdy / stdx)

    def __intercept(self):
        median_x = mean(self.x)
        median_y = mean(self.y)
        return median_y - median_x * self.__inclination

    def predict(self, value):
        return self.__intercept + (self.__inclination * value)


# %%
x = array([1, 2, 3, 4, 5])
y = array([2, 4, 6, 8, 10])

lr = LinearRegression(x, y)
prevision = lr.predict(6)
print(f"Prevision: {prevision}")

# %%