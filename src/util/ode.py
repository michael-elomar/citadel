# This module may contain several other methods for ODE solutions such as
# Euler, RK2, Predictor-Corrector etc.

class RK2:
    """_summary_

    an instance of this class can calculate the required increment for the current state to reach the next time
    instant. in other words, calculate dX such that we can get X[i+1] = X[i] + dX.

    See  https://en.wikipedia.org/wiki/Runge–Kutta_methods

    nbStates : the dimension of the state vector
    nbInputs : the number of inputs to the system, or the dimension of the input vector

    t: Current time instant
    dt: The time increment. Default value dt = 0.001 
    """

    def __init__(self, nb_states, dt=0.001) -> None:
        self.nb_states = nb_states
        self.dt = dt

    def __call__(self, x, t, f, *args):
        """_summary_

            Uses RK4 Algorithm to calculate the appropriate increment in order to move forward in time.
            Returns : dx s.t. x[i+1] = x[i] + dx
        """
        assert len(x) == self.nb_states, "Wrong dimension of the current state vector"
        K1 = self.dt * f(t, x, *args)
        K2 = self.dt * f(t + self.dt, x + K1, *args)
        return (K1 + K2) / 2


class RK4:
    """
    At the current time instant, an instance of this class with calculate the required increment for the current state
    to reach the next time instant.

    See  https://en.wikipedia.org/wiki/Runge–Kutta_methods
    nbStates : Number of states
    nbInputs : Number of inputs
    t: Current instant
    dt : The time increment. By default, dt = 0.0005
    """

    def __init__(self, nb_states, dt=0.0005):
        self.nb_states = nb_states
        self.dt = dt

    """
    Uses RK4 Algorithm to calculate the appropriate increment in order to move forward in time.
    Returns : dX s.t. x[i+1] = x[i] + dx
    """

    def __call__(self, x, t, f, *args):
        assert len(x) == self.nb_states
        K1 = self.dt * f(t, x, *args)
        K2 = self.dt * f(t + 0.5 * self.dt, x + 0.5 * K1, *args)
        K3 = self.dt * f(t + 0.5 * self.dt, x + 0.5 * K2, *args)
        K4 = self.dt * f(t + self.dt, x + K3, *args)

        return (K1 + 2 * K2 + 2 * K3 + K4) / 6
