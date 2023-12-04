import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from GARCH_BS_model import s_HSI
from GARCH_BS_model import s_NKY
from GARCH_BS_model import s_SPX
from heston_simulation import sim_result_s
from risk_free import discount_rates

r_max = 0.1
r_min = 0.05
T = 2
num_simulations = 10000
num_steps = 2 * 252
delta_t = 1 / 252
r = discount_rates
# r[n] = discount rate at t=0.5*(n-1)
q = 0
denomination = 10000

methods = ["heston", "garch"]
indexs = ["HSI", "NKY", "SPX"]
S_0 = [17733.89063, 33354.14063, 4538.189941]

B_KnockIn = [0.5 * initial_spot for initial_spot in S_0]
B_KnockOut = [1.1 * initial_spot for initial_spot in S_0]

np.random.seed(4150)
z1 = np.random.normal(0, 1, size=(num_simulations, num_steps))

# # for stochastic vol, if time allows
# kappa = 2.0  # Mean reversion speed
# theta = 0.04  # Long-term average variance
# sigma = 0.3  # Volatility of the volatility
# rho = -0.5  # Correlation between asset price and volatility
# v0 = 0.04  # Initial variance
# z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(size=(num_simulations, num_steps))
#
# v = np.zeros((num_simulations, num_steps + 1, len(indexs)))
# v[:, 0, :] = v0

stock_prices = np.zeros((num_simulations, num_steps + 1, len(indexs)))
stock_prices[:, 0, :] = np.array(S_0)
relative_stock_prices = np.zeros((num_simulations, num_steps + 1, len(indexs)))
laggard = np.zeros((num_simulations, num_steps + 1))

#
# HSI_data_IV = pd.read_csv('Data/SURF_HSI.csv')
# [HSI_data_IV, HSI_forward] = IV_data_process(HSI_data_IV)
# HSI_interp_surface = get_surface(HSI_data_IV, 'HSI', func='linear', smooth=3)
#
# NKY_data_IV = pd.read_csv('Data/SURF_NKY.csv')
# [NKY_data_IV, NKY_forward] = IV_data_process(NKY_data_IV)
# NKY_interp_surface = get_surface(NKY_data_IV, 'NKY', func='cubic', smooth=-3)
#
# SPX_data_IV = pd.read_csv('Data/SURF_SPX.csv')
# [SPX_data_IV, SPX_forward] = IV_data_process(SPX_data_IV)
# SPX_interp_surface = get_surface(SPX_data_IV, 'SPX', func='cubic', smooth=-2)


def simulate_price(cs, method):
    global stock_prices
    # if method == "dupire":
        # for iSimulation in range(num_simulations):
        #     for iStep in range(num_steps):
        #         # Calculate current time
        #         t = iStep * delta_t
        #         for iIndex in range(len(indexs)):
        #             # Get local volatility using the function
        #             K = cs * S_0[iIndex] / stock_prices[iSimulation, iStep, iIndex] # CS % * S_0 / S_T
        #             if iIndex == 0:
        #                 spline = LocalVol.get_spline(HSI_data_IV, 'HSI', func='linear', smooth=3)
        #             elif iIndex == 1:
        #                 spline = LocalVol.get_spline(NKY_data_IV, 'NKY', func='cubic', smooth=-3)
        #             elif iIndex == 2:
        #                 spline = LocalVol.get_spline(SPX_data_IV, 'SPX', func='cubic', smooth=-2)
        #             vol = LocalVol.localvol_cal(K, T-t, spline)**0.5
        #             # vol = 0.23
        #             daily_return = (r - q - 0.5 * vol ** 2) * delta_t + (
        #                     vol * (delta_t ** (1 / 2)) * z1[iSimulation][iStep])
        #
        #             # Calculate stock price at each step
        #             stock_prices[iSimulation, iStep + 1, iIndex] = stock_prices[iSimulation, iStep, iIndex] * np.exp(
        #                 daily_return)
    if method == "garch":
        stock_prices[:, :, 0] = np.copy(s_HSI)
        stock_prices[:, :, 1] = np.copy(s_NKY)
        stock_prices[:, :, 2] = np.copy(s_SPX)
        # for iSimulation in range(num_simulations):
        #     for iStep in range(num_steps):
        #         stock_prices[iSimulation, iStep, 0] = s_HSI[iSimulation, iStep]
        #         stock_prices[iSimulation, iStep, 1] = s_NKY[iSimulation, iStep]
        #         stock_prices[iSimulation, iStep, 2] = s_SPX[iSimulation, iStep]
    # Heston
    elif method == "heston":
        stock_prices[:,1:,2] = sim_result_s.transpose((2,0,1))[:,:,0]
        stock_prices[:, 1:, 1] = sim_result_s.transpose((2, 0, 1))[:, :, 2]
        stock_prices[:, 1:, 0] = sim_result_s.transpose((2, 0, 1))[:, :, 1]


def plot_path(index):
    # INDEX: 0 = "HSI", 1 = "NKY", 2 = "SPX"
    plt.figure(figsize=(10, 6))
    for iSimulation in range(num_simulations):
        plt.plot(np.arange(num_steps + 1) * delta_t, stock_prices[iSimulation, :, index])
    plt.axhline(B_KnockIn[index], color='r', linestyle='-', label='Knock-In Level')
    plt.axhline(B_KnockOut[index], color='r', linestyle='--', label='Knock-Out Level')
    plt.xlabel('Time')
    plt.ylabel('Index Level')
    plt.title(indexs[index] + ' Price Paths')
    plt.grid(True)
    plt.show()


def variable_interest(iSimulation, obs_date, cs):  # e.g. obs_date = 3 for 3rd obs date (t = 1.5)
    global laggard
    if obs_date <= 0:
        return 0
    elif laggard[iSimulation][obs_date*126] >= cs:
        return r_max * denomination / 2 * np.exp(-r[obs_date-1]*0.5*obs_date) + variable_interest(iSimulation, obs_date - 1, cs)
    else:
        return r_min * denomination / 2 * np.exp(-r[obs_date-1]*0.5*obs_date) + variable_interest(iSimulation, obs_date - 1, cs)


def price(cs, method):
    global laggard, stock_prices
    simulate_price(cs, method)
    payoffs = np.zeros(num_simulations)
    for iIndex in range(len(indexs)):
        relative_stock_prices[:, :, iIndex] = stock_prices[:, :, iIndex]/S_0[iIndex]
    # Find the relative value of the laggard index for each simulation at each step
    laggard = np.min(relative_stock_prices, axis=2)
    # Check which simulations have hit the knock-out barrier at multiples of 126 steps
    knockout_hit = np.all(relative_stock_prices[:, ::126, :] >= 1.1, axis=2)
    # Find out when was the knock-out barrier hit
    times_hit_knockout = np.argmax(knockout_hit[:, :], axis=1)
    # Check which simulation have hit the knock-in barrier
    # Slower method (commented out)
    # knockin_hit = np.any(np.any(relative_stock_prices[:, :, :] <= 0.5, axis=1), axis=1)
    knockin_hit = np.any(laggard[:, :] <= 0.5, axis=1)
    for iSimulation in range(num_simulations):
        # Payoff if knock-out happens
        if times_hit_knockout[iSimulation]:
            # knock-out redemption
            payoffs[iSimulation] = denomination*np.exp(-r[times_hit_knockout[iSimulation]-1]*0.5*times_hit_knockout[iSimulation])
            # variable interests
            payoffs[iSimulation] += variable_interest(iSimulation, times_hit_knockout[iSimulation], cs)
        # Payoff if knock-in happens
        elif knockin_hit[iSimulation]:
            # final redemption
            payoffs[iSimulation] = denomination * min(1, laggard[iSimulation, num_steps]) * np.exp(-r[3]*T)
            payoffs[iSimulation] += variable_interest(iSimulation, 4, cs)
        # Payoff if neither knock-in nor knock-out happens
        else:
            payoffs[iSimulation] = denomination * np.exp(-r[3]*T)
            payoffs[iSimulation] += variable_interest(iSimulation, 4, cs)
    return np.mean(payoffs)


def find_cs(m): #m: methods
    for iMethod in range(0,2):
        print("Under " + m[iMethod] + " model")
        try:
            cs_solution = brentq(lambda cs: price(cs, m[iMethod]) - 0.98 * denomination, 0.00001, 2)
            print("Value of cs:", cs_solution)
            print("Price at cs_solution:", price(cs_solution, m[iMethod]))
        except ValueError:
            print("Not possible for any CS to make the price of the Note close to 98% of issue price. Maximum price = "
                  + str(price(0, m[iMethod])))


# test = sim_result_s

# print(price(0.00001, methods[1]))
# print(stock_prices[123, 243, 0])
# print(stock_prices[123, 243, 1])
# print(stock_prices[123, 243, 2])
# print(price(2, methods[1]))
# print(stock_prices[123, 243, 0])
# print(stock_prices[123, 243, 1])
# print(stock_prices[123, 243, 2])
find_cs(methods)

