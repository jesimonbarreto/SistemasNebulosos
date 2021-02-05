from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import skfuzzy as fuzz
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter

# for flake8 check
Axes3D

# Problem: from service quality and food quality to tip amount
#x_service = np.arange(0, 10.01, 0.5)
#x_food = np.arange(0, 10.01, 0.5)
#x_tip = np.arange(0, 25.01, 1.0)



def fsin_x(x):
    return np.sin(x)

def generate_yi(x, p, q, cond):
    #condi é um vetor com o valor de len == len x em cada posição tem 0 a 1 pertencimento a classe A
    y = np.array([(i * p + q) if i>= cond[0] and i<= cond[1] else 0 for i in x])
    return y

def generate_sign_type(x, tri, sign_type, sigma = 0.5):
    if sign_type == 'tri':
        sign = fuzz.trimf(x, tri)
    elif sign_type == 'gaus':
        sign = fuzz.gaussmf(x, tri[1], sigma)
    
    return sign


def generate_ui(x, stt, fnl, sp, dist, sign_type = 'tri',const = 1, direc = 'right'):
    
    if direc == 'right':
        fnl -= dist
        trianf = [fnl-sp, fnl, fnl+sp]
        sign = generate_sign_type(x, trianf, sign_type)
        
    elif direc == 'left':
        stt += dist
        triani = [stt-sp, stt, stt+sp] 
        sign = generate_sign_type(x, triani, sign_type)
    else:
        fnl -= dist
        stt += dist
        trianf = [fnl-sp, fnl, fnl+sp]
        triani = [stt-sp, stt, stt+sp]
        sign = generate_sign_type(x, triani, sign_type) + generate_sign_type(x, trianf, sign_type)

    ai_const = x.copy()
    ai_const = np.array([const if i>= stt and i<= fnl else 0 for i in ai_const])
    ui = ai_const + sign
    ui = np.array([const if i>const else i for i in ui])
    return ui


def generate_y_final(yi , ui):
    n_samples = len(yi[0])
    n_condi = len(yi)
    Y = []
    
    for n in range(n_samples):
        vp = 0
        norm = 0
        for i in range(n_condi):
            norm += ui[i][n]
            vp += ui[i][n] * yi[i][n]
        
        Y.append(vp/norm)
    return np.array(Y)

def MSE(y, y_pred):
    mse = np.mean((y - y_pred)**2)
    return mse


#generate input 0 a 2pi
lim1 = 0
lim2 = 2 * np.pi
step = 100
input_x = np.linspace(lim1, lim2, step)
#generate result real
sin_x = fsin_x(input_x) 

#generate ui
#valores de determinada classe Fuzzy
angle = 0.5
dist = angle
start1, final1 = 0, np.pi/2
start2, final2 = np.pi/2, 3*np.pi/2
start3, final3 = 3*np.pi/2, 2*np.pi

u1 = generate_ui(input_x.copy(), start1, final1, angle, dist, sign_type = 'gaus', const = 1, direc = 'right')
u2 = generate_ui(input_x.copy(), start2, final2, angle, dist, sign_type = 'gaus', const = 1, direc = 'both')
u3 = generate_ui(input_x.copy(), start3, final3, angle, dist, sign_type = 'gaus', const = 1, direc = 'left')
ui = [u1,u2,u3]

#generate yi valores de p e q obtidos com a analise
p1, q1, cond1 = 2/np.pi, 0, [start1, final1]
p2, q2, cond2 = -2/np.pi, 2, [start2, final2]
p3, q3, cond3 = 2/np.pi, -4, [start3, final3]

y1 = generate_yi(input_x.copy(), p1, q1, cond1)
y2 = generate_yi(input_x.copy(), p2, q2, cond2)
y3 = generate_yi(input_x.copy(), p3, q3, cond3)
yi = [y1,y2,y3]

#generate final
Y = generate_y_final(yi , ui)

mse = MSE(sin_x,Y)
print('Erro medio quadratico: {:.5f}'.format(mse))

#plt.plot(input_x, label="u1", marker=".")
plt.plot(input_x,u1, label="u1", marker=".")
plt.plot(input_x,u2, label="u2", marker=".")
plt.plot(input_x,u3, label="u3", marker=".")
plt.plot(input_x,y1, label="y1", marker=".")
plt.plot(input_x,y2, label="y2", marker=".")
plt.plot(input_x,y3, label="y3", marker=".")
plt.plot(input_x,sin_x, label="sinX", marker=".")
plt.plot(input_x,Y, label="Y", marker=".")
plt.legend(loc="upper left")
plt.show()


'''plt.subplot(row, col, 1)
plt.title("Service Quality")
plt.plot(x_service, service_low, label="low", marker=".")
plt.plot(x_service, service_middle, label="middle", marker=".")
plt.plot(x_service, service_high, label="high", marker=".")
plt.plot(service_score, 0.0, label="service score", marker="D")
plt.plot(service_score, service_low_degree,
         label="low degree", marker="o")
plt.plot(service_score, service_middle_degree,
         label="middle degree", marker="o")
plt.plot(service_score, service_high_degree,
         label="high degree", marker="o")
plt.legend(loc="upper left")


plt.subplot(row, col, 2)
plt.title("Food Quality")
plt.plot(x_food, food_low, label="low", marker=".")
plt.plot(x_food, food_middle, label="middle", marker=".")
plt.plot(x_food, food_high, label="high", marker=".")
plt.plot(food_score, 0.0, label="food score", marker="D")
plt.plot(food_score, food_low_degree, label="low degree", marker="o")
plt.plot(food_score, food_middle_degree, label="middle degree", marker="o")
plt.plot(food_score, food_high_degree, label="high degree", marker="o")
plt.legend(loc="upper left")

# =======================================
# z = f(x, y)
"""
# should use 3D display
plt.subplot(row, col, 3)
plt.title("Tip")
plt.plot(x_tip, tip_low, label="low", marker=".")
plt.plot(x_tip, tip_middle, label="middle", marker=".")
plt.plot(x_tip, tip_high, label="high", marker=".")
plt.legend(loc="upper left")
"""
ax3 = fig.add_subplot(row, col, 3, projection="3d")
plt.title("Tip Equation: low/middle/high")

ax3.set_xlabel("Food")
ax3.set_ylabel("Service")
ax3.set_zlabel("Tip")
ax3.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
surf1 = ax3.plot_surface(f, s, tip_low_grid, cmap=cm.coolwarm, linewidth=0)
surf2 = ax3.plot_surface(f, s, tip_middle_grid, cmap=cm.coolwarm, linewidth=0)
surf3 = ax3.plot_surface(f, s, tip_high_grid, cmap=cm.coolwarm, linewidth=0)

# =======================================
# Mamdani (max-min) inference method:
# * min because of logic 'and' connective.
# 1) low_degree <-> tip_low
# 2) middle_degree <-> tip_middle
# 3) high_degree <-> tip_high

# =======================================
# bad food OR bad service
low_degree = np.fmax(service_low_degree, food_low_degree)
# medium service
middle_degree = service_middle_degree
# good food OR good service
high_degree = np.fmax(service_high_degree, food_high_degree)

plt.subplot(row, col, 4)
plt.title("Some Fuzzy Rules")
t = ("bad food or bad service <-> bad\n"
     "medium service <-> middle\n"
     "good food or good service <-> good")
plt.text(0.1, 0.5, t)

plt.subplot(row, col, 5)
plt.title("Tip Activation: Sugeno Inference Method")

# Apply the equaltion:
w1 = low_degree
w2 = middle_degree
w3 = high_degree

z1 = 5.0 + 0.2 * food_score + 0.2 * service_score
z2 = 5.0 + 0.5 * food_score + 0.5 * service_score
z3 = 5 + 1.0 * food_score + 1.0 * service_score
z = (w1 * z1 + w2 * z2 + w3 * z3) / (w1 + w2 + w3)
print(z)

plt.plot(z1, w1, label="low tip", marker=".")
plt.xlim(0, 25)
plt.vlines(z1, 0.0, w1)
plt.plot(z2, w2, label="middle tip", marker=".")
plt.vlines(z2, 0.0, w2)
plt.plot(z3, w3, label="high tip", marker=".")
plt.vlines(z3, 0.0, w3)
plt.plot(z, 0.0, label="final tip", marker="o")
plt.legend(loc="upper left")

# =======================================
ax6 = fig.add_subplot(row, col, 6, projection="3d")
plt.title("Whole Tip Response Curve")

for i in range(0, len(f)):
    for j in range(0, len(s)):
        x = f[i, j]
        y = s[i, j]
        f_low_degree = fuzz.interp_membership(x_food, food_low, x)
        f_middle_degree = fuzz.interp_membership(x_food, food_middle, x)
        f_high_degree = fuzz.interp_membership(x_food, food_high, x)

        s_low_degree = fuzz.interp_membership(x_service, service_low, y)
        s_middle_degree = fuzz.interp_membership(x_service, service_middle, y)
        s_high_degree = fuzz.interp_membership(x_service, service_high, y)

        w1 = np.fmax(s_low_degree, f_low_degree)
        w2 = s_middle_degree
        w3 = np.fmax(s_high_degree, f_high_degree)

        tip_high_grid[i, j] = (w1 * tip_low_grid[i, j]
                               + w2 * tip_middle_grid[i, j]
                               + w3 * tip_high_grid[i, j]) / (w1 + w2 + w3)

ax6.set_xlabel("Food")
ax6.set_ylabel("Service")
ax6.set_zlabel("Tip")
ax6.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
surf6 = ax6.plot_surface(f, s, tip_high_grid, cmap=cm.coolwarm, linewidth=0)

plt.savefig("img/8-tipping-problem-sugeno.png")
plt.show()'''