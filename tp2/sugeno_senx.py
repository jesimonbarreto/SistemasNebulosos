from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import skfuzzy as fuzz
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter

# for flake8 check
#Axes3D

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

