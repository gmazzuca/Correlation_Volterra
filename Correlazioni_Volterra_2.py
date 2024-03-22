''' Libraries'''

import numpy as np
from scipy.special import polygamma as Polygamma
from scipy.special import digamma as Digamma
from scipy.special import gamma as Gamma
import sys
from numba import vectorize,float64,float32,complex64,complex128



########################################
########################################
########################################
''' Generic Functions'''

@vectorize([float64(complex128),float32(complex128)])
def my_abs(x):
    # Compute the modulus of a complex number
    return np.sqrt(x.real**2 + x.imag**2)

@vectorize([float64(complex128),float32(complex128)])
def abs2(x):
    # Compute the modulus squared of a complex number
    return x.real**2 + x.imag**2

@vectorize([float64(complex128),float32(complex128)])
def abs4(x):
    # Compute the modulus quartic of a complex number
    return (x.real**2 + x.imag**2)**2

def DHT(x,axis = 0):
    ''' Discrete Hartley Transform orthogonal'''
    fx = np.fft.fft(x,norm = 'ortho',axis = axis)
    return np.real(fx) - np.imag(fx)


########################################
########################################
########################################


def Energy_Volterra(u):
    ''' Energy for volterra'''

    n,snap = u.shape

    if snap == 1:
        return np.sum(u)
    else:
        return np.sum(u,axis = 0)



def H1_Volterra(u):
    '''return \sum_j \log(u_j), which is a conserved quantity for Volterra'''
    n,snap = u.shape

    if snap == 1:
        return np.sum(np.log(u))
    else:
        return np.sum(np.log(u),axis = 0)

def H2_Volterra(u):
    '''return \sum_j \log(u_j), which is a conserved quantity for Volterra'''
    n,snap = u.shape

    if snap == 1:
        return np.sum(    2*u*np.roll(u,-1, axis = 0) + (u + np.roll(u,1, axis = 0))**2)
    else:
        return np.sum(2*u*np.roll(u,-1, axis = 0) + (u + np.roll(u,1, axis = 0))**2 ,axis = 0)

def evolution_periodic(u0, time = [0,10], time_step = 0.001):

    ''' Evolution according to the Euler symplectic scheme, as reported in Poisson integrators for Volterra lattice equations '''
    #setting the output vectors
    time_snap = len(time)
    particles = len(u0)

    u = np.zeros((particles,time_snap))
    u[:,0] = u0
    for j in range(1,time_snap):
        u[:,j] = symplectic_euler(u[:,j-1], time[j] - time[j-1], time_step)

    return u
    
def symplectic_euler(u,time=1, dt = 0.05):
    ''' Evolution according the symplectic Euler method'''
    u_even = u[0::2]
    u_odd = u[1::2]

    delta_t = 0

    while(delta_t < time):
    
        u_even = np.linalg.solve(np.diag((1-dt*np.ediff1d(u_odd, to_end = u_odd[0] - u_odd[-1]))),u_even)
        u_odd = u_odd*(1 + dt*np.roll(np.ediff1d(u_even, to_end = u_even[0] - u_even[-1]), 1))

        delta_t += dt

    u[0::2] = u_even
    u[1::2] = u_odd

    
    return u

def sample_general_gibbs(particles, beta=1,eta=1):
    u = np.random.chisquare(beta, size = particles)*(0.5)/eta
    
    return u


def correlation_volterra(u):

    n,snap = u.shape

    corr_u_main = np.zeros((n,snap))
    corr_log_main = np.zeros((n,snap))
    corr_mix_main = np.zeros((n,snap))

    log_u = np.log(u)

    
    for j in range(n):
        corr_u_main += u[j,0]*np.roll(u,-j,axis = 0)
        corr_log_main += log_u[j,0]*np.roll(log_u,-j,axis = 0)
        corr_mix_main += log_u[j,0]*np.roll(u,-j,axis = 0)

        
    return corr_u_main/n,  corr_log_main/n,  corr_mix_main/n 


def my_rk45(tfinal, u0, t_eval , tau = 0.1):

    if tfinal < t_eval[-1]:
        return 0

    
    # set the vector for the evolution
    u = np.zeros((len(u0), len(t_eval)))
    u[:,0] = u0

    for j in range(1,len(t_eval)):
        u[:,j] = step_evo(t_eval[j] - t_eval[j-1], u[:,j-1], tau)

    return u
    
def step_evo(deltat, y, tau = 0.1):

    time = 0
    


    while(time < deltat ):       
        k1 = tau * force1(y,np.roll(y, -1),np.roll(y,1))
    
        y_tmp = y + k1*0.5
        k2 = tau * force1(y_tmp, np.roll(y_tmp, -1), np.roll(y_tmp, 1)  )

        y_tmp = y + k2*0.5
        k3 = tau * force1(y_tmp, np.roll(y_tmp, -1), np.roll(y_tmp, 1)  )

        y_tmp = y + k3
        k4 = tau * force1(y_tmp, np.roll(y_tmp, -1), np.roll(y_tmp, 1)  )
        y = finalRK(y,k1, k2, k3,k4)
        time += tau
    

    return y


@vectorize([float64(float64,float64,float64,float64,float64),float32(float32,float32,float32,float32,float32)])
def finalRK(y,k1, k2, k3,k4):
    return y + (k1 + 2*k2 + 2*k3 + k4)/6

@vectorize([float64(float64,float64,float64),float32(float32,float32,float32)])
def force1(y,y_plus1,y_minus1):
    return y*(y_plus1 - y_minus1)



#################################################
#################################################
################# MAIN ##########################
#################################################
#################################################


if (len(sys.argv) < 8):
    print('error: give n snap tfinal trials beta eta numbering')
    exit()

# PARAMETERS FOR EVOLUTION FROM COMMAND LINE
n = int(sys.argv[1])
snap = int(sys.argv[2])
tfinal = float(sys.argv[3])
trials = int(sys.argv[4])
beta = float(sys.argv[5])
eta = float(sys.argv[6])
numbering = int(sys.argv[7])

time_snap = np.linspace(0,tfinal,num = snap)

mid_particle = int(n*0.5)


corr_u = np.zeros((n,snap))
corr_log = np.zeros((n,snap))
corr_mix = np.zeros((n,snap))

corr_u_main = np.zeros((n,snap))
corr_log_main = np.zeros((n,snap))
corr_mix_main = np.zeros((n,snap))

mean = [beta/eta*0.5, -np.log(eta) + Digamma(beta*0.5)]

for k in range(trials):
    # Evolution
    u0 = sample_general_gibbs(n,beta,eta)
    
    u  = my_rk45(tfinal,u0,time_snap,0.01)

    
    # Correlation
    corr_u_main_tmp, corr_log_main_tmp, corr_mix_main_tmp = correlation_volterra(u)
    corr_u_main += corr_u_main_tmp
    corr_log_main += corr_log_main_tmp
    corr_mix_main += corr_mix_main_tmp



corr_u = np.roll(corr_u_main/trials - mean[0]**2 ,mid_particle,axis = 0)
corr_log = np.roll(corr_log_main/trials - mean[1]**2 ,mid_particle,axis = 0)
corr_mix = np.roll(corr_mix_main/trials - mean[0]*mean[1] ,mid_particle,axis = 0)



#######################################
############## SAVE ###################
#######################################


np.savetxt('Data/Volterra_u_n_%d_beta_%0.1f_eta_%0.1f_time_%0.1f_numbering_%05d.dat' %(n,beta,eta,time_snap[-1], numbering), corr_u)
np.savetxt('Data/Volterra_logu_n_%d_beta_%0.1f_eta_%0.1f_time_%0.1f_numbering_%05d.dat' %(n,beta,eta,time_snap[-1], numbering), corr_log)
np.savetxt('Data/Volterra_mix_n_%d_beta_%0.1f_eta_%0.1f_time_%0.1f_numbering_%05d.dat' %(n,beta,eta,time_snap[-1], numbering), corr_mix)

if numbering == 0:
    np.savetxt('Data/Volterra_timesnap_n_%d_beta_%0.1f_eta_%0.1f_time_%0.1f.dat' %(n,beta,eta,time_snap[-1]), time_snap)
