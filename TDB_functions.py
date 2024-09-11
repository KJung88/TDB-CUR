import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import natsort

import time
from numpy import linalg
from scipy.linalg import expm

alpha_3 = [8.0/15.0, 5.0/12.0, 3.0/4.0]  
beta_3 =  [-17.0/60.0, -5.0/12.0, 0.0] 

def rk3_3stage_step(y, dt, L):
    nstage = 3
    n_var = np.shape(y)[0]
    y_array = np.zeros((n_var, 3))  # Solution, R.H.S., Intermediate

    y_array[:, 0] = np.copy(y)  # Initial state
    y_array[:, 1] = 0.0  # Initial R.H.S. as zero
    y_array[:, 2] = np.copy(y_array[:, 0])  # Intermediate

    for jstage in range(nstage):
        rr_tmp = np.dot(y_array[:, 0], L)  # R.H.S.
        y_array[:, 0] = y_array[:, 2] + alpha_3[jstage] * dt * rr_tmp  # Update the solution
        y_array[:, 2] = y_array[:, 0] + beta_3[jstage] * dt * rr_tmp  # Store for the next stage
    return y_array[:,0]

def rk3_3_stage_subcycle_rev(y, dt, L, n_sub, p_index, s_index, missing_p, missing_s):

    nstage = 3

    n_data = np.shape(y)[0]
    n_var = np.shape(y)[1]
    y_array = np.zeros(((n_data,n_var,3)))  ## first : solution vector, second: r.h.s vector, third : intermediate
    
    y_array[:,:,0] = np.copy(y)
    y_array[:,:,1] = 0.0
    y_array[:,:,2] = np.copy(y_array[:,:,0])

    y_sol = np.zeros((n_data,n_var))
    
    for jstage in range (0,nstage):            
        y_array[:,:,1] = 0.0                
        y_array[:,:,1] = y_array[:,:,0] @ L  ## R.H.S.
        y_array[:,missing_s,1] = 0.0  ## R.H.S. =0 if not F(:,s)
        y_update = np.copy(y_array)    
        for n in range (0,len(p_index)):
            p = p_index[n]            
            if (jstage == 1): 
                y_array[p,:,1] = np.dot(y_update[p,:,2], L)         ## alpha + beta = 0 (not stiff) 
            else:
                for _ in range(n_sub):
                    dt_sub = dt*(alpha_3[jstage] + beta_3[jstage])/n_sub
                    y_update[p,:,2] = rk3_3stage_step(y_update[p,:,2],dt_sub,L)                 
                y_array[p,:,1] = (y_update[p,:,2] - y_array[p,:,2]) /(dt_sub*n_sub)
            G_s = y_array[:,s_index,1]            
            Q, _ = np.linalg.qr(G_s, mode='reduced')
            Q_dagger = (np.linalg.inv(Q[p_index,:].T @ Q[p_index,:]) @ Q[p_index,:].T)
            F_appro = Q @ Q_dagger @ y_array[p_index,:,1]
        
            #Option 3: F(not p,sa) is approximated by a oblique projection 
            for q in range (0,len(missing_s)):
                y_array[missing_p,missing_s[q],1] = F_appro[missing_p,missing_s[q]]  ###       
        
        y_array[:,:,0] = y_array[:,:,2] + alpha_3[jstage] * dt * y_array[:,:,1]
        y_array[:,:,2] = y_array[:,:,0] + beta_3[jstage] * dt * y_array[:,:,1]
        
    return y_array[:,:,0]

alpha_5 = [0.225022458726, 0.544043312951, 0.144568243494, 0.786664342198, 0.348667178999]
beta_5 = [-0.173793152085, -0.163088487225, -0.517920839886, -0.194163057172, 0.000000000000]


def rk4_5stage_step(y, dt, L):
    nstage = 5
    n_var = np.shape(y)[0]
    y_array = np.zeros((n_var, 3))  # Solution, R.H.S., Intermediate

    y_array[:, 0] = np.copy(y)  # Initial state
    y_array[:, 1] = 0.0  # Initial R.H.S. as zero
    y_array[:, 2] = np.copy(y_array[:, 0])  # Intermediate

    for jstage in range(nstage):
        rr_tmp = np.dot(y_array[:, 0], L)  # R.H.S.
        y_array[:, 0] = y_array[:, 2] + alpha_5[jstage] * dt * rr_tmp  # Update the solution
        y_array[:, 2] = y_array[:, 0] + beta_5[jstage] * dt * rr_tmp  # Store for the next stage
    return y_array[:,0]

def rk4_5_stage_subcycle_rev(y, dt, L, n_sub, p_index, s_index, missing_p, missing_s):

    nstage = 5

    n_data = np.shape(y)[0]
    n_var = np.shape(y)[1]
    y_array = np.zeros(((n_data,n_var,3)))  ## first : solution vector, second: r.h.s vector, third : intermediate
    
    y_array[:,:,0] = np.copy(y)
    y_array[:,:,1] = 0.0
    y_array[:,:,2] = np.copy(y_array[:,:,0])

    y_sol = np.zeros((n_data,n_var))
    
    for jstage in range (0,nstage):            
        y_array[:,:,1] = 0.0                
        y_array[:,:,1] = y_array[:,:,0] @ L  ## R.H.S.
        y_array[:,missing_s,1] = 0.0  ## R.H.S. =0 if not F(:,s)
        y_update = np.copy(y_array)    
        for n in range (0,len(p_index)):
            p = p_index[n]            
            if (jstage ==2):
                y_array[p,:,1] = y_array[p,:, 0] @ L
            else:
                for _ in range(n_sub):
                    dt_sub = dt*(alpha_5[jstage] + beta_5[jstage])/n_sub
                    y_update[p,:,2] = rk4_5stage_step(y_update[p,:,2],dt_sub,L)                 
                y_array[p,:,1] = (y_update[p,:,2] - y_array[p,:,2]) /(dt_sub*n_sub)

        G_s = y_array[:,s_index,1]    
        
        Q, _ = np.linalg.qr(G_s, mode='reduced')

        Q_dagger = (np.linalg.inv(Q[p_index,:].T @ Q[p_index,:]) @ Q[p_index,:].T)
        F_appro = Q @ Q_dagger @ y_array[p_index,:,1]
        
        #Option 3: F(not p,sa) is approximated by a oblique projection 
        for q in range (0,len(missing_s)):
            y_array[missing_p,missing_s[q],1] = F_appro[missing_p,missing_s[q]]  ###       
        
        y_array[:,:,0] = y_array[:,:,2] + alpha_5[jstage] * dt * y_array[:,:,1]
        y_array[:,:,2] = y_array[:,:,0] + beta_5[jstage] * dt * y_array[:,:,1]
        
    return y_array[:,:,0]

def DEIM (A,n_aug):
    [n,m]=np.shape(A)
    
    U = np.zeros((n,m))
    P = np.zeros((n,m))
    
    phi = np.zeros(m+n_aug)

    ####### version 1
    # phi[0] = np.argmax(abs(A[:,0]))
    # P[int(phi[0]),0] = 1.0
    # U[:, 0] = A[:,0]
    
    # for l in range (1,m):
    #     uL = A[:,l];
    #     c_temp = np.matmul(np.transpose(P[:,0:l]),U[:,0:l])  ## L by L
    #     temp = np.matmul(np.transpose(P[:,0:l]),uL)    ## L by 1
    #     c =  np.linalg.lstsq(c_temp,temp,rcond=None)[0]  ## L by 1
    #     r = uL - np.matmul(U[:,0:l],c)   ## N by 1
    #     phi[l] = np.argmax(abs(r))
    #     U[:,l] = uL
    #     P[int(phi[l]),l] = 1.0

    # phi_int=phi.astype(int)
    # phi_sorted = natsort.natsorted(phi_int,reverse=True)

    ######### version 3
    I = [np.argmax(np.abs(A[:,0]))]
    phi[0] = np.argmax(np.abs(A[:,0]))
    for i in range(1,m):
        res = A[:,i] - A[:,:i] @ np.linalg.solve(A[I,:i], A[I,i])
        I  += [np.argmax(np.abs(res))]
        phi[i] = np.argmax(np.abs(res))

    phi_int=phi.astype(int)
    phi_sorted = natsort.natsorted(phi_int,reverse=True)    
    
    for i in range (0,n_aug):
        truncatedSVD=TruncatedSVD(m)
        U_truncated = truncatedSVD.fit_transform(A[phi_sorted[:m+i],:])
        g = truncatedSVD.singular_values_[-2]**2 - truncatedSVD.singular_values_[-1]**2                        
        VT = truncatedSVD.components_
        Ub = VT.T @ A.T
        R = g + np.sum(Ub**2, axis=0)
        R = R - np.sqrt(R**2 - 4*g*Ub[-1,:]**2)
        I = np.argsort(R)[::-1]        
        e = 0
        while I[e] in phi_sorted[:m+i]:
            e += 1        
        phi[m+i] = I[e]
        phi_int=phi.astype(int)
        phi_sorted = natsort.natsorted(phi_int,reverse=True)       
    
    return phi_sorted

def DEIM_penalty (A,fast_species):
    [n,m]=np.shape(A)
    
    U = np.zeros((n,m))
    P = np.zeros((n,m))
    
    phi = np.zeros(m)
    
    phi[0] = np.argmax(abs(A[:,0]))
    P[int(phi[0]),0] = 1.0
    U[:, 0] = A[:,0]
    
    for l in range (1,m):
        uL = A[:,l];
        c_temp = np.matmul(np.transpose(P[:,0:l]),U[:,0:l])  ## L by L
        temp = np.matmul(np.transpose(P[:,0:l]),uL)    ## L by 1
        c =  np.linalg.lstsq(c_temp,temp,rcond=None)[0]  ## L by 1
        r = uL - np.matmul(U[:,0:l],c)   ## N by 1
        r[fast_species] = 0.0
        phi[l] = np.argmax(abs(r))
        U[:,l] = uL
        P[int(phi[l]),l] = 1.0
    
    phi=phi.astype(int)
    phi_sorted = natsort.natsorted(phi)
    return phi_sorted