# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 16:16:36 2017

@author: karlo
"""


import numpy as np
import matplotlib.pyplot as plt

class ising:

    def __nn_energy2D(grid, J):
        """
        Calculates the initial energy of the 2D grid grid given
        the interaction energy J
        """
        N = np.shape(grid)[0]
        Energy = 0
        # going to the right and down in the grid
        # each spin has two neigbhours to the left 
        # and above
        for i in range(N):
            for j in range(N):
                up=i-1 # neighbour above
                if i==0: #periodic BC
                    up=N-1
                left=j-1 #neibour left
                if j==0: #periodic BC
                    left=N-1
                Energy += -J*grid[i,j]*(grid[up,j] + grid[i,left])
        return Energy
    
    def __deltaE(grid, J, pos):
        """
        Calculates the energy difference given the 2D grid grid, the 
        interaction energy J, and the position pos of the spin to
        be flipped
        """
        i = pos[0] #row
        j = pos[1] #column
        s = (i,j)
        N = np.shape(grid)[0]
        # left neighbour
        left = (i, j-1)
        if j==0: # periodic BC
            left = (i, N-1)
        # above neighbour
        up = (i-1, j)   
        if i==0:    # periodic BC
            up = (N-1, j)
        #right neighbour
        right = (i, j+1)
        if j==N-1:  # periodic BC
            right=(i,0)
        #below neighbour
        down = (i+1, j)
        if i==N-1:  # periodic BC
            down = (0,j)
        #energy change after flipping spin at (i,j) = (pos[0], pos[1])
        dE = 2*J*grid[s]*(grid[left]+grid[right]+grid[up]+grid[down])
        return dE
    
    
    def ising2D(self, T, N, J, init_grid=None, plot_flag=False):
        N=int(N)
        J=float(J)
        T=float(T)
        grid =  np.sign(.5-np.random.rand(N,N)) # Random initial configuration
        if init_grid != None:
            grid = init_grid
        t=int(1e4*(N*N)) # number of iterations
        Elist = []
        Mlist=[]
        Energy = self.__nn_energy2D(grid, J) # initial energy
        Magnet = np.sum(grid)   # initial magnetization
        
        #######################################################
        ## FIND THERMALIZATION
        ## We keep a tally of the average magnetization and 
        ## energy for the system, simultaneously for a cold 
        ## and a hot start. We define thermalization as the
        ## point when the average energy and magnetization
        ## (average over the last 10 sweeps, ie 10*N^2 
        ## iterations) for the cold start and the hot start
        ## differ by less than 10%
        #######################################################
        # initialize for the cold start
        trials_c = np.random.randint(0,N,(t, 2))
        grid_c = np.ones((N,N))
        Energy_c = self.__nn_energy2D(grid_c,J)
        Magnet_c = np.sum(grid_c)
        Elist_c =[]
        Mlist_c =[]
        # initialize for the hot start
        trials_h = np.random.randint(0,N,(t, 2))
        grid_h = np.sign(.5-np.random.rand(N,N))
        Energy_h = self.__nn_energy2D(grid_h,J)
        Magnet_h = np.sum(grid_h)
        Elist_h =[]
        Mlist_h =[]
    
        ## Perform the MCMC algorithm for hot/cold start simultaneously
        if init_grid==None:
            for k in range(t):
                # cold start 
                s_c=trials_c[k]
                dE_c = self.__deltaE(grid_c,J,s_c)
                p_c=np.exp(-dE_c/T)
                # acceptance test, including the case dE<0
                if np.random.rand() <= p_c:
                    grid_c[s_c[0],s_c[1]] = -1*grid_c[s_c[0],s_c[1]]
                    Energy_c = Energy_c + dE_c
                    Magnet_c=Magnet_c+2*grid_c[s_c[0],s_c[1]]            
                Mlist_c.append(Magnet_c)
                Elist_c.append(Energy_c)
                
                # hot start
                s_h = trials_h[k]
                dE_h = self.__deltaE(grid_h,J,s_h)
                p_h = np.exp(-dE_h/T)  
                # acceptance test, including the case dE<0
                if np.random.rand() <= p_h:
                    grid_h[s_h[0],s_h[1]] = -1*grid_h[s_h[0],s_h[1]]
                    Energy_h = Energy_h + dE_h
                    Magnet_h = Magnet_h+2*grid_h[s_h[0],s_h[1]]
                Mlist_h.append(Magnet_h)
                Elist_h.append(Energy_h)
                
                ########################################################
                ## THERMALIZATION CHECK
                ## Every 10*N^2 iterations (10 sweeps), we
                ## calculate the average energy and magnetization
                ## over the last 10 sweeps for the hot and cold 
                ## starts. If the average energy and magnetization for hot
                ## and cold starts do not differ by more than 5% (arbitrary,
                ## but works well), then we declare the system thermalized
                ## and then assign the last thermalized configuration
                ## to be the starting point of the main algorithm.
                ########################################################
                if (k>10*N*N) & (k % (10*N*N) == 0):    
                    M_h = np.sum(np.abs(Mlist_h[k-N*N*10:]))/np.size(Mlist_h[k-N*N*10:])/(N*N)
                    E_h = np.sum(Elist_h[k-N*N*10:])/np.size(Elist_h[k-N*N*10:])/(N*N)
                    M_c = np.sum(np.abs(Mlist_c[k-N*N*10:]))/np.size(Mlist_c[k-N*N*10:])/(N*N)
                    E_c = np.sum(Elist_c[k-N*N*10:])/np.size(Elist_c[k-N*N*10:])/(N*N)
                    diff = np.sqrt(np.power(np.abs(M_c - M_h)/np.abs(M_h),2) + np.power(np.abs(E_c-E_h)/np.abs(E_h),2))
                    if diff < 0.05:
                        grid=grid_h
                        break
                if int(k)==int(t-1): 
                    grid=grid_h   
        
        ## Perform the MCMC algorithm for the thermalized configurations
        Energy = self.__nn_energy2D(grid,J) ## re-initialize for the thermalized grid config
        Magnet= np.sum(grid)  ## re-initialize for the thermalized grid config
        trials = np.random.randint(0,N,(t, 2))       
        for j in range(t):
            s=trials[j]
            dE = self.__deltaE(grid,J,s)
            p=np.exp(-dE/T)
            # acceptance test, including the case dE<0
            if np.random.rand() <= p:
                grid[s[0],s[1]] = -1*grid[s[0],s[1]]
                Energy = Energy + dE
                Magnet=Magnet+2*grid[s[0],s[1]]
            Mlist.append(Magnet)
            Elist.append(Energy)
    
        # post processing the obtained data
        Mlist = np.array(Mlist) # use numpy array (more versatile)
        Elist = np.array(Elist) 
        Mlist = np.abs(Mlist)   # absolute value of the magnetization is needed
        
        # calculate quntities of interest (magnetization, 
        # energy, magnetic susceptibility, heat capacity all per spin)
        M = np.sum(Mlist)/np.size(Mlist)/(N*N)
        E = np.sum(Elist)/np.size(Elist)/(N*N)
        chi=(np.sum(Mlist**2)/np.size(Mlist)-np.sum(Mlist)**2/np.size(Mlist)**2)/T/(N*N);
        Cv=(np.sum(Elist**2)/np.size(Elist)-np.sum(Elist)**2/np.size(Elist)**2)/(T**2)/(N*N);
        
        #Normalize of the number of sites
        Mlist = Mlist/(N*N)     
        Elist = Elist/(N*N)
        
        #for plot of location of thermalization
        #where k is the iteration at which 
        #thermalization criteria was met
        if init_grid==None:
            therm_x = k*np.ones(10)
            therm_y = np.linspace(-2.50,1.50,10)
        else: 
            k = 0
    
        
        # plot the evolution of the MCMC algorithm
        if plot_flag:
            if init_grid==None:
                plt.plot([i for i in range(np.size(Mlist_h))],np.abs(Mlist_h)/(N*N), 
                         label="Hot start magnetization", linestyle='--')
                plt.plot([i for i in range(np.size(Mlist_c))],np.abs(Mlist_c)/(N*N), 
                         label="Cold start magnetization", linestyle='-.')
                plt.plot([i for i in range(np.size(Elist_h))],np.array(Elist_h)/(N*N), 
                         label="Hot start energy", linestyle='--')
                plt.plot([i for i in range(np.size(Elist_c))],np.array(Elist_c)/(N*N), 
                         label="Cold start energy", linestyle='-.')
                plt.plot(therm_x,therm_y, label="Location of thermalization")
                plt.plot([i+k for i in range(np.size(Elist))],np.array(Elist),
                         label="Thermalized Energy")
                plt.plot([i+k for i in range(np.size(Mlist))],np.array(Mlist),
                         label="Thermalized magnetization")
                plt.title("Evolution of MCMC for T="+str(T)+", N="+str(N)+", J="+str(J))
                plt.xlabel("Iteration")
                plt.ylabel("Energy / Magnetization")
                #plt.legend(loc='center right')
                plt.tight_layout()
                plt.savefig('Ising2D T=' + str(T)+", N="+str(N)+", J="+str(J)+'.png', dpi=600)
            else:
                plt.plot([i for i in range(np.size(Elist))],np.array(Elist))
                plt.plot([i for i in range(np.size(Mlist))],np.array(Mlist))
                
        
        return M, E,chi,Cv,k
    
    def ising1D(T, N, J, init_grid=None, hot_start=True):
        N=int(N)
        J=float(J)
        T=float(T)
        
        if (init_grid==None) & (hot_start):
            grid =  np.sign(.5-np.random.rand(N))
        elif (init_grid==None) & (not hot_start):
            grid = np.ones(N)
        else:
            grid = init_grid
            
        t=int(1e4*N)
        Elist = np.zeros(t)
        Mlist=np.zeros(t)
        jm=N-1
        Energy = 0
        for j in range(N):
            Energy += -J*grid[j]*grid[jm]
            jm=j
        Magnet = np.sum(grid)
        trials = np.random.randint(0,N,t)
        
        for i in range(t):
            s=trials[i]
            if s!=0:
                left=grid[s-1]
            else:
                left=grid[N-1]
            if s!=N-1:
                right=grid[s+1]
            else:
                right=grid[0]
            
            dE = 2*J*grid[s]*(left+right)
            p=np.exp(-dE/T)
            if np.random.rand() <= p:
                grid[s] = -1*grid[s]
                Energy = Energy + dE
                Magnet=Magnet+2*grid[s]
            
            Mlist[i] = Magnet
            Elist[i] = Energy
        
        
        Mlist = [Mlist[i] for i in np.nonzero(Mlist)][0]
        Elist = [Elist[i] for i in np.nonzero(Elist)][0]
        Mlist = np.abs(Mlist)
        Mlist = Mlist/N
        Elist = Elist/N
        #plt.plot(Mlist)
        #plt.plot([i for i in range(np.size(Elist))],Elist)
        
        M = np.sum(Mlist[50*N:])/np.size(Mlist[50*N:])
        E = np.sum(Elist[50*N:])/np.size(Elist[50*N:])
        print("E = " + str(E))
        print("M = " + str(M))

        


