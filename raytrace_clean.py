# -*- coding: utf-8 -*-
"""
D.Hill, dominicwhill@gmail.com
Ray trace Python test program (for translation into Fortran)
Solves the ray trace equations to find the propagation of laser light through a plasma.

Set up for a linear density ramp test
c-- THIS RAY TRACE FOLLOWS THE ALGORITHM OUTLINED IN 
  Benkevitch et al. Algorithm for Tracing Radio Rays in Solar Corona
          .and. Chromosphere, arxiV:1006.5635v3 (2010) 


Currently set up to perform a reflection test:
Rays incident at different angles propagate up a linear density ramp. (from right to left)

"""
import numpy as np
#import sys, os, re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gspec
import getpass,site
import raytrace_aux as RA


#----------------------------
pi = np.pi
c=299792458.0
epsilon = 8.8541878e-12
q_e = 1.602e-19
m_e = 9.11e-31
m_p = 1.67e-27
k_b = 1.38e-23
epsilon0 = 8.854e-12
pythonoffset=-1

#-- Reflection test
reflection_test = True
#----------------------------
#---- IMPACT variables
if reflection_test:
    xmint,xmaxt = 0.0,100.0
    ymint,ymaxt = -100.0,100.0
else:
    xmint,xmaxt = 0.0,20.0
    ymint,ymaxt = -1.0,1.0


# initialise grid
dx = 0.5
nxm,nym = 400,400
ngc = 1
dx = (xmaxt - xmint)/float(nxm)
dy = (ymaxt - ymint)/float(nym)
xb = np.linspace(xmint-dx,xmaxt+dx,nxm+1+2*ngc)
yb = np.linspace(ymint-dy,ymaxt+dy,nym+1+2*ngc)
xc = 0.5*(xb[1:] + xb[:-1])
yc = 0.5*(yb[1:] + yb[:-1])

qn_vec = np.zeros((nym,nxm))
cellmin = xb[1] - xb[0] # I assume cellmin = dx_min



#-------- laser input variables ----------------------------------------
lambda_mu = 0.35 # laser wavelength in micrometers
w_laser=2.0*np.pi*c/((lambda_mu*1e-6))#856.5e12*2*pi
qlaser_time = 1.1e-9    #!Total time laser is on
#ymint,ymaxt = 
qlaser_energy =  2.5e3 #14.e3*(cos(zmint)-cos(zmaxt))*(ymaxt-ymint)/(4.*pi*pi)  #!2.5e3  !Total energy of the laser
pulse_type = 2         # !1=Gaussian pulse, 2=square pulse
max_intensity = qlaser_energy/(qlaser_time*5./6)
ramp_time = qlaser_time/6.         
# BCS
yrefl = False # reflective bcs on
#   Algorithm settings:----------
dv_max = 1e4 # max ray velocity change per time step
step_max = cellmin/3  #   !The max size of the step in the discrete laser trace algorithm. Maximum advised value of qdx/3
step_min = cellmin/200
qmax_energy_inc=1.  # !Maximum fractional energy increase for a cell due to laser energy deposition (this restricts the hydrodynamic timestep)
#----------------Constants--------------------


qnc = 35486330.*epsilon*w_laser*w_laser # rho crit = (m_e epsilon_0 omega^2/e^2) #m^-3
qnc = (epsilon0*m_e/(q_e**2))*w_laser*w_laser
qnc_c2 = 2*qnc/(c*c)
qa1 = -4.493775894e16/qnc  #The constant here =-c*c/2
qkappaconst = 2.922e-12 


#------------------------------------------------------------------------------
#== test ne data
#print('ncrit = ',ne_ref)
ne_ref = qnc#(1e6)*1.0e19 # m^-3
ne_ref_cm3 = ne_ref/(1e6)
Z,Ar = 1.0,1.0
Te_ref = 20 # eV
lnLambda = 4.0
#--- compute collision time etc.
dict = RA.impact_inputs(ne_ref_cm3,Te_ref,Z,1.0,Ar) # Bz must be in tesla because this is the units of gorgon data
lambda_mfp = dict['lambda_mfp']
vte = dict['vte']
tau_ei = dict['tau_ei']
logLambda = dict['log_lambda']
print(' vte = ', vte )
#------ normalisations------------
norm_te = 1.0/(2.0*Te_ref) # eV
norm_ne = 1.0/(ne_ref) # cm^-3
norm_vel = 1.0/(vte) # m/s -> v_te
norm_R = 1.0/(lambda_mfp) # m -> lambda_mfp
norm_Bz = 1.0/dict['Bz_norm']
#---------------------------------
# linear ramp
ramp = np.linspace(qnc*0.8,qnc*1.5,nxm+2*ngc)
dxn0 = 0.01
if reflection_test:
    ramp = dxn0*qnc*xc#np.linspace(0.01*n_crit,n_crit)

#-------------------------------------------
#--- input ================================
dxe0 = dxn0 # should be minus dxn0 however we flip ramp later 
x0_in = xmaxt - dx
y0_in = ymint*0.9
e0 = 1.0 - dxn0*(xmaxt-x0_in)

X,ramp_2d = np.meshgrid(np.ones((nym+2*ngc)),ramp[::-1])
#----------
#n_e = np.ones((nxm+2*ngc,nym+2*ngc))*qnc*0.5##ramp_2d##np.ones((nxm,nym))*qnc*0.5
n_e = ramp_2d##np.ones((nxm,nym))*qnc*0.5

T_e = np.ones((nxm+2*ngc,nym+2*ngc))*2.0*Te_ref
n_crit = qnc
n_crit_norm = n_crit/ne_ref
rho_vac = qnc*0.00001
rho_min = qnc*0.001
n_e_min = rho_min
epsilon_dielectric = np.ones((np.shape(n_e))) - n_e/n_crit
vgroup = np.where(n_e<n_crit,c*np.abs(epsilon_dielectric)**0.5,0.0)
#---------->>>> 
# -- calculate the ray absorption coefficient
abs_coeff = RA.calc_abs(n_e/ne_ref,T_e/(2.0*Te_ref),vgroup,dict)
#<<<<---------


#------------------------------------------------------------------------------

class ray:
    def __init__(self,rayid):
        self.rayid = rayid
        self.xpos_list = []
        self.ypos_list = []
        self.vx_list = []
        self.vy_list = []
        self.global_array = np.zeros((5))
#------------------------------------------------------------------------------

class ray_list:

    def __init__(self,nrays):
        #self.global_array = np.zeros((5,nrays))
        #self.global_array = self.laser_init_1D(-c,0.0,self.global_array)
        self.ray_list = []
        
        if reflection_test:
            self.angle_arr= np.array([10.0,30.0,60.0])*(2.0*np.pi/360.0)
            #self.global_array = self.laser_init_1D_refltest(self.angle_arr,self.global_array)  
            for ii in range(nrays):
                ray_loc = ray(ii)
                #---- initalise ray
                ray_loc.global_array= self.laser_init_1D_refltest(self.angle_arr[ii])
                self.ray_list.append(ray_loc)
    
    def laser_init_1D_refltest_old(self,angle,glob_array):
        '''
            Initialise ray values for 1D reflection test
        '''
        # initialise position
        x0 = x0_in*1.0#xmaxt - dx
        y0 = y0_in*1.0 #ymint*0.9
        glob_array[index_iray('x'),:] = x0
        glob_array[index_iray('y'),:] = y0
        
        
        # initialise velocity
        glob_array[index_iray('vx'),:] = -1.0*c*np.cos(angle)
        glob_array[index_iray('vy'),:] = c*np.sin(angle)
        
        vrt0 = calc_vrt(glob_array[index_iray('vx'),:],glob_array[index_iray('vy'),:])
        # normalise to speed of light
        glob_array[index_iray('vx'),:] = glob_array[index_iray('vx'),:]*(c/vrt0)
        glob_array[index_iray('vy'),:] = glob_array[index_iray('vy'),:]*(c/vrt0)
        
        #--- relative power of each ray ---
        
        glob_array[index_iray('power'),:] = 1.0 # all the same power
        return glob_array

    def laser_init_1D_refltest(self,angle):
        '''
            Initialise ray values for 1D reflection test
        '''
        # initialise position
        x0 = x0_in*1.0#xmaxt - dx
        y0 = y0_in*1.0 #ymint*0.9
        glob_array = np.zeros((5))
        glob_array[index_iray('x')] = x0
        glob_array[index_iray('y')] = y0
        
        
        # initialise velocity
        glob_array[index_iray('vx')] = -1.0*c*np.cos(angle)
        glob_array[index_iray('vy')] = c*np.sin(angle)
        
        vrt0 = calc_vrt(glob_array[index_iray('vx')],glob_array[index_iray('vy')])
        # normalise to speed of light
        glob_array[index_iray('vx')] = glob_array[index_iray('vx')]*(c/vrt0)
        glob_array[index_iray('vy')] = glob_array[index_iray('vy')]*(c/vrt0)
        
        #--- relative power of each ray ---
        
        glob_array[index_iray('power')] = 1.0 # all the same power
        
        return glob_array

    def deg_to_rad(theta_deg):
        return 2.0*np.pi*(theta_deg/360.0)
#------------------------------------------------------------------------------



def get_raypath(x0,y,y0, alpha, e0, dxe0):
    '''
        raytrace analytic solution for reflection problem
    '''
    x = x0 - (e0/dxe0)* (np.cos(alpha)**2) + (1.0/dxe0)*((np.sqrt(e0)*np.cos(alpha)) + (dxe0/(2.0*np.sqrt(e0)*np.sin(alpha)))*(y-y0))**2
    return x
#------------------------------------------------------------------------------

def vertex(x0,y0,alpha,e0,dxe0):
    '''
        ray trace analytic solution for reflection problem
    '''
    xv = x0 - (e0/dxe0)*np.cos(alpha)**2
    yv = y0 - (2.0*e0/dxe0)*np.cos(alpha)*np.sin(alpha)
    return xv, yv

#------------------------------------------------------------------------------
def index_iray(var):
    '''
        gives the index of array in global_array
        var = 'x','y','vx','vy', 'power'
    '''
    ivar=-1
    if var=='x':
        ivar = 0
    elif var=='y':
        ivar = 1
    elif var=='vx':
        ivar = 2
    elif var=='vy':
        ivar = 3
    elif var=='power':
        ivar = 4
    
    if ivar==-1:
        print('ERROR attempting to access nonexistent property: ', var)
    return ivar
#------------------------------------------------------------------------------
def calc_vrt(vx,vy):
    '''
    calculate the ray normalisation
    
    vrt = calc_vrt(current_ray,global_array)
    '''
    vrt = np.sqrt(vx**2 + vy**2)
    return vrt


#------------------------------------------------------------------------------
def map_to_cc(xi,yi):
    '''
        map xi and yi indices onto cell centred values
    '''

    return xi-1,yi-1
#------------------------------------------------------------------------------

def cross_product(A,B):
    '''
        Computes the cross product of vectors A and B
    '''
    out_vec = np.zeros(np.shape(A))
    out_vec[0] = A[1]*B[2] - A[2]*B[1]
    out_vec[1] = A[2]*B[0] - A[0]*B[2]
    out_vec[2] = A[0]*B[1] - A[1]*B[0]
    return out_vec
#------------------------------------------------------------------------------

def find_cell(x1,x2,y1,y2,x_pos,y_pos):
    '''
        x1,x2,y1,y2 are cells from the last time step
        x1,x2,y1,y2 = find_cell(x1,x2,y1,y2,x_pos,y_pos)
        returns the new values of x1,x2,y1,y2
    '''
    #print('--- find cell -----')
    #print(xc[x1],xc[x2],yc[y1],yc[y2],x_pos,y_pos)
    if (x1 == -1):#!check the edges first:
        #!--X--
        if(x_pos<xc[1]):
            x1=0
        elif (x_pos>=xc[nxm-1]):
            x1=nxm+pythonoffset 
        
        #!--Y--
        if (y_pos < yc[1]):
            y1=0 
        elif (y_pos >= yc[nym-1]):
            y1=nym+pythonoffset
        
        
    else: #!check cell from last timestep, then adjacent cells:
        #!--X--
        if (x_pos< xc[x2]) and (x_pos >= xc[x1]):
            x1=x1
        elif (x_pos< xc[x1]) and (x_pos >= xc[x1-1]):
            x1=x1-1
        elif (x_pos< xc[x2+1]) and (x_pos >= xc[x2]):
            x1=x1+1
        else:
            x1=-1   #!Cell not found
        
        #!--Y--
        if (y_pos< yc[y2]) and (y_pos >= yc[y1]):
            y1=y1
        elif (y_pos< yc[y1]) and (y_pos >= yc[y1-1]):
            y1=y1-1
        elif (y_pos< yc[y2+1]) and (y_pos >= yc[y2]):
            y1=y1+1
        else: 
            y1=-1   #!Cell not found
    
    #!if still not found, search the whole domain
    #!--X--
    if (x1==-1):
        for ix in range(1,nxm):
            if (x_pos < xc[ix+1]) and (x_pos >= xc[ix]):
                x1=ix
                # x1 = ix+1 in fortran
                #break
    #!--Y--
    if y1 == -1:
        for iy in range(1,nym):
            if y_pos < yc[iy+1] and y_pos >= yc[iy]:
                y1=iy
                #y1 = iy+1 in fortran
                #break
    
    x2=x1+1
    y2=y1+1

    return x1,x2,y1,y2
#------------------------------------------------------------------------------

def ray_in_bounds(global_array,current_ray):
    '''
        l_inbounds, global_array = ray_in_bounds(global_array,current_ray)
    '''
    l_x = True
    l_y = True
    l_inbounds = False
    
    if (global_array[index_iray('x')] < xmint) or (global_array[index_iray('x')] >= xmaxt):
        l_x = False
    
    #!--Y--
    if(global_array[index_iray('y')] < ymint):
        if yrefl:
            # !Reflective BC
            global_array[index_iray('y')] = ymint + (ymint - global_array[index_iray('y')]) 
            global_array[index_iray('vy')] = - global_array[index_iray('vy')]
        else:
            l_y = False
        
    if (global_array[index_iray('y')] >= ymaxt):
        if yrefl:
            #!Reflective BC
            global_array[index_iray('y')]= ymaxt + (ymaxt- global_array[index_iray('y')]) 
            global_array[index_iray('vy')]= - global_array[index_iray('vy')]
        else:
            l_y = False
        
    #!Final answer: is the ray now within the bounds of the processor?
    if (l_x and l_y):
        l_inbounds = True
    return l_inbounds, global_array
#------------------------------------------------------------------------------
def find_cell_start(x1,x2,y1,y2,x_pos,y_pos):
    '''
    cell_start = find_cell_start(x1,x2,y1,y2,x_pos,y_pos)
    
    find the initial cell position for x_pos, y_pos
    '''
    cell_start = np.zeros((2),dtype=int)        
    #print('cell start = ')
    #print('x1,x2,y1,y2 = ',x1,x2,y1,y2,x_pos,y_pos )
    if (x_pos< xb[x2]):
        cell_start[0] = x1
    else:
        cell_start[0] = x2
        
    if (y_pos < yb[y2]):
        cell_start[1] = y1
    else:
        cell_start[1] = y2
        
    
    #==== if at the boundary ...
    if x1  >= nxm:
        cell_start[0] = nxm
    if y1 >= nym:
        cell_start[1] = nym
    return cell_start
#------------------------------------------------------------------------------
def reflect_at_critical(v0,r0,epsilon,grad_epsilon):
    '''
        depsilon = grad epsilon
        epsilon = np.sqrt(1 - ne/ncrit)
        l_inbounds, global_array = ray_in_bounds(global_array,current_ray)
        r1,v1 = reflect_at_critical(np.array([v0_vec[0],v0_vec[1]]),np.array([x_pos,y_pos]),eps,np.array([grad_epsilon[0],grad_epsilon[1]]))

    '''
    grad_epsilon_mag = (grad_epsilon[0]**2 + grad_epsilon[1]**2)
    L_cos_alpha = -(v0[0]*grad_epsilon[0] + v0[1]*grad_epsilon[1])/(grad_epsilon_mag) # v0.nabla epsilon/(nabla epsilon**2)
    if L_cos_alpha <=0.0:
        print('v0 = ', v0, 'grad_epsilon = ',grad_epsilon, 'already reflected')
        return [0,0],[0,0]
    print('v0 = ',v0, ' grad_epsilon = ',grad_epsilon, 'epsilon = ', epsilon)
    print( 'L_cos_alpha = ', L_cos_alpha,'grad_epsilon_mag = ',  grad_epsilon_mag)
    r1 = np.zeros((2))
    v1 = np.zeros((2))
    r1[0] = r0[0] + 4.0*epsilon*L_cos_alpha*(v0[0] + L_cos_alpha*grad_epsilon[0])
    r1[1] = r0[1] + 4.0*epsilon*L_cos_alpha*(v0[1] + L_cos_alpha*grad_epsilon[1])
    
    
    ##cos_alpha = L_cos_alpha*np.abs(np.sqrt(grad_epsilon_mag))
    v1[0] = v0[0] + 2.0*L_cos_alpha*grad_epsilon[0]
    v1[1] = v0[1] + 2.0*L_cos_alpha*grad_epsilon[1]
    
    return r1,v1



#------------------------------------------------------------------------------
def deposition(dt,power,kappa):
    '''
        returns the energy deposition within step dt
    '''
    power_dep = power*(1.0-np.exp(-dt*kappa))
    #current_ray_power -= power_dep
    #qpower_dep +=power_dep
    return power_dep
#------------------------------------------------------------------------------

def interpolate_arr(arr,xpos,ypos,x1,x2,y1,y2):
    '''
        arr_cc =  interpolate_arr(arr,xpos,ypos,x1,x2,y1,y2)
    '''
    
    dy1 = np.abs(ypos - yc[y1])
    dy2 = np.abs(ypos - yc[y2])
    
    dx1 = np.abs(xpos - xc[x1])
    dx2 = np.abs(xpos - xc[x2])
    
    dvol = (yc[y2]-yc[y1])*(xc[x2] - xc[x1])
    dvol = 1.0
    w1 = dx1*dy1/dvol
    w2 = dx1*dy2/dvol
    w3 = dx2*dy1/dvol
    w4 = dx2*dy2/dvol
    wsum = w1 + w2 + w3 + w4
    w1 /= wsum
    w2 /= wsum
    w3 /= wsum
    w4 /= wsum
    
    #check 
    #print('dx1, dx2 = ', dx1,dx2,dy1,dy2,dvol)
    #print('weights summed = ', w1 + w2 + w3 + w4)
    return w1*arr[x1,y1] + w2*arr[x1,y2] + w3*arr[x2,y1] + w4*arr[x2,y2]
    
#------------------------------------------------------------------------------
def propagate_rays(ray_in,qpower,current_ray):
    '''
        main function, propagate 'current_ray' through simulation domain
        until completely absorbed
        
        fills global array
    '''
    global_array = ray_in.global_array
    x1,x2,y1,y2 = -1,-1,-1,-1
    power_cutoff = global_array[index_iray('power')]/1000.0
    xpos_list,ypos_list = [],[]
    vx_list,vy_list = [],[]
    #depsilon_x_list,depsilon_y_list = [],[]
    i = 0
    l_inbounds = True
    rnec_prev = 0.0
    print ('glob power = ', global_array[index_iray('power')],power_cutoff)
    #while i <30000 and l_inbounds:
    
    while global_array[index_iray('power')]>power_cutoff and l_inbounds:
        # if using MPI domain decomposition - check that ray is still in domain        
        #--- get dt ================
    
        #nabla epsilon/epsilon 
        x_pos,y_pos = global_array[index_iray('x')],global_array[index_iray('y')]
        xpos_list.append(x_pos)
        ypos_list.append(y_pos)
        
        v0_vec = np.zeros((3))
        v0_vec[0] = global_array[index_iray('vx')]         
        v0_vec[1] = global_array[index_iray('vy')]
        
        
        
        x1,x2,y1,y2 = find_cell(x1,x2,y1,y2,x_pos,y_pos)
        cell_start = find_cell_start(x1,x2,y1,y2,x_pos,y_pos)
        ne_loc =  interpolate_arr(n_e,x_pos,y_pos,x1,x2,y1,y2) # interpolate ne to x_pos,y_pos
        epsilon = 1.0 - ne_loc/n_crit#(n_e[cell_start[0],cell_start[1]]/n_crit)
        


        del_eps_x = -1.0*(n_e[x2,cell_start[1]] - n_e[x1,cell_start[1]])/(2.0*(n_crit-n_e[cell_start[0],cell_start[1]]))
        del_eps_y = -1.0*(n_e[cell_start[0],y2] - n_e[cell_start[0],y1])/(2.0*(n_crit-n_e[cell_start[0],cell_start[1]]))        
    
        vrt = calc_vrt(v0_vec[0],v0_vec[1])
        # max ne gradient allowed
        del_ne_mag=np.max(np.array([1.0e20*np.sqrt((del_eps_x*1.0e-20)**2 + (del_eps_y*1.0e-20)**2),1e-18])) # max is depsilon/epsilon
        step = np.min(np.array([step_max/qnc_c2, vrt*dv_max/del_ne_mag]))
        step = step*qnc_c2
        step = np.max(np.array([step,step_min]))
        dt = step/vrt
  
        
        #--- grad epsilon = gradient of d_epsilon
        grad_epsilon = np.zeros((3))
        grad_epsilon[0] = -1.0*(n_e[x2,cell_start[1]] - n_e[x1,cell_start[1]])/(dx*n_crit)
        grad_epsilon[1] = -1.0*(n_e[cell_start[0],y2] - n_e[cell_start[0],y1])/(dy*(n_crit))
      
        d_epsilon = np.zeros((3))        
        d_epsilon[0] = -1.0*(n_e[x2,cell_start[1]] - n_e[x1,cell_start[1]])/(2.0*dx*(n_crit-n_e[cell_start[0],cell_start[1]]))
        d_epsilon[1] = -1.0*(n_e[cell_start[0],y2] - n_e[cell_start[0],y1])/(2.0*dy*(n_crit-n_e[cell_start[0],cell_start[1]]))
        d_epsilon[2] = 0.0
        #<-----
        dt_h = 0.5*dt 
        
        
        #----------------------------------------------------------------------------------
        # trace ray - update position + velocity
        # x[0] = x[0] + v[0]*dt
        
        # 1
        rx_half = global_array[index_iray('x')] + 0.5*v0_vec[0]*dt
        ry_half = global_array[index_iray('y')] + 0.5*v0_vec[1]*dt

        # 2
        omega0  = cross_product(d_epsilon,v0_vec*dt_h)
        
        # 3
        beta0 = (v0_vec + cross_product(v0_vec,omega0))*dt_h
        omega_h = cross_product(d_epsilon,beta0)
        
        # 4
        #omega_mag2 = omega_h[0]**2 + omega_h[1]**2 + omega_h[2]**2
        prefactor = 2.0/(1.0 + (omega_h[0]**2 + omega_h[1]**2 + omega_h[2]**2)*(dt_h**2))
        beta2 = prefactor*(v0_vec + cross_product(v0_vec,omega_h))
        dv1 = cross_product(beta2,omega_h)
        
        #-- don't accelerate or deposit eneryg in the vacuum
        if (n_e[cell_start[0],cell_start[1]]<=rho_vac):
            dv1[0] =0.0
            dv1[1] = 0.0
        

        
        v1x = v0_vec[0] + dv1[0]
        v1y  = v0_vec[1] + dv1[1] 
        vx_list.append(v1x)
        vy_list.append(v1y)
        
        
        # 5
        r1x = rx_half + 0.5*v1x*dt
        r1y = ry_half + 0.5*v1y*dt
        
        #------- critical surface reflection
        rnec_crit =  interpolate_arr(n_e/n_crit,r1x,r1y,x1,x2,y1,y2)
        
        if rnec_crit>=1.0:
            print(' \n\n reflecting: xpos = ', x_pos, 'v0_vec = ', v0_vec,'rnec_rit = ', rnec_crit,'rnec_prev = ', rnec_prev, 'x1,x2 = ', x1,x2,cell_start[0])
            ##reflect_at_critical reflect_at_critical(v0,r0,epsilon,depsilon)
            eps = epsilon#1.0- rnec_crit
            r1,v1 = reflect_at_critical(np.array([v0_vec[0],v0_vec[1]]),np.array([x_pos,y_pos]),eps,np.array([grad_epsilon[0],grad_epsilon[1]]))
            if not (np.sum(r1)==0 and np.sum(v1) == 0):
                r1x,r1y = r1[0],r1[1]
                v1x,v1y = v1[0],v1[1]
                print('reflected values = r1x = ', r1x,'ry = ', r1y, 'v1x = ', v1x,v1y)
            else:
                print('== no reflection ==\n')
                
        if (v1x*v0_vec[0]<0.0):
            
            print(' gradients = ', d_epsilon)
            print(' a reflection has occured ')
        global_array[index_iray('x')] = r1x
        global_array[index_iray('y')] = r1y
        global_array[index_iray('vx')]  = v1x
        global_array[index_iray('vy')]  = v1y
        
        rnec_prev = rnec_crit
        #================ normalise ray velocity ==============================

        if (n_e[cell_start[0],cell_start[1]]<n_crit) and (n_e[cell_start[0],cell_start[1]]> n_e_min):
        
            vrt = calc_vrt(global_array[index_iray('vx')],global_array[index_iray('vy')])
            vrt2 = 299792458.*np.sqrt(1-n_e[cell_start[0],cell_start[1]]/n_crit)
            global_array[index_iray('vx')] = global_array[index_iray('vx')]*(vrt2/vrt)
            global_array[index_iray('vy')] = global_array[index_iray('vy')]*(vrt2/vrt)   
        

        if (n_e[cell_start[0],cell_start[1]]>n_crit):
            vrt = calc_vrt(global_array[index_iray('vx')],global_array[index_iray('vy')])

        #----------------------------------------------------------------------
        
        # calculate energy deposition
        if (n_e[cell_start[0],cell_start[1]]<n_crit) and (n_e[cell_start[0],cell_start[1]]> n_e_min):
            kappa = abs_coeff[cell_start[0],cell_start[1]]#
            ray_power_init =  global_array[index_iray('power')]
            ray_power_dep = deposition(dt,global_array[index_iray('power')],kappa)
            # this is necessary because ne + xc are ccg values while qpower is a cc value
            xc_cell_start_cc,yc_cell_start = map_to_cc(cell_start[0],cell_start[1]) 
            # --- note we can improve this by instead allocating ray_power deposition
            #     split between the 4 surrounding cells using bilinear interpolation
            qpower[xc_cell_start_cc,yc_cell_start] = qpower[xc_cell_start_cc,yc_cell_start] + ray_power_dep
            global_array[index_iray('power')] = global_array[index_iray('power')] - ray_power_dep
            # interpolate this power deposition to IMPACT grid
            
        #print('--- ')
        i+=1
        #----------------------------------------
        # put in bcs -------------------->
        
        l_inbounds, laser.global_array = ray_in_bounds(global_array,current_ray)
        # reflect arrays at critical...
        #print('l_inbounds = ', l_inbounds,'r1x = ', r1x,r1y)
        #----
        #normalise velocity
    print('number of steps = ', i)
    
    
    ray_in.xpos_list = xpos_list
    ray_in.ypos_list = ypos_list
    ray_in.vx_list = vx_list
    ray_in.vy_list = vy_list

    return ray_in,qpower
#------------------------------------------------------------------------------

#-----------------------------------------------------------------------
#   MAIN program         
#--- update
nrays =3
laser = ray_list(nrays) # generate list of rays

qpower = np.zeros((nrays,nxm,nym)) # energy deposition
for iray in range(nrays):
    laser.ray_list[iray],qpower[iray,:,:] = propagate_rays(laser.ray_list[iray],qpower[iray,:,:],iray) # update global array matrix ray by ray
    
    #---normalise power deposited
    # --- send back to impact
    
#-----------------------------------------------------------------------
#   PLOT ---------------------------------------------------------------
fig = plt.figure(figsize=(10,6))
gs = gspec.GridSpec(2,3)
gs.update(wspace=0.7,hspace=0.08,bottom=0.1,left=0.13,right=0.96)
ax = fig.add_subplot(gs[0,0])
axlineout = fig.add_subplot(gs[1,0])
axn = axlineout.twinx()
ax2 = fig.add_subplot(gs[:,1])
ax3 = fig.add_subplot(gs[:,2])


lims = [xc[0],xc[-1],yc[0],yc[-1]]
pn = axn.plot(xc,n_e[:,0]/n_crit,c='k')
axn.set_ylabel(r'$n_e/n_{crit}$')
axlineout.set_ylabel(r'integrated energy deposition',color='C0')
#axp.plot(xc[1:-1],qpower[0,:,1],c='r')
qp_sum = np.sum(qpower,axis=0)
im = ax.imshow(qp_sum.T,extent=lims,aspect='auto')
ax.set_xticks([])
# qpower linout
pqpower = axlineout.plot(xc[1:-1],np.sum(qp_sum.T,axis=0),c='#1f77b4')
#plt.colorbar(im,ax=)
ax.set_title('energy deposited')
#ax.legend([pn,pqpower],[r'density', r'$q_{power}$'])
for iray in laser.ray_list:
    print(iray)
    xpos = iray.xpos_list
    ypos = iray.ypos_list
    vx = iray.vx_list
    vy = iray.vy_list
    
    p_raytrace, = ax2.plot(xpos,ypos)

    #---- analytic sol ----------
    alpha = np.arctan(vy[0]/vx[0])
    print(' alpha = ', alpha)
    x0_in = xpos[0]
    y0_in = ypos[0]
    e0 = 1.0 - dxe0*(xmaxt-x0_in)
    y_ray = ypos[::100]
    
    x_ray = get_raypath(x0_in,y_ray,y0_in, (np.pi+alpha), e0, dxe0)
    print(' e0 = ', e0, 'dxe0 = ', dxe0, 'alpha = ', alpha, ' = deg = ', alpha*(360.0/(2.0*np.pi)))
    xv,yv = vertex(x0_in,y0_in,alpha,e0,dxe0)
    p_analytic = ax2.scatter(x_ray,y_ray,c='k',marker='x')
    
    
    #-----------------------------------
    ax3.plot(vx,vy)
axlineout.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
#ax.set_xlabel(r'$x$')
ax3.set_title('velocity')
ax3.set_ylabel(r'$v_y$')
ax3.set_xlabel(r'$v_x$')
ax2.set_title('position')
ax2.set_ylabel(r'$y$')
ax2.set_xlabel(r'$x$')
ax2.legend([p_analytic,p_raytrace],['analytic solution', 'ray trace'])

plt.show()
# for iray in nrays
