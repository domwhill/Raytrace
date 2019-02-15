import numpy as np
'''
    auxillary functions for ray trace
    stripped from IMPACT_norm module
'''
# constants
q_e = 1.602e-19
m_e = 9.11e-31
m_p = 1.67e-27
k_b = 1.38e-23
epsilon0 = 8.854e-12
##qa1 = -4.493775894e16/qnc  #The constant here =-c*c/2
qkappaconst = 2.922e-12 

#-----------------------------------------------------------------------

def extract_power(x):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return b    
#-----------------------------------------------------------------------


def impact_inputs(ne,Te,Z,Bz,Ar):
    '''
        ne, Te, Z, Bz, Ar are the IMPACT reference density [cm**-3], temperature [eV], ionisation (Z)
        magnetic field [T], relative atomic mass (Ar)
        
        returns a dictionary containing the normalisation parameters for the IMPACT/raytrace reference material
    '''
    
    dict = {}
    dict['ne'] = ne
    dict['Te'] = Te
    dict['Z'] = Z

    # Convert ne to 10**21 cm**-3
    ne = ne / 1.0e21
    ni = ne / Z

    # Calculate Coulomb np.logarithm.
    lambda_bar = (2.76e-10) / np.sqrt(Te) 
    b0 = (1.44e-9) * (Z / Te)

    if (lambda_bar > b0):
        log_lambda = np.log(Te) - np.log(ne)/2.0 - 0.16
    else: 
        log_lambda = (3.0/2.0)*np.log(Te) - np.log(Z) - np.log(ne)/2.0 - 1.8


    # Reference values
    #   v0 : thermal velocity [m/s],      t0 : collision time [s].
    #   nu0 : collision frequency [s**-1], l0 : mean free path [m])
    v0 = (0.5931e6) * (Te**0.5)
    t0 = (2.588e-16) * (Te**(1.5)) / (Z*Z*ni*log_lambda)
    nu0 = 1.0 / (t0)
    l0 = (1.535e-10) * (Te*Te) / (Z*Z*ni*log_lambda)

    # IMPACT inputs 
    wpe_by_nu_ei = 0.4618 * (Te**(1.5)) / (Z * np.sqrt(ne) * log_lambda)
    # for tau_B instead of tau_ei as in IMPACTA - note that ln_lam will be
    # different too
    #wpe_by_nu_ei = 0.4618 * (Te**(3/2)) / (Z * sqrt(ne) * log_lambda) * (3*sqrt(pi)/4)
    c_by_v0 = 1.0 / ( (1.9784e-3) * np.sqrt(Te) )
    prof_Bz_ave = (1.756e11) * Bz * t0

    # Display
    
    print('\nINPUT QUANTITIES')
    print('Density \t', ne * 1e21,'[cm**3]')

    print('Temperature \t', Te, '[eV]')
    print('Ionisation \t',Z, '[ ]')
    print('Bz \t\t', Bz, '[T]')

    print('\n IMPACT VARIABLES')
    print('log_lambda \t', log_lambda)

    print('wpe_by_nu_ei \t', wpe_by_nu_ei)
 
    print('c / v0 \t\t', c_by_v0)
    print('prof_Bz_ave \t', prof_Bz_ave)


    #clear arry
    print('\n\nPLASMA REFERENCE VARIABLES')
    print('Reference thermal velocity \t %1.5e [m/s]' % (v0))
    print('Reference collision time \t %1.5e [s]' % (t0))
    print('Reference collision frequency \t%1.5e [s**-1]' % (nu0))
    print('Reference mfp \t\t \t %1.5e [m]\n' % (l0))

    dict['vte'] = v0
    dict['tau_ei'] = t0
    dict['nu_ei'] = nu0
    dict['lambda_mfp'] = l0
    dict['c_over_vte'] = c_by_v0
    dict['log_lambda'] = log_lambda
    dict['wpe_over_nu_ei'] = wpe_by_nu_ei 
    dict['Bz_norm'] = prof_Bz_ave
    # --- get transport coeffs
    # if T in Kelvin...
    # kappa*gradT = (1/(msec))*(J/K) * (K/m) = kappa*k_b *grad (T/Kelvin)  
    # W/m^2= [kappa*gradT] = (1/(m*sec))*J/m = J/(m^2 sec) 
    
    
    return dict


#-----------------------------------------------------------------------
def calc_norms(var,normal_dict,sample =0.0,forced_power=[]):
    '''
        norm_const, ylab = calc_norms(var)
    '''
    c_fmt = '%3.2f'
    
    v_te = normal_dict['vte']
    tau_ei = normal_dict['tau_ei']
    nu_ei = normal_dict['nu_ei']
    lambda_mfp = normal_dict['lambda_mfp']
    n0,T0 = normal_dict['ne'], normal_dict['Te']
    #-----------------

    if var== 'Cx' or var=='Cy':
        
        norm_const = v_te*1e-3
        title = r'$' + var[0] + '_' + var[1] + r'$ [ $  kms^{-1}$ ]'
        
    elif var=='n':
        power = extract_power(n0)
        mod = r'$ 10^{' +str(power)  + '} $'

        norm_const = n0*(10**-power)
        title = r'$n_e$ [' + mod + r'$\si{cm^{-3}}$'+ r']'

    elif var=='Te':
        norm_const = 2.0*T0
        title = r'$T_e$ [ $eV$ ]'
        c_fmt = '%3.0f'
    elif var=='Ui':
        norm_const = (2.0/3.0)*(2.0*T0)
        title = r'$T_i$ [ $eV$ ]'
        c_fmt = '%3.0f'
    
    elif var=='Bz':
        norm_const = (m_e/(q_e*tau_ei))
        
        power = extract_power(sample*norm_const)
        norm_const*= (10**-power)
        ##print ' sample = ', sample, 'power = ', power

        var_name = '$B_z$'
        if power==0:
            mod = ''            
            units = r'[$si{T}$]'
            title = r'$B_z$ [$\si{T}$ ]'

        else:
            mod = r'$ 10^{' +str(power)  + '} $'
        
            units = r'[' + mod + '$\si{T}$]'
            title = r'$B_z$ [' + mod + '$\si{T}$ ]'
        c_fmt = '%1.1f'
        
    elif var=='wt':
        if len(forced_power)!=0:
            power = forced_power[0]
        else:
            power = extract_power(sample)
        
        if power!= 0:
            mod = r'$ 10^{' +str(power)  + '} $'
        else:
            mod = r''            
        ##mod = r'$ 10^{' +str(power)  + '} $'
        
        norm_const = 1.0*(10**(-power))
        
        title = mod + r'$\omega \tau_{ei}$'
        c_fmt = '%1.1f'
        
    elif var[0]=='E':
        #print 'DOING E-field - min sample = ', sample
        norm_const = (lambda_mfp/(tau_ei**2))*(m_e/q_e)
        power = extract_power(norm_const*sample)
        #print ' power extracted = ', power
        c_fmt = '%1.1f'
        mod = r'$ 10^{' +str(power)  + '}$'
        norm_const = norm_const*(10**-power)
        if power==0:
            mod=''           
        title = r'$' + var[0] + '_' + var[1] + r'$ [ ' + mod + '$V/m$ ]'
    
    elif var[0] =='q':
        
        c_fmt = '%1.1f'
        power = extract_power(sample)
        norm_const = 1.0*(10**-power)
        mod = r'$ 10^{' +str(power)  + '} $'
        if power==0:
            mod=''
        #c_fmt = '%1.0e'
        #norm_const = m_e*(v_te**3)*n_04
        title = r'$' + var[0] + '_' + var[1] + r'$ [ ' + mod + '$q_0$ ]'# + r' [ $Jms^{-1}$ ]'

    elif var[0] =='j':
        c_fmt = '%1.1f'
        power = extract_power(sample)
        norm_const = 1.0*(10**-power)
        mod = r'$ 10^{' +str(power)  + '} $'
        if power==0:
            mod=''
        #norm_const = 1.0#q_e*n0*v_te
        
        title = r'$' + var[0] + '_' + var[1] + r'$ [ ' +mod + '$j_0$ ]'#r' [ $C/m^2 s$ ]'    
    elif var =='U':
        
        c_fmt = '%1.1f'
        power = extract_power(sample)
        norm_const = 1.0*(10**-power)
        mod = r'$ 10^{' +str(power)  + '} $'
        #c_fmt = '%1.0e'
        #norm_const = m_e*(v_te**3)*n_04
        title = r'$' + var[0] + r'$ [ ' + mod + '$m_e v_n^2 n_0$ ]'# + r' [ $Jms^{-1}$ ]'

    return norm_const, title, c_fmt
#-----------------------------------------------------------------------
def calc_vosc2_overI(ref_dict):
    '''
        This function only uses norms - it is a function of purely the laser wavelength
    '''
    
    vte = ref_dict['vte']
    Te_ref = ref_dict['Te']
    ne_ref_cm3 = ref_dict['ne']
    alpha_l = 1.0 # 0.5 for linera 1.0 for circular polarisation
    lambda_mu = 0.35 # wavelength in micrometer
    
    vte_cm = vte*100.0# cm/s
    Te_keV = Te_ref/1000.0
    #ne_ref = 1.0e21# cm^-3
    vosc2_I = (0.093/alpha_l)*(lambda_mu**2)*((ne_ref_cm3*vte_cm**3)/(1e15))*(Te_keV**-1)
    return vosc2_I
#------------------------------------------------------------------------------

def get_tau_ei_norm(ne,Te):
    '''
        Work out the collision time in reference collision times of cell with numberdensity ne and temp Te (normlised units)
    '''
    vte_norm = (2.0*Te)**0.5
    tau_ei_norm = (vte_norm**3)/ne
    # tau_ei_norm_SI = tau_ei_norm*tau_ei_n
    return tau_ei_norm
#------------------------------------------------------------------------------

def calc_abs(ne,Te,vgroup,ref_dict):
    '''
        Calculates the ray absorption coefficient
    '''
    vte = ref_dict['vte']
    vosc2_I = calc_vosc2_overI(ref_dict)
    ##vgroup = c*(1.0 - ne/n_crit_norm)**0.5
    v_laser_norm = vgroup/vte
    vte_norm = (2.0*Te)**0.5
    f0_v0_norm = ne/((np.pi**1.5)*(vte_norm**3))
    
    tau_ei_norm = get_tau_ei_norm(ne,Te) # collision time in units of tau_ref
    kappa_IB = (v_laser_norm*(1.5*(8.0*np.pi/9.0))*(0.5*m_e*vosc2_I)*f0_v0_norm*(vte_norm**3/tau_ei_norm))
    
    return kappa_IB