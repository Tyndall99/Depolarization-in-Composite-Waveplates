"""
Code related with the work of depolaization of light through composite waveplates

author: Cristian Hernández Cely

"""
import pandas as pd
import numpy as np
import re, glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap
from rpy2.robjects.packages import importr
import scienceplots
from matplotlib import cm
from scipy.optimize import curve_fit
plt.style.use(['science', 'notebook', 'grid'])
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

class Depolarization:
    
    def __init__(self,**kwargs):
        self.path = kwargs.get("path", [])
        self.data = []
        self.initial_state = []
        self.materials = []
        self.data_fit = []

    def add_linear_wp(self, theta_values, delta_values):
        """
        Parameters
        ----------
        theta_values : THE ORIENTATION OF THE LINEAR WP WITH RESPECT THE HORIZONTAL
        delta_values : THE TOTAL PHASE-SHIFT INTRODUCED BY THE LINEAR WP 

        Returns
        -------
        APPENDS TO THE self.materials LIST, THE PARAMETERS OF THE LINEAR WAVEPLATES
        """    
        materials = [theta_values, delta_values]

        self.materials.append(materials)
        

    def read_data(self):
        """
        This function reads the polarimeter data and extract its mean values 
        
        """
        for path in self.path:
            #leer todos los archivos .csv de la carpeta
            all_files = glob.glob(path + '*.csv') 
        
            #ordenar como windows
            all_files.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)',
                                                    x)]) 
        
            data = np.array([pd.read_csv(filename, engine='python', #to read haders between quotation marks like "Header"
                              index_col=False, header = 23, #The first data column is not longer use as colum index
                                          usecols = [1,2,3,4,5,8], 
                                          encoding='latin1').mean(axis=None) for filename in all_files]).T
        
            self.data.append(data)
    
    # COHERENCE FUNCTION

    def coherence(self,x, l_g , l_l, wavelength = 0.633):
        """
        Parameters
        ----------
        l_l :  LORENTZIAN COHERENCE LENGHT.
        l_g :  GAUSSIAN COHERENCE LENGHT.
        x : VARIABLE.
        wavelenght: MEAN WAVELENGTH OF THE LIGHT SOURCE

        Returns
        -------
        gamma_1 : THE LORENTZIAN-GAUSSIAN COHERENCE FUNCTION
        
        """

        #gamma_1 = np.exp(-(np.pi/2)*(x/l_g)**2 - (np.abs(x)/l_l) - (2j*x*np.pi/wavelenght)) 
        gamma_1 = np.exp(-(np.pi/2)*(x/l_g)**2 - (2j*x*np.pi/wavelength))
        
        return gamma_1
    
    
    #Operators

    def T_elements(self, Matrix):
        return np.array([i.T for i in Matrix])

    def Ret_matrix(self, OptPathDiff, l_c=[], wavelength=0.633 ):
        
        Matrix = np.array([[self.coherence(OptPathDiff, l_c[0], l_c[1], wavelength), 0],
                     [0,1]])
    
        if Matrix.size > 4:
                return self.T_elements(Matrix.T)
        else:
                return Matrix
    
    def Pol_matrix(self, init_state = []):
        
        alpha = init_state[0]
        chi = init_state[1]
        
        E_x = np.cos(alpha)*np.cos(chi) - 1j*np.sin(alpha)*np.sin(chi)
        E_y = np.sin(alpha)*np.cos(chi) + 1j*np.cos(alpha)*np.sin(chi)
        
        Matrix = np.array([[E_x*np.conjugate(E_x) , E_x*np.conjugate(E_y)],
                     [E_y*np.conjugate(E_x) , E_y*np.conjugate(E_y)]]) 
        
        if Matrix.size > 4:
                return self.T_elements(Matrix.T)
        else:
                return Matrix
        
    def Rot_matrix(self, angle):

        Matrix = np.array([[np.cos(angle) , np.sin(angle)],
                     [-np.sin(angle) , np.cos(angle)]]) 
                     
        if Matrix.size > 4:
                return self.T_elements(Matrix.T)
        else:
                return Matrix


    #METHOD TO BUILD CW
        
    def CW(self, angle ,OptPathDiff = [], Angles = [], init_state = [], l_c=[], wavelength=0.633):
    
        aux = np.array([[1,0],[0,1]])
        for i in range(OptPathDiff.size):
            R = self.Rot_matrix(angle + Angles[i])
            Ret = self.Ret_matrix(OptPathDiff[i], l_c, wavelength)
            aux = self.T_elements(R) @ Ret @ R @ aux

        J = self.Pol_matrix(init_state)
        return aux @ J @ self.T_elements(aux.conjugate())
    
    #PARTICULAR CASES OF WAVEPLATES WITH A LINEAR BASIS
    
    def WP_simple(self, OptPathDiff, l_c = [], init_state = []):
        
        """
        Parameters
        ----------
        OptPathDiff: OPTICAL PATH DIFFERENCE INTRODUCE BY THE RETARDER
        l_c : A list of the form [l_g, l_l]
            l_g :  GAUSSIAN COHERENCE LENGHT.
            l_l : LORENTZIAN COHERENCE LENGHT.
        init_state : A list of the form [alpha, chi]
            alpha = ORIENTATION ANGLE OF THE INCIDENT POLARIZATION STATE.
            CHI = ELIPTICITY ANGLE OF THE INCIDENT POLARIZATION STATE.

        Returns
        -------
        DOP, S1, S2, S3
        
        DOP : THE DEGREE OF POLARIZATION
        [S1, S2, S3] : THE TRANSFORMED STOKES PARAMETERS

        """
        
        alpha = init_state[0]
        chi = init_state[1]
        
        E_x = np.cos(alpha)*np.cos(chi) - 1j*np.sin(alpha)*np.sin(chi)
        E_y = np.sin(alpha)*np.cos(chi) + 1j*np.cos(alpha)*np.sin(chi)
        
        A = E_x*np.conjugate(E_y)
        
        J11 = E_x*np.conjugate(E_x)
        
        J22 = 1 - J11
        
        J12 = A*self.coherence(OptPathDiff, l_c[0], l_c[1])
        
        J21 = np.conjugate(J12)
        
        DOP = np.sqrt((1 - 4*(J11 * J22 - J21*J12) / (J11 + J22)**2))
        
        return np.real(DOP), np.real(J11 - J22), np.real(J12 + J21), np.real(1j*(J12 - J21))
    
    def WP_circ(self, OptPathDiff, l_c = [], init_state = []):
        
        """
        Parameters
        ----------
        OptPathDiff: OPTICAL PATH DIFFERENCE INTRODUCE BY THE RETARDER
        l_c : A list of the form [l_g, l_l]
            l_g :  GAUSSIAN COHERENCE LENGHT.
            l_l : LORENTZIAN COHERENCE LENGHT.
        init_state : A list of the form [alpha, chi]
            alpha = ORIENTATION ANGLE OF THE INCIDENT POLARIZATION STATE.
            CHI = ELIPTICITY ANGLE OF THE INCIDENT POLARIZATION STATE.

        Returns
        -------
        DOP, S1, S2, S3
        
        DOP : THE DEGREE OF POLARIZATION
        [S1, S2, S3] : THE TRANSFORMED STOKES PARAMETERS

        """
        
        alpha = init_state[0]
        chi = init_state[1]
        
        E_x = np.cos(alpha)*np.cos(chi) - 1j*np.sin(alpha)*np.sin(chi)
        E_y = np.sin(alpha)*np.cos(chi) + 1j*np.cos(alpha)*np.sin(chi)
        
        #E_x = (1/np.sqrt(2))*(np.cos(chi) - np.sin(chi))*np.exp(1j*alpha)
        #E_y = (1/np.sqrt(2))*(np.cos(chi) + np.sin(chi))*np.exp(-1j*alpha)
        
        #Coherence functions
        Real = np.real( self.coherence(OptPathDiff, l_c[0], l_c[1]) )
        Imag = np.imag( self.coherence(OptPathDiff, l_c[0], l_c[1]) )
        
        A = E_y*np.conjugate(E_y) - E_x*np.conjugate(E_x)
        A1 = E_y*np.conjugate(E_y) + E_x*np.conjugate(E_x)
        B = E_x*np.conjugate(E_y) - E_y*np.conjugate(E_x)
        B1 = E_x*np.conjugate(E_y) + E_y*np.conjugate(E_x)
        
        #J11 = -1/4*(A*2*Real + B*2j*Imag - 2*A1)
        J11 = -(1/4)*(A*2*Real + B1*2*Imag - 2*A1)
        J22 = 1 - J11
        
        #J12 =  1/4*(A*2j*Imag + B*2*Real + 2*B1)
        J12 = (1/4)*(-A*2*Imag + B1*2*Real + 2*B)
        J21 = np.conjugate(J12)
        
        DOP = np.sqrt((1 - 4*(J11 * J22 - J21*J12) / (J11 + J22)**2))
        
        return np.real(DOP), np.real(J11 - J22), np.real(J12 + J21), np.real(1j*(J12 - J21))
    
    def WP(self, angle, OptPathDiff, l_c = [], init_state = []):
        
        """
        Parameters
        ----------
        angle = ROTATION ANGLE OF THE WAVEPLATE
        OptPathDiff: OPTICAL PATH DIFFERENCE INTRODUCE BY THE RETARDER
        l_c : A list of the form [l_g, l_l]
            l_g = GAUSSIAN COHERENCE LENGHT.
            l_l = LORENTZIAN COHERENCE LENGHT.
        init_state : A list of the form [alpha, chi]
            alpha = ORIENTATION ANGLE OF THE INCIDENT POLARIZATION STATE.
            CHI = ELIPTICITY ANGLE OF THE INCIDENT POLARIZATION STATE.
            
        Returns
        -------
        DOP, S1, S2, S3
        
        DOP : THE DEGREE OF POLARIZATION
        [S1, S2, S3] : THE TRANSFORMED STOKES PARAMETERS

        """
        
        alpha = init_state[0]
        chi = init_state[1]
        
        E_x = np.cos(alpha)*np.cos(chi) - 1j*np.sin(alpha)*np.sin(chi)
        E_y = np.sin(alpha)*np.cos(chi) + 1j*np.cos(alpha)*np.sin(chi)

        #Some constants
        A = (E_x*np.conjugate(E_y)) + (np.conjugate(E_x)*E_y)
        A1 = (E_x*np.conjugate(E_y)) - (np.conjugate(E_x)*E_y)
        B = E_x*np.conjugate(E_x) - E_y*np.conjugate(E_y)
        
        #Coherence functions
        Real = np.real( self.coherence(OptPathDiff, l_c[0], l_c[1]) )
        Imag = np.imag( self.coherence(OptPathDiff, l_c[0], l_c[1]) )
        
        J11_1 =  A * (2 * Real - 2  ) * np.sin(4*u)
        J11_2 =  B * (2 * Real - 2 ) * np.cos(4*u)
        J11_3 =  A1 * ( 4j * Imag ) * np.sin(2*u)
        J11_4 = -B * ( 2  * Real )  - 2*E_y*np.conjugate(E_y) - 6*E_x*np.conjugate(E_x)
        
        J11 = -(J11_1 + J11_2 + J11_3 + J11_4)/8

        
        J22 = 1 - J11

        J21_1 = - B * (2 * Real  - 2 ) * np.sin(4*u)
        J21_2 =  A *( 2 * Real - 2 ) * np.cos(4*u)
        J21_3 = -B * ( 4j*Imag ) * np.sin(2*u)
        J21_4 =  (E_x*np.conjugate(E_y)) *( 8j * Imag ) * np.cos(2*u)
        J21_5 =   (3*(E_x*np.conjugate(E_y)) - (np.conjugate(E_x)*E_y)) * 2 * Real + 2*A 
        
        J12 = (J21_1 + J21_2 + J21_3 + J21_4 + J21_5)/8
        
        J21 = np.conjugate(J12)
        
        DOP = np.sqrt((1 - 4*(J11 * J22 - J21*J12) / (J11 + J22)**2))
        
        return np.real(DOP), np.real(J11 - J22), np.real(J12 + J21), np.real(1j*(J12 - J21))
    
    def CW1(self, total_angle, angle_second ,OptPathDiff, l_c = [], init_state = []):
        
        """
        Parameters
        ----------
        total_angle : ROTATION ANGLE OF THE WAVEPLATE
        angle_second : ANGLE OF THE SECOND WAVEPLATE WITH RESPECT TO THE FIRST ONE
        OptPathDiff : OPTICAL PATH DIFFERENCE INTRODUCE BY THE RETARDER
        l_c : A list of the form [l_g, l_l]
            l_g = GAUSSIAN COHERENCE LENGHT.
            l_l = LORENTZIAN COHERENCE LENGHT.
        init_state : A list of the form [alpha, chi]
            alpha = ORIENTATION ANGLE OF THE INCIDENT POLARIZATION STATE.
            chi = ELIPTICITY ANGLE OF THE INCIDENT POLARIZATION STATE.
            
        Returns
        -------
        DOP, S1, S2, S3
        
        DOP : THE DEGREE OF POLARIZATION
        [S1, S2, S3] : THE TRANSFORMED STOKES PARAMETERS

        """
        
        alpha = init_state[0]
        chi = init_state[1]
        u = total_angle
        phi = angle_second
        
        E_x = np.cos(alpha)*np.cos(chi) - 1j*np.sin(alpha)*np.sin(chi)
        E_y = np.sin(alpha)*np.cos(chi) + 1j*np.cos(alpha)*np.sin(chi)
        
        #E_x = (1/np.sqrt(2))*(np.cos(chi) - np.sin(chi))*np.exp(1j*alpha)
        #E_y = (1/np.sqrt(2))*(np.cos(chi) + np.sin(chi))*np.exp(-1j*alpha)
        
        #Some constants
        A = (E_x*np.conjugate(E_y)) + (np.conjugate(E_x)*E_y)
        A1 = (E_x*np.conjugate(E_y)) - (np.conjugate(E_x)*E_y)
        B = E_x*np.conjugate(E_x) - E_y*np.conjugate(E_y)
        Jxy = E_x*np.conjugate(E_y)
        Jyx = np.conjugate(E_x)*E_y
        
        #Coherence functions
        Real1 = np.real(self.coherence(OptPathDiff, l_c[0], l_c[1]))
        Imag1 = np.imag(self.coherence(OptPathDiff, l_c[0], l_c[1]))
        Real2 = np.real(self.coherence(2*OptPathDiff, l_c[0], l_c[1]))
        Imag2 = np.imag(self.coherence(2*OptPathDiff, l_c[0], l_c[1]))
        
        # Elements of the final polarization matrix
        
        #Element J11 and J22
        J11_1 = ( 2*Real2 - 2 ) * ( A * ( np.sin(4*u + 4*phi) + np.sin(4*u) +  2*np.sin(4*u + 2*phi) ) + B * ( np.cos(4*u + 4*phi) + np.cos(4*u) +  2*np.cos(4*u + 2*phi) ) )
        J11_2 = ( 4j*Imag2 + 8j*Imag1 ) * ( A1 * ( np.sin(2*u + 2*phi) ) )
        J11_3 = 8j*Imag2 * A1 * np.sin(2*u) + ( 4j*Imag2 - 8j*Imag1 ) * ( A1* np.sin(2*u - 2*phi) ) 
        J11_4 = 2*Real2 * ( A * ( -np.sin(4*phi) - 2*np.sin(2*phi) ) + B * (-np.cos(4*phi) - 2*np.cos(2*phi) - 1 ) )
        J11_5 = 2*Real1 * ( A *4*np.sin(4*phi) + B * (4*np.cos(4*phi) - 4 ) )
        J11_6 = A*( -6*np.sin(4*phi) + 4*np.sin(2*phi) ) + B*( -6*np.cos(4*phi) + 4*np.cos(2*phi) ) - 10*E_y*np.conjugate(E_y) - 22*E_x*np.conjugate(E_x)
        
        J11 = -( J11_1 + J11_2 + J11_3 + J11_4 + J11_5 + J11_6 ) / 32
        
        
        J22 = 1 - J11
        
        #Element J12 and J21
        
        J12_1 = ( 2*Real2 - 2 ) * ( B*( -np.sin(4*u + 4*phi) - np.sin(4*u) -  2*np.sin(4*u + 2*phi) ) + A*( np.cos(4*u + 4*phi) + np.cos(4*u) +  2*np.cos(4*u + 2*phi) ) )
        J12_2 = ( 4j*Imag2 - 8j*Imag1 ) * ( B*( - np.sin(2*u + 4*phi)) + A*np.cos(2*u + 4*phi) + A1*np.cos(2*u - 2*phi) )
        J12_3 = (2j*Imag2*(6*Jxy + 2*Jyx) + 8j*Imag1*A1)*np.cos(2*u + 2*phi) - ( 4j*Imag2 + 8j*Imag1 )*B*np.sin(2*u)
        J12_4 = (2j*Imag2*(6*Jxy - 2*Jyx) + 8j*Imag1*A)*np.cos(2*u) - 8j*Imag2*B*np.sin(2*u + 2*phi)
        J12_5 = 2*Real2 * ( B*(-np.sin(4*phi) - 2*np.sin(2*phi)) + A*np.cos(4*phi) + (6*Jxy - 2*Jyx)*np.cos(2*phi) + 5*Jxy - 3*Jyx )
        J12_6 = 2*Real1 * ( 4*B*np.sin(4*phi) + A*(-4*np.cos(4*phi) + 4 ) )
        J12_7 = B*(-6*np.sin(4*phi) + 4*np.sin(2*phi)) + A*6*np.cos(4*phi) - (12*Jxy - 4*Jyx)*np.cos(2*phi) + 14*Jxy - 2*Jyx
        
        J12 = (J12_1 + J12_2 + J12_3 + J12_4 + J12_5 + J12_6 + J12_7)/32
        
        J21 = np.conjugate(J12)
        
        
        #calculate the DOP
        DOP = np.sqrt(1 - (4*(J11 * J22 - J12*J21)/(J11 + J22)**2))
        
        return np.real(DOP), J11 - J22, J12 + J21, 1j*(J12 - J21)
        #return np.real(DOP)
    
    # GENERAL CASE OF A COMPOSITE WAVEPLATE
    def CW(angle ,OptPathDiff = [], Angles = [], init_state = [], l_c=[], wavelength=0.633):
    
        aux = np.array([[1,0],[0,1]])
        for i in range(OptPathDiff.size):
            R = Rot_matrix(angle + Angles[i])
            Ret = Ret_matrix(OptPathDiff[i], l_c, wavelength)
            aux = T_elements(R) @ Ret @ R @ aux

        J = Pol_matrix(init_state)
        Gamma = aux @ J @ T_elements(aux.conjugate())

        J11, J12, J21, J22 = Gamma[:,0,0], Gamma[:,0,1], Gamma[:,1,0], Gamma[:,1,1]

        s0, s1, s2, s3 = np.real(J11 + J22), np.real(J11 - J22), np.real(J12 + J21), np.real(1j*(J12 - J21))

        return s0, s1, s2, s3

    # GRAPHIC METHODS
    def graph_fit_DOP(self,i):
        
        
        xdata = np.linspace(0, np.pi, 46)
        
        ydata = self.data[i][5]/100
        
        popt, pcov = curve_fit(self.CW1, xdata, ydata, p0 = [-43*np.pi/180,25, (35+ 1/4)*0.633 ,0,30*np.pi/180] , bounds=((-50*np.pi/180,10,0, 0, 0),(-40*np.pi/180, 35, 60, np.pi, np.pi/4 )) )
        
        x = np.linspace(0, np.pi, 100) 
        
        DOP = self.CW1(x, popt[0], popt[1], popt[2], popt[3], popt[4])
        
        plt.style.use(['science', 'notebook', 'grid'])
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Helvetica"
        })
        
    
        
        plt.plot(x, DOP)
        plt.plot(np.linspace(0,np.pi,46), self.data[i][5]/100, 'o')
        plt.title('Degree of polarization $P$ vs. $\\theta$')
        plt.xlabel('$\\theta$')
        plt.ylabel('Degree of Polarization $P$')
        plt.show()
        print(popt, pcov)
        
        
    def graphic(self, s_1, s_2, s_3):

        u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:30j]
        r = 1
        x = r* np.cos(u)*np.sin(v)
        y =  r* np.sin(u)*np.sin(v)
        z = r* np.cos(v)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
    # Graficar la esfera
        ax.plot_wireframe(x, y, z, rstride=5, cstride=6, color='grey', alpha=0.3,
                      linewidth=1.3)
        ax.plot_surface(x, y, z, rstride=1, cstride=1, color='grey', alpha=0.2,
                    linewidth=0)
    
    #Graficar los ejes
        ax.plot([-1.1, 1.1], [0, 0], [0, 0], color='black', linewidth=1.5, alpha = 0.7)
        ax.plot([0, 0], [-1.1, 1.1], [0, 0], color='black', linewidth=1.5, alpha = 0.7)
        ax.plot([0, 0], [0, 0], [-1.1, 1.1], color='black', linewidth=1.5, alpha = 0.7)
    
    #Graphic some meridians for aesthetic
        theta = np.linspace(0, 2*np.pi, 100)
 
        xx = r * np.cos(theta)
        yy = r * np.sin(theta)
        zz = np.zeros_like(theta)
        ax.plot(xx, yy, zz, color='gray', linewidth=1.5, alpha = 1)
        ax.plot(yy, zz, xx, color='gray', linewidth=1.5, alpha = 1)

    # Graphic the Data on the Sphere
        ax.scatter(s_1, s_2, s_3, color='red', alpha = 1, s=10)

    #Configure the image
        fig.set_size_inches(9, 9)
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_zlim([-1.1, 1.1])
        ax.set_aspect("equal")
        ax.patch.set_alpha(0)
        plt.tight_layout()
        plt.axis('off')
    
    # Add the S_1 S_2 and S_3 names for the main axis of Poincaré Sphere
        ax.text(1.15, 0, 0, '$S_1$', fontsize=18)
        ax.text(0, 1.15, 0, '$S_2$', fontsize=18)
        ax.text(0, 0, 1.15, '$S_3$', fontsize=18)
        
        plt.show()
    
    def graph_exp_data(self, i):
       data = np.array(pd.read_csv(path[i], engine='python', #to read haders between quotation marks like "Header"
                         index_col=False, header = 23, #The first data column is not longer use as colum index
                                     usecols = [0,1,2,3,4,5,8], 
                                     encoding='latin1'))
       t = data[:,0]
       s_1 = data[:,1]
       s_2 = data[:,2]
       s_3 = data[:,3]

       plt.style.use(['science', 'notebook', 'grid'])
       plt.rcParams.update({
           "text.usetex": True,
           "font.family": "Helvetica"
       })
       
       plt.plot(t,s_1)
       plt.plot(t,s_2)
       plt.plot(t,s_3)
       plt.show()
    
    def dop(self):
        
        x, y = np.meshgrid(np.linspace(0,np.pi,100),np.linspace(0,np.pi/4,100))
        
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        #ax.plot_surface(x,y,self.CW1(0,np.pi/8,(14 + 1/2)*0.633,[20,0.1],[x,y])[0], cmap=cm.coolwarm, linewidth=2)
        ax.plot_surface(x,y,self.CW(9, [10,10], [0, np.pi/4], [x,y])[0], cmap=cm.coolwarm, linewidth=2)
        #ax.plot_surface(x,y,self.WP_circ(10.4, [20,10], [x,y])[0], cmap=cm.coolwarm, linewidth=2)

path1 = 'Despolarizacion datos/Articulo despolarizacion/datos/biplaca theta 45/incidente lineal/'
path2 = 'Despolarizacion datos/Articulo despolarizacion/datos/biplaca theta 45/incidente circular/'
path3 = 'Despolarizacion datos/Articulo despolarizacion/datos/biplaca theta 45/incidente eliptico/'
path = [path1, path2, path3]



u = np.linspace(0, np.pi, 100)



#path = [path1, path2, path3, path4]
m = Depolarization(path = path)
m.read_data()
#DOP, S1, S2, S3 = m.WP_circ((40)*0.633,[20,10],[u,1])
#DOP, S1, S2, S3 = m.CW1(0,np.pi/4,(10 + 1/2)*0.633,[20,0.1],[u,1])
#m.graphic(S1/DOP, S2/DOP, S3/DOP)