import re, glob
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import scipy as sp
import sympy as spy
import mpmath
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Ellipse
from matplotlib.transforms import Affine2D
plt.style.use(['science', 'notebook', 'grid'])
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})


class Coherence:

    coherences = ['Gaussian', 'Lorentzian', 'GaussLorentz', 'BlackBody']

    def gaussian(OptPathDiff, coh_length, wavelength = 0.633):
        func = np.exp(-(np.pi/2)*(OptPathDiff/coh_length)**2 - (2j*OptPathDiff*np.pi/wavelength))
        return func
    
    def lorentzian(OptPathDiff, coh_length, wavelength = 0.633):
        func = np.exp(-np.abs(OptPathDiff)/coh_length - (2j*OptPathDiff*np.pi/wavelength))
        return func
    
    def gauss_lorentz(OptPathDiff, coh_length, wavelength = 0.633):
        coh_lenght_G, coh_lenght_L = coh_length
        func = np.exp(-(np.pi/2)*(OptPathDiff/coh_lenght_G)**2 -np.abs(OptPathDiff)/coh_lenght_L 
                      - (2j*OptPathDiff*np.pi/wavelength))
        return func
    
    def blackbody(OptPathDiff, wavelength):
        func = 90/np.pi**4 * mpmath.zeta(4, 1 + 1j*1.16e-19*OptPathDiff/wavelength)
        return func
    
    def __init__(self, coherence_func = 'Gaussian', **kwargs):
        self.coherence_func = coherence_func
        if self.coherence_func not in self.coherences:
            raise ValueError(f'Coherence function not valid, please use one in {self.coherences}')
        self.wavelength = kwargs.get("wavelength")
        self.temperature = kwargs.get("temperature")
        self.kwargs = kwargs
          
    def eval(self, OptPathDiff):
        if self.coherence_func == 'Gaussian':
            coh_lenght = self.kwargs.get('coh_length')
            self.coherence_length = coh_lenght
            self._coherence = Coherence.gaussian(OptPathDiff, self.coherence_length, self.wavelength)
            return self._coherence
        
        elif self.coherence_func == 'Lorentzian':
            coh_lenght = self.kwargs.get('coh_length')
            self.coherence_length = coh_lenght
            self._coherence = Coherence.lorentzian(OptPathDiff, self.coherence_length, self.wavelength)
            return self._coherence
        
        elif self.coherence_func == 'GaussLorentz':
            coh_lenght_G, coh_lenght_L = self.kwargs.get('coh_length', ())
            self.coherence_length = (coh_lenght_G, coh_lenght_L)
            self._coherence = Coherence.gauss_lorentz(OptPathDiff, self.coherence_length, self.wavelength)
            return self._coherence
            
        elif self.coherence_func == 'BlackBody':
            self._coherence = Coherence.blackbody(OptPathDiff, self.wavelength)
            return self._coherence
        
class State(spy.Matrix):

    def pauli_matrices():
        
        sigma0 = spy.Matrix([
                        [1, 0],
                        [0, 1]
                        ])
        
        sigma1 = spy.Matrix([
                        [1, 0],
                        [0, -1]
                        ])
        
        sigma2 = spy.Matrix([
                        [0, 1],
                        [1, 0]
                        ])
        
        sigma3 = spy.Matrix([
                        [0, -1j],
                        [1j, 0]
                        ])
        
        return [sigma0, sigma1, sigma2, sigma3]
    
    def basis_change(alpha_0, chi_0):
        
        alpha = np.deg2rad(alpha_0)
        chi = np.deg2rad(chi_0)
        T11 = np.cos(alpha)*np.cos(chi) - 1j*np.sin(alpha)*np.sin(chi)
        T12 = np.cos(alpha)*np.sin(chi) + 1j*np.sin(alpha)*np.cos(chi)
        T21 = np.sin(alpha)*np.cos(chi) + 1j*np.cos(alpha)*np.sin(chi)
        T22 = np.sin(alpha)*np.sin(chi) - 1j*np.cos(alpha)*np.cos(chi)

        matrix = spy.Matrix([[T11, T12],
                           [T21, T22]])

        return spy.simplify(matrix)
    
    def __new__(cls, alpha = 0, chi = 0, basis = (0, 0)):
        cls.alpha = np.deg2rad(alpha)
        cls.chi = np.deg2rad(chi)
        cls.basis = basis
        cls.Matrix_Basis_Change = State.basis_change(basis[0], basis[1])
        cls._state = spy.Matrix([
                            np.cos(cls.alpha)*np.cos(cls.chi) - 1j*np.sin(cls.alpha)*np.sin(cls.chi),
                            np.sin(cls.alpha)*np.cos(cls.chi) + 1j*np.cos(cls.alpha)*np.sin(cls.chi)
                            ])
        
        if basis == (0, 0) or basis == [0, 0]:
            cls.E1 = spy.N(spy.nsimplify(spy.simplify(cls._state[0]), rational=True, tolerance=1e-3), 4)
            cls.E2 = spy.N(spy.nsimplify(spy.simplify(cls._state[1]), rational=True, tolerance=1e-3), 4) 
        
            return super(State, cls).__new__(cls, [
                [spy.nsimplify(spy.simplify(cls.E1*np.conjugate(cls.E1)), rational=True, tolerance=1e-3),
                spy.nsimplify(spy.simplify(cls.E1*np.conjugate(cls.E2)), rational=True, tolerance=1e-3)],
                [spy.nsimplify(spy.simplify(cls.E2*np.conjugate(cls.E1)), rational=True, tolerance=1e-3), 
                spy.nsimplify(spy.simplify(cls.E2*np.conjugate(cls.E2)), rational=True, tolerance=1e-3)]
                ])
        
        else:
            cls.state = cls.Matrix_Basis_Change.inv() * cls._state
            cls.E1 = spy.N(spy.nsimplify(spy.simplify(cls.state[0]), rational=True, tolerance=1e-3), 4)
            cls.E2 = spy.N(spy.nsimplify(spy.simplify(cls.state[1]), rational=True, tolerance=1e-3), 4) 
        
            return super(State, cls).__new__(cls, [
                [spy.nsimplify(spy.simplify(cls.E1*np.conjugate(cls.E1)), rational=True, tolerance=1e-3),
                spy.nsimplify(spy.simplify(cls.E1*np.conjugate(cls.E2)), rational=True, tolerance=1e-3)],
                [spy.nsimplify(spy.simplify(cls.E2*np.conjugate(cls.E1)), rational=True, tolerance=1e-3), 
                spy.nsimplify(spy.simplify(cls.E2*np.conjugate(cls.E2)), rational=True, tolerance=1e-3)]
                ])
    
    def __init__(self, alpha, chi, basis = (0, 0)):
        self.alpha = alpha
        self.chi = chi
        self.basis = basis
        self.dop = 1
        self.Matrix_Basis_Change = State.basis_change(basis[0], basis[1])

    @property
    def stokes(cls):
        if cls.basis == (0, 0) or cls.basis == [0, 0]:
            pauli_vector = State.pauli_matrices()
        else: 
            pauli_vector = cls.Matrix_Basis_Change.inv() * State.pauli_matrices() * cls.Matrix_Basis_Change

        Stokes = [float(spy.N(spy.nsimplify(spy.trace(sigma*cls), tolerance = 1e-3), 3)) for sigma in pauli_vector]
        return Stokes 
    
    @property
    def s1(cls):
        return cls.stokes[1]
    
    @property
    def s2(cls):
        return cls.stokes[2]
    
    @property
    def s3(cls):
        return cls.stokes[3]

    def ellipse(self):
        '''
        This function returns the graphic of the polarization ellipse asociated with the Polarization State
        '''
        fig, ax = plt.subplots(subplot_kw={"aspect": "equal"})

        horizontal_axis = 1 / np.sqrt(np.tan( np.deg2rad(self.chi) )**2 + 1)
        vertical_axis = np.tan(np.deg2rad(self.chi)) / np.sqrt(np.tan(np.deg2rad(self.chi))**2 + 1)

        ellipse = Ellipse(
            xy = (0, 0),
            width = horizontal_axis,
            height = vertical_axis,
            angle = self.alpha,
            facecolor="none",
            edgecolor="b"
            )
        
        ax.add_patch(ellipse)

        # Plot an arrow marker at the end point of minor axis
        vertices = ellipse.get_co_vertices()
        t = Affine2D().rotate_deg(ellipse.angle)

        ax.plot(
            vertices[0][0],
            vertices[0][1],
            color="b",
            marker=MarkerStyle(">", "full", t),
            markersize=10
                )
        #Note: To reverse the orientation arrow, switch the marker type from > to <.
        plt.xlim([-0.5,0.5])
        plt.ylim([-0.5,0.5])

        plt.show()

    def Poincare_sphere(cls):
        Graphic([cls])
    
    def operate(cls, operators, coherence):
        transformed_states = [spy.expand(operator * cls * operator.adjoint()).rewrite(spy.cos).expand() for operator in operators]
        operated_state = []
        for operator, state in zip(operators, transformed_states):
            list_of_OPD, list_of_sines, list_of_cosines = operator.list_of_phases()
            for OPD, sine, cosine in zip(list_of_OPD, list_of_sines, list_of_cosines):
                state = state.subs(sine, np.imag(coherence.eval(OPD)))
                state = state.subs(cosine, np.real(coherence.eval(OPD)))
            J11 = state[0]
            J12 = state[1]
            J21 = state[2]
            J22 = state[3]
            operated_state.append(Partial_State(J11, J12, J21, J22))
        return operated_state

class Partial_State(spy.Matrix):
    
    def __new__(cls, J11, J12, J21, J22, basis=(0,0)):
        return super(Partial_State, cls).__new__(cls, [
            [J11, J12],
            [J21, J22]
        ])

    def __init__(self, J11, J12, J21, J22, basis=(0,0)):
        self.J11 = J11
        self.J12 = J12
        self.J12 = J21
        self.J22 = J22
        self.base = basis
        
    @property
    def dop(cls):
        return float(spy.sqrt(1-4*cls.det()))
    
    @property
    def stokes(cls):
        pauli_vector = State.pauli_matrices()
        Stokes = [spy.N(spy.nsimplify(spy.re(spy.trace(sigma*cls)), tolerance = 1e-3), 3) for sigma in pauli_vector]
        return Stokes 
    
    @property
    def s1(cls):
        return cls.stokes[1]
    
    @property
    def s2(cls):
        return cls.stokes[2]
    
    @property
    def s3(cls):
        return cls.stokes[3]

    def Poincare_sphere(cls):
        Graphic([cls])

class Rotation(spy.Matrix):

    def __new__(cls, angle, basis = (0,0)):
        cls.angle = np.deg2rad(angle)
        cls.Basis = basis
        if basis == (0,0) or basis == [0,0]:
            cls._rotation = spy.Matrix([
                [spy.cos(cls.angle), spy.sin(cls.angle)],
                [-spy.sin(cls.angle), spy.cos(cls.angle)]
                ])
            return spy.nsimplify(spy.simplify(cls._rotation), tolerance=0.001)
        else:
            cls.transformation_matrix = State.basis_change(cls.Basis[0], cls.Basis[1])
            cls.rotation_matrix = cls.transformation_matrix.inv() * cls._rotation * cls.transformation_matrix

            return spy.nsimplify(spy.simplify(cls.rotation_matrix), tolerance=0.001)
    
class Waveplate(spy.Matrix):

    def __new__(cls, OptPathDiff, angle = 0, eigenstate = (0,0), basis = (0,0)):
        
        cls.OptPathDiff = OptPathDiff
        cls.angle = np.deg2rad(angle)
        cls.basis = basis
    
        cls.operator = spy.exp(spy.I*cls.OptPathDiff)
        cls._retarder = spy.Matrix([
                                [cls.operator, 0],
                                [0, 1]
                                   ])
        
        #if the basis is linear
        if basis == (0,0) or basis == [0,0]:
            #change of base matrix from eigenstate to linear
            cls.change_basis_eigenstate = State.basis_change(eigenstate[0], eigenstate[1])

            #from eigenstate basis to selected basis
            cls.retarder = cls.change_basis_eigenstate * cls._retarder * cls.change_basis_eigenstate.inv() 
            cls.retarder = Rotation(angle).inv() * cls.retarder * Rotation(angle)
            cls.retarder = spy.nsimplify(spy.simplify(cls.retarder), rational= True, tolerance= 1e-3)

            return super(Waveplate, cls).__new__(cls, [
            [cls.retarder[0], cls.retarder[1]],
            [cls.retarder[2], cls.retarder[3]]
            ])
        
        #if the basis is another
        else:
            #change of base matrix from eigenstate to linear
            cls.change_basis_eigenstate = State.basis_change(eigenstate[0], eigenstate[1])
            #change of base matrix from selected basis to linear
            cls.matrix_basis_change = State.basis_change(basis[0], basis[1])

            cls.retarder = cls.change_basis_eigenstate * cls._retarder * cls.change_basis_eigenstate.inv()
            cls.retarder = cls.matrix_basis_change.inv() * cls.retarder * cls.change_basis_eigenstate
            cls.retarder = Rotation(angle, basis).inv() * cls.retarder * Rotation(angle, basis)
            cls.retarder = spy.nsimplify(spy.simplify(cls.retarder), rational= True, tolerance= 1e-3)

            return super(Waveplate, cls).__new__(cls, [
                [cls.retarder[0], cls.retarder[1]],
                [cls.retarder[2], cls.retarder[3]]
                ])
        
    def __init__(self, OptPathDiff, angle = 0, eigenstate = (0,0), basis = (0,0)):
        self.OptPathDiff = OptPathDiff
        self.angle = np.deg2rad(angle)
        self.basis = basis
        self.eigenstate = eigenstate 
    
    def rotate(self, angle):
        return Waveplate(self.OptPathDiff, np.rad2deg(self.angle) + angle, self.eigenstate, self.basis)
        
        
class Composite_waveplate(spy.Matrix):
    
    def _subsetSums(nums):
        # There are total 2^n subsets
        s = [0]
        for i in range(len(nums)):
            v = len(s)
            for t in range(v):
                s.append(s[t] + nums[i]) # add this element with previous subsets
                s.append(-s[t] + nums[i]) # substract this element with previous subsets
        del s[0]
        
        return sorted(list(set(np.abs(s))))
    
    def __new__(cls, Waveplates):
        cls.waveplates = Waveplates
        cls.operator = spy.expand(spy.prod(cls.waveplates[::-1]))
        return super(Composite_waveplate, cls).__new__(cls, [
            [cls.operator[0], cls.operator[1]],
            [cls.operator[2], cls.operator[3]]
            ])
    
    def __init__(self, Waveplates):
        self.waveplates = Waveplates
        super().__init__()
        
    def rotate(self, angle):
        list_of_wp = [wp.rotate(angle) for wp in self.waveplates]
        return Composite_waveplate(list_of_wp)

    def list_of_phases(self):
        OptPathDiff_list = [wp.OptPathDiff for wp in self.waveplates]  
        list_of_OPD = Composite_waveplate._subsetSums(OptPathDiff_list)
        list_of_sines = [spy.sin(OPD) for OPD in list_of_OPD]
        list_of_cosines = [spy.cos(OPD) for OPD in list_of_OPD]
        return list_of_OPD, list_of_sines, list_of_cosines

    def operate(cls, states, coherence):
        transformed_states = [spy.expand(cls.operator * state * cls.operator.adjoint()).rewrite(spy.cos).expand() for state in states]
        list_of_OPD, list_of_sines, list_of_cosines= cls.list_of_phases()
        operated_state = []
        for state in transformed_states:
            for OPD, sine, cosine in zip(list_of_OPD, list_of_sines, list_of_cosines):
                state = state.subs(sine, np.imag(coherence.eval(OPD)))
                state = state.subs(cosine, np.real(coherence.eval(OPD)))
            J11 = state[0]
            J12 = state[1]
            J21 = state[2]
            J22 = state[3]
            operated_state.append(Partial_State(J11, J12, J21, J22))
        return operated_state

def Graphic(State: list):
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
    s1 = [state.s1 / state.dop for state in State]
    s2 = [state.s2 / state.dop for state in State]
    s3 = [state.s3 / state.dop for state in State]

    ax.scatter(s1, s2, s3, color='red', alpha = 1, s=30)

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

class polarimeter_data:
    pass

