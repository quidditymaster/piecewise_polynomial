#Author Tim Anderton
#Created Feb 2012

"""A module for representing and fitting piecewise polynomial functions 
with and without regularity constraints.
"""

import numpy as np
lna = np.linalg
poly1d = np.poly1d
import matplotlib.pyplot as plt
#Legendre = np.polynomial.legendre.Legendre

class Centered_Scaled_Polynomial:
    "represents polynomials P(y) in a centered scaled coordinate y = (x-c)/s"
    
    def __init__(self, coefficients, center = 0.0, scale = 1.0):
        self.poly = poly1d(coefficients)
        self.center = center
        self.scale = scale
    
    def __call__(self, xdat):
        return self.poly((xdat-self.center)/self.scale)
    
    def deriv(self):
        new_coeffs = self.poly.deriv().c/self.scale #divide by scale because of chain rule
        return Centered_Scaled_Polynomial(new_coeffs, self.center)

class Polynomial_Basis:
    "a class representing a collection of polynomials"
    
    def __init__(self, poly_coefficients, center = 0.0, scale = 1.0):
        """coefficients: a (n_basis,  poly_order+1) shaped array with the polynomial
        coefficients"""
        self.coeffficients = poly_coefficients.transpose()
        self.n_basis, order_plus_one = poly_coefficients.shape
        self.order = order_plus_one - 1
        self.center = center
        self.scale = scale
        self.basis_polys = []
        for poly_idx in xrange(self.n_basis):
            self.basis_polys.append(Centered_Scaled_Polynomial(self.coefficients[poly_idx], self.center, self.scale))
        
    def evaluate_polynomial(self, basis_coefficients):
        "returns a polynomial as a weighted sum of the basis polynomials"
        output_poly_coeffs = np.dot(self.coefficients, basis_coefficients)
        outpol = Centered_Scaled_Polynomial(output_poly_coeffs, self.center, self.scale)
        return outpol
    
    def evaluate_basis(self, xvals):
        """returns a (self.n_basis, len(xvals)) shaped array 
        containing the polynomials evaluated at the positions in xvals"""
        xvec = np.array(xvals)
        out_basis = np.zeros((self.n_basis, len(xvec)))
        for basis_idx in xrange(self.n_basis):
            out_basis[basis_idx] = self.basis_polys[basis_idx](xvec)
        return out_basis

class Centered_Scaled_Multinomial:
    
    def __init__(self, coefficients, powers, center, scale = 1.0):
        self.coeffs = np.array(coefficients)
        self.powers = np.array(powers, dtype = int)
        self.max_powers = np.max(self.powers, axis = 0)
        self.center = np.array(center)
        self.n_coeffs, self.n_dims = self.powers.shape
        if scale == 1.0:
            self.scale = np.ones(self.n_dims, dtype = float)
        else:
            self.scale = np.array(scale)
    
    def __add__(self, B):
        new_coeffs = []
        new_powers = []
        powers_dict = {}
        for coeff_idx in xrange(self.n_coeffs):
            new_coeffs.append(self.coeffs[coeff_idx])
            new_powers.append(self.powers[coeff_idx])
            powers_dict[tuple(self.powers[coeff_idx])] = coeff_idx
        for coeff_idx in xrange(B.n_coeffs):
            cpow = tuple(B.powers[coeff_idx])
            out_idx = powers_dict.get(cpow)
            if out_idx != None:
                new_coeffs[out_idx] += B.coeffs[coeff_idx]
            else:
                new_coeffs.append(B.coeffs[coeff_idx])
                new_powers.append(B.powers[coeff_idx])
        return Centered_Scaled_Multinomial(new_coeffs, new_powers, self.center, self.scale)
    
    def __mul__(self, B):
        new_coeffs = []
        new_powers = []
        powers_dict = {}
        cur_out_idx = 0
        for coeff_idx_1 in xrange(self.n_coeffs):
            for coeff_idx_2 in xrange(B.n_coeffs):
                cpow = self.powers[coeff_idx_1] + B.powers[coeff_idx_2]
                tcpow = tuple(cpow)
                ccoeff = self.coeffs[coeff_idx_1]*B.coeffs[coeff_idx_2]
                out_idx = powers_dict.get(tcpow)
                if out_idx != None:
                    new_coeffs[out_idx] += ccoeff
                else:
                    powers_dict[tcpow] = cur_out_idx
                    new_coeffs.append(ccoeff)
                    new_powers.append(cpow)
                    cur_out_idx += 1
        return Centered_Scaled_Multinomial(new_coeffs, new_powers, self.center, self.scale)
    
    def __call__(self, x):
        xpowers = [[1.0] for i in xrange(self.n_dims)]
        for dim_idx in xrange(self.n_dims):
            pow_num = 1
            while pow_num <= self.max_powers[dim_idx]:
                xpowers[dim_idx].append(xpowers[dim_idx][-1]*x[dim_idx])
                pow_num += 1
        result = 0
        for coeff_idx in xrange(self.n_coeffs):
            cpow = 1.0
            for dim_idx in xrange(self.n_dims):
                pow_idx = self.powers[coeff_idx, dim_idx]
                cpow *= xpowers[dim_idx][pow_idx]
            result += self.coeffs[coeff_idx]*cpow
        return result
    
def multinomial_from_polynomial(poly_coeffs, center, scale, axis):
    """creates a multinomial from a 1d polynomial
    poly_coeffs: the 1d polynomial coefficients highest order first
    center: the multi-dimensional center M(x) is M_shift((x-center)/scale)  
    scale: the multi-dimensional scale
    axis: the number of the dimension which the multinomial will be a function
    """
    n_coeffs = len(poly_coeffs)
    n_dims = len(center)
    powers = np.zeros((n_coeffs, n_dims), dtype = int)
    powers[:, axis] = np.arange(n_coeffs-1, -1, -1)
    return Centered_Scaled_Multinomial(poly_coeffs, powers, center, scale)

class Binning:
    
    def __init__(self, bins):
        self.bins = bins
        self.lb = bins[0]
        self.ub = bins[-1]
        self.n_bounds = len(self.bins)
        self.last_bin = bins[0], bins[1]
        self.last_bin_idx = 0
    
    def get_bin_index(self, xvec):
        xv = np.array(xvec)
        out_idxs = np.zeros(len(xv.flat), dtype = int)
        for x_idx in xrange(len(xv.flat)):
            #check if the last solution still works
            if self.last_bin[0] <= xvec[x_idx] <= self.last_bin[1]:
                out_idxs[x_idx] = self.last_bin_idx
                continue
            lbi, ubi = 0, self.n_bounds-1
            #import pdb; pdb.set_trace()
            while True:
                mididx = (lbi+ubi)/2
                midbound = self.bins[mididx]
                if midbound <= xvec[x_idx]:
                    lbi = mididx
                else:
                    ubi = mididx
                if self.bins[lbi] <= xvec[x_idx] <= self.bins[lbi+1]:
                    self.last_bin = self.bins[lbi], self.bins[lbi+1]
                    self.last_bin_idx = lbi
                    break
            out_idxs[x_idx] = lbi
        return out_idxs

class Piecewise_Polynomial:
    
    def __init__(self, coefficients, control_points, centers = None, scales = None, bounds = (float("-inf"), float("inf")), fill_value = np.nan):
        """represents a piecewise polynomial function which transitions from one polynomial
        to the next at the control points.
        coefficients should be an (m, n) array
        m is the number of polynomial pieces == len(control_points) + 1
        n is the order of the polynomial pieces
        
        The function takes on values which are determined by the polynomial coefficients with the highest order terms coming first and each polynomail being centered around either the corresponding value in the centers array if it is passed as an argument By default the center is chosen as the midpoint of its two bounding points. If one of the current bounding points is + or -infinity the other bounding point is taken as the "center" of that polynomial bin
        
        Example:
        coefficients = np.array([[3, 2], [1, 0], [-1, -1]]) control_points = [5, 6]
        and bounds = (-float('inf'), 8)
        
        because the centers are 
        
        would be evaluated at a point x < 5 as 
        y = 3*(x-5) + 2
        
        and at a point 5 < x < 6 
        
        y = 1*(x-4.5) + 0
        
        and at a point 6 < x < 8
        
        y = -1*(x-7) + -1
        
        points above the upper bound of 8 will return nan
        """
        self.coefficients = coefficients
        self.bounds = bounds
        self.control_points = control_points
        n_polys, poly_order = coefficients.shape
        self.poly_order = poly_order
        self.ncp = len(control_points)  
        self.fill_value = fill_value
        boundary_points = np.zeros(self.ncp+2)
        boundary_points[0] = bounds[0]
        boundary_points[-1] = bounds[1]
        boundary_points[1:-1] = control_points
        self.binning = Binning(boundary_points)
        self.n_polys = n_polys
        if centers == None:
            self.centers = np.zeros(n_polys)
            #set the centers in such a way to allow for infinite bounds
            for center_idx in range(n_polys):
                lb = boundary_points[center_idx]
                ub = boundary_points[center_idx+1]
                if lb == float("-inf"):
                    lb = boundary_points[center_idx+1]
                if ub == float("inf"):
                    ub = boundary_points[center_idx]
                self.centers[center_idx] = 0.5*(lb+ub)
        else:
            self.centers = centers
        if scales == None:
            self.scales = np.ones(n_polys)
        else:
            self.scales = scales
        self.poly_list = []
        for poly_idx in range(n_polys):
            self.poly_list.append(Centered_Scaled_Polynomial(coefficients[poly_idx], self.centers[poly_idx], self.scales[poly_idx]))

    def __call__(self, x_in):
        output = np.zeros(x_in.shape)
        poly_idxs = self.binning.get_bin_index(x_in)
        output[np.isnan(poly_idxs)] = self.fill_value
        for p_idx in xrange(self.n_polys):
            pmask = poly_idxs == p_idx
            output[pmask] = self.poly_list[p_idx](x_in[pmask])
        return output

class Regularity_Constrained_Piecewise_Polynomial_Basis:

    def __init__(self, poly_order, control_points, centers = None, scales = None, regularity_constraints = None, bounds = (float("-inf"), float("inf"))):
        self.bounds = bounds
        self.control_points = control_points
        self.poly_order = poly_order
        self.ncp = len(control_points)
        if regularity_constraints == None:
            self.regularity_constraints = np.ones((poly_order, self.ncp), dtype = bool)
        else:
            self.regularity_constraints = regularity_constraints
        boundary_points = np.zeros(self.ncp+2)
        boundary_points[0] = bounds[0]
        boundary_points[-1] = bounds[1]
        boundary_points[1:-1] = control_points
        self.binning = Binning(boundary_points)
        n_polys = self.ncp+1
        self.n_polys = n_polys
        if centers == None:
            self.centers = np.zeros(n_polys)
            #set the centers in such a way to allow for infinite bounds
            for center_idx in range(n_polys):
                lb = boundary_points[center_idx]
                ub = boundary_points[center_idx+1]
                if lb == float("-inf"):
                    lb = boundary_points[center_idx+1]
                if ub == float("inf"):
                    ub = boundary_points[center_idx]
                self.centers[center_idx] = 0.5*(lb+ub)
        else:
            self.centers = centers
        if scales == None:
            scales = np.ones(n_polys)
        self.scales = scales
        poly_basis_list = [[] for i in range(n_polys)]
        for poly_i in range(n_polys):
            #cdomain = (self.boundary_points[poly_i], self.boundary_points[poly_i+1])
            for comp_i in range(poly_order+1):
                comp_vec = np.zeros((poly_order+1))
                comp_vec[comp_i] = 1.0
                #poly_basis_list[poly_i].append(Legendre(comp_vec, domain = cdomain)) 
                poly_basis_list[poly_i].append(Centered_Scaled_Polynomial(comp_vec, self.centers[poly_i], self.scales[poly_i]))
        #generate the constraint matrix
        #nrows = self.poly_order*self.ncp
        nrows = np.sum(self.regularity_constraints)
        constraint_matrix = np.zeros((nrows, (self.poly_order+1)*self.n_polys))
        constraint_number = 0
        nco, ncp = self.regularity_constraints.shape
        for control_i in range(ncp):
            c_control_point = self.control_points[control_i]
            l_basis = poly_basis_list[control_i] #left basis functions
            r_basis = poly_basis_list[control_i+1] #right basis functions
            for constraint_order in range(nco):
                if not self.regularity_constraints[constraint_order, control_i]:
                    continue
                fp_coeff_idx = control_i*(self.poly_order+1)
                sp_coeff_idx = (control_i+1)*(self.poly_order+1)
                #print "cp", control_i, "sp i", sp_coeff_idx
                for coefficient_i in range(self.poly_order+1):
                    lreg_coeff = l_basis[coefficient_i](c_control_point)
                    rreg_coeff = r_basis[coefficient_i](c_control_point)
                    constraint_matrix[constraint_number, fp_coeff_idx+coefficient_i] = lreg_coeff
                    constraint_matrix[constraint_number, sp_coeff_idx+coefficient_i] = -rreg_coeff
                #go up to the next order constraint by taking the derivative of our basis functions
                constraint_number += 1
                l_basis = [cpoly.deriv() for cpoly in l_basis]
                r_basis = [cpoly.deriv() for cpoly in r_basis]
        self.constraint_matrix = constraint_matrix
        u, s, v = lna.svd(self.constraint_matrix, full_matrices=True)
        self.n_basis = (self.poly_order+1)*self.n_polys-nrows
        self.basis_coefficients = np.zeros((self.n_basis, self.n_polys, self.poly_order+1))
        self.basis_polys = [[] for bi in range(self.n_basis)]
        for basis_i in range(self.n_basis):
            for poly_i in range(self.n_polys):
                coeff_lb = (self.poly_order+1)*poly_i
                coeff_ub = coeff_lb + self.poly_order+1
                ccoeffs = v[-(basis_i+1)][coeff_lb:coeff_ub]
                self.basis_coefficients[basis_i, poly_i] = ccoeffs
                self.basis_polys[basis_i].append(Centered_Scaled_Polynomial(ccoeffs, self.centers[poly_i], self.scales[poly_i]))    
    
    def get_basis(self, in_vec):
        xvec = np.array(in_vec)
        poly_idxs = self.binning.get_bin_index(xvec)
        out_basis = np.zeros((self.n_basis, len(xvec)))
        for basis_idx in xrange(self.n_basis):
            for poly_idx in xrange(self.n_polys):
                xmask = poly_idxs == poly_idx 
                cx = xvec[xmask]
                out_basis[basis_idx][xmask] = self.basis_polys[basis_idx][poly_idx](cx)
        return out_basis
    
    def regularize_ppol_basis_wrt_x_basis(self, xvec):
        """ if xvec has at least 1+ncp+order distinct values in it this function will modify the basis polynomial coefficients so that they represent piecewise polynomial functions that are orthogonal with respect to the basis in x. (Because of the way the piecewise polynomials are generated they are originally orthogonal in the space of polynomial coefficients)
"""
        cbasis = self.get_basis(xvec)
        u, s, v = lna.svd(cbasis, full_matrices = False)
        
        #TODO: use the orthogonalized basis vectors in v to set the self.basis_coefficients variable
        #import pdb; pdb.set_trace()
        
def fit_piecewise_polynomial(x, y, order, control_points, bounds = (float("-inf"), float("inf")), regularity_constraints = None, centers = None, scales = "autoscale"):
    if scales == "autoscale":
        scales = np.ones(len(control_points)+1)*np.std(x)*(len(control_points)+1)
    pp_gen = Regularity_Constrained_Piecewise_Polynomial_Basis(order, control_points=control_points, bounds = bounds, regularity_constraints = regularity_constraints, centers = centers, scales = scales)
    gbasis = pp_gen.get_basis(x)
    n_polys = len(control_points) + 1
    n_coeffs = order+1
    out_coeffs = np.zeros((n_polys, n_coeffs))
    fit_coeffs = np.linalg.lstsq(gbasis.transpose(), y)[0]
    for basis_idx in xrange(pp_gen.n_basis):
        c_coeffs = pp_gen.basis_coefficients[basis_idx].reshape((n_polys, n_coeffs))
        out_coeffs += c_coeffs*fit_coeffs[basis_idx]
    return Piecewise_Polynomial(out_coeffs, control_points, centers = centers, scales = scales, bounds = bounds)
    
        
RCPPB = Regularity_Constrained_Piecewise_Polynomial_Basis #a shortcut for the reiculously long name

if __name__ == "__main__":
    test_x = np.linspace(-1, 1, 4000)
    test_y = test_x * 2 - test_x**2 + 3.14
    
    ppol = fit_piecewise_polynomial(test_x, test_y, 3, np.array([-0.5, 0.5]))
    fit_y = ppol(test_x)
    if np.sum(np.abs(test_y-fit_y)) <= 1e-10:
        print "PASSED exact fit test"
    else:
        print "FAILED exact fit test"
    
    A = Centered_Scaled_Multinomial([1, 1], [[1, 0], [0, 1]], center = 0, scale = 0)
    B = Centered_Scaled_Multinomial([1, 1], [[1, 0], [0, 2]], center = 0, scale = 0)
    
    
    ##orthogonalization 
    #randx = np.random.random(30)-0.5
    #rcppb = RCPPB(3, [-0.5, 0.5])
    
