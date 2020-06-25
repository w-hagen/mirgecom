import numpy as np

class prometheus():

    def __init__(self):

        self.num_elements  = 3
        self.num_species   = 9
        self.num_reactions = 24
        self.num_falloff   = 2
        self.one_atm       = 1.01325e5;
        self.one_third     = 1.0 / 3.0;
        self.gas_constant  = 8314.4621;
        self.big_number    = 1.0e300;

        self.wts = np.array( [ 2.016000e+00, 1.008000e+00, 3.199800e+01, 1.599900e+01, 1.700700e+01, 3.300600e+01, 3.401400e+01, 1.801500e+01, 2.801400e+01 ] )
        self.iwts = np.reciprocal( self.wts )

        return

    def get_species_specific_heats_R(self, T):

        tt0 = T
        tt1 = T * tt0
        tt2 = T * tt1
        tt3 = T * tt2
        tt4 = np.power( T, -1.0 )
        tt5 = tt4 * tt4

        nz,ny,nx = T.shape
        cp0_R = np.zeros( self.num_species )

        cp_high  = 3.337279e+00 - 4.940247e-05 * tt0 + 4.994568e-07 * tt1 - 1.795664e-10 * tt2 + 2.002554e-14 * tt3
        cp_low   = 2.344331e+00 + 7.980521e-03 * tt0 - 1.947815e-05 * tt1 + 2.015721e-08 * tt2 - 7.376118e-12 * tt3
        cp0_R[0] = np.where( tt0 < 1.000000e+03, cp_low, cp_high )

        cp_high  = 2.500000e+00 - 2.308430e-11 * tt0 + 1.615619e-14 * tt1 - 4.735152e-18 * tt2 + 4.981974e-22 * tt3
        cp_low   = 2.500000e+00 + 7.053328e-13 * tt0 - 1.995920e-15 * tt1 + 2.300816e-18 * tt2 - 9.277323e-22 * tt3
        cp0_R[1] = np.where( tt0 < 1.000000e+03, cp_low, cp_high )

        cp_high  = 3.282538e+00 + 1.483088e-03 * tt0 - 7.579667e-07 * tt1 + 2.094706e-10 * tt2 - 2.167178e-14 * tt3
        cp_low   = 3.782456e+00 - 2.996734e-03 * tt0 + 9.847302e-06 * tt1 - 9.681295e-09 * tt2 + 3.243728e-12 * tt3
        cp0_R[2] = np.where( tt0 < 1.000000e+03, cp_low, cp_high )

        cp_high  = 2.569421e+00 - 8.597411e-05 * tt0 + 4.194846e-08 * tt1 - 1.001778e-11 * tt2 + 1.228337e-15 * tt3
        cp_low   = 3.168267e+00 - 3.279319e-03 * tt0 + 6.643064e-06 * tt1 - 6.128066e-09 * tt2 + 2.112660e-12 * tt3
        cp0_R[3] = np.where( tt0 < 1.000000e+03, cp_low, cp_high )

        cp_high  = 2.864729e+00 + 1.056504e-03 * tt0 - 2.590828e-07 * tt1 + 3.052187e-11 * tt2 - 1.331959e-15 * tt3
        cp_low   = 4.125306e+00 - 3.225449e-03 * tt0 + 6.527647e-06 * tt1 - 5.798536e-09 * tt2 + 2.062374e-12 * tt3
        cp0_R[4] = np.where( tt0 < 1.000000e+03, cp_low, cp_high )

        cp_high  = 4.017211e+00 + 2.239820e-03 * tt0 - 6.336581e-07 * tt1 + 1.142464e-10 * tt2 - 1.079085e-14 * tt3
        cp_low   = 4.301798e+00 - 4.749121e-03 * tt0 + 2.115829e-05 * tt1 - 2.427639e-08 * tt2 + 9.292251e-12 * tt3
        cp0_R[5] = np.where( tt0 < 1.000000e+03, cp_low, cp_high )

        cp_high  = 4.165003e+00 + 4.908317e-03 * tt0 - 1.901392e-06 * tt1 + 3.711860e-10 * tt2 - 2.879083e-14 * tt3
        cp_low   = 4.276113e+00 - 5.428224e-04 * tt0 + 1.673357e-05 * tt1 - 2.157708e-08 * tt2 + 8.624544e-12 * tt3
        cp0_R[6] = np.where( tt0 < 1.000000e+03, cp_low, cp_high )

        cp_high  = 3.033992e+00 + 2.176918e-03 * tt0 - 1.640725e-07 * tt1 - 9.704199e-11 * tt2 + 1.682010e-14 * tt3
        cp_low   = 4.198641e+00 - 2.036434e-03 * tt0 + 6.520402e-06 * tt1 - 5.487971e-09 * tt2 + 1.771978e-12 * tt3
        cp0_R[7] = np.where( tt0 < 1.000000e+03, cp_low, cp_high )

        cp_high  = 2.926640e+00 + 1.487977e-03 * tt0 - 5.684760e-07 * tt1 + 1.009704e-10 * tt2 - 6.753351e-15 * tt3
        cp_low   = 3.298677e+00 + 1.408240e-03 * tt0 - 3.963222e-06 * tt1 + 5.641515e-09 * tt2 - 2.444854e-12 * tt3
        cp0_R[8] = np.where( tt0 < 1.000000e+03, cp_low, cp_high )

        return cp0_R

    def get_species_enthalpies_RT(self, T):

        tt0 = T
        tt1 = T * tt0
        tt2 = T * tt1
        tt3 = T * tt2
        tt4 = np.power( T, -1.0 )
        tt5 = tt4 * tt4
        tt6 = np.log(tt0) * tt4

        nz,ny,nx = T.shape
        h0_RT = np.zeros( self.num_species )

        h_high  = 3.337279e+00 - 4.940247e-05 * 0.50 * tt0 + 4.994568e-07 * self.one_third * tt1 - 1.795664e-10 * 0.25 * tt2 + 2.002554e-14 * 0.20 * tt3 - 9.501589e+02 * tt4
        h_low   = 2.344331e+00 + 7.980521e-03 * 0.50 * tt0 - 1.947815e-05 * self.one_third * tt1 + 2.015721e-08 * 0.25 * tt2 - 7.376118e-12 * 0.20 * tt3 - 9.179352e+02 * tt4
        h_RT[0] = np.where( tt0 < 1.000000e+03, h_low, h_high )

        h_high  = 2.500000e+00 - 2.308430e-11 * 0.50 * tt0 + 1.615619e-14 * self.one_third * tt1 - 4.735152e-18 * 0.25 * tt2 + 4.981974e-22 * 0.20 * tt3 + 2.547366e+04 * tt4
        h_low   = 2.500000e+00 + 7.053328e-13 * 0.50 * tt0 - 1.995920e-15 * self.one_third * tt1 + 2.300816e-18 * 0.25 * tt2 - 9.277323e-22 * 0.20 * tt3 + 2.547366e+04 * tt4
        h_RT[1] = np.where( tt0 < 1.000000e+03, h_low, h_high )

        h_high  = 3.282538e+00 + 1.483088e-03 * 0.50 * tt0 - 7.579667e-07 * self.one_third * tt1 + 2.094706e-10 * 0.25 * tt2 - 2.167178e-14 * 0.20 * tt3 - 1.088458e+03 * tt4
        h_low   = 3.782456e+00 - 2.996734e-03 * 0.50 * tt0 + 9.847302e-06 * self.one_third * tt1 - 9.681295e-09 * 0.25 * tt2 + 3.243728e-12 * 0.20 * tt3 - 1.063944e+03 * tt4
        h_RT[2] = np.where( tt0 < 1.000000e+03, h_low, h_high )

        h_high  = 2.569421e+00 - 8.597411e-05 * 0.50 * tt0 + 4.194846e-08 * self.one_third * tt1 - 1.001778e-11 * 0.25 * tt2 + 1.228337e-15 * 0.20 * tt3 + 2.921758e+04 * tt4
        h_low   = 3.168267e+00 - 3.279319e-03 * 0.50 * tt0 + 6.643064e-06 * self.one_third * tt1 - 6.128066e-09 * 0.25 * tt2 + 2.112660e-12 * 0.20 * tt3 + 2.912226e+04 * tt4
        h_RT[3] = np.where( tt0 < 1.000000e+03, h_low, h_high )

        h_high  = 2.864729e+00 + 1.056504e-03 * 0.50 * tt0 - 2.590828e-07 * self.one_third * tt1 + 3.052187e-11 * 0.25 * tt2 - 1.331959e-15 * 0.20 * tt3 + 3.718858e+03 * tt4
        h_low   = 4.125306e+00 - 3.225449e-03 * 0.50 * tt0 + 6.527647e-06 * self.one_third * tt1 - 5.798536e-09 * 0.25 * tt2 + 2.062374e-12 * 0.20 * tt3 + 3.381538e+03 * tt4
        h_RT[4] = np.where( tt0 < 1.000000e+03, h_low, h_high )

        h_high  = 4.017211e+00 + 2.239820e-03 * 0.50 * tt0 - 6.336581e-07 * self.one_third * tt1 + 1.142464e-10 * 0.25 * tt2 - 1.079085e-14 * 0.20 * tt3 + 1.118567e+02 * tt4
        h_low   = 4.301798e+00 - 4.749121e-03 * 0.50 * tt0 + 2.115829e-05 * self.one_third * tt1 - 2.427639e-08 * 0.25 * tt2 + 9.292251e-12 * 0.20 * tt3 + 2.948080e+02 * tt4
        h_RT[5] = np.where( tt0 < 1.000000e+03, h_low, h_high )

        h_high  = 4.165003e+00 + 4.908317e-03 * 0.50 * tt0 - 1.901392e-06 * self.one_third * tt1 + 3.711860e-10 * 0.25 * tt2 - 2.879083e-14 * 0.20 * tt3 - 1.786179e+04 * tt4
        h_low   = 4.276113e+00 - 5.428224e-04 * 0.50 * tt0 + 1.673357e-05 * self.one_third * tt1 - 2.157708e-08 * 0.25 * tt2 + 8.624544e-12 * 0.20 * tt3 - 1.770258e+04 * tt4
        h_RT[6] = np.where( tt0 < 1.000000e+03, h_low, h_high )

        h_high  = 3.033992e+00 + 2.176918e-03 * 0.50 * tt0 - 1.640725e-07 * self.one_third * tt1 - 9.704199e-11 * 0.25 * tt2 + 1.682010e-14 * 0.20 * tt3 - 3.000430e+04 * tt4
        h_low   = 4.198641e+00 - 2.036434e-03 * 0.50 * tt0 + 6.520402e-06 * self.one_third * tt1 - 5.487971e-09 * 0.25 * tt2 + 1.771978e-12 * 0.20 * tt3 - 3.029373e+04 * tt4
        h_RT[7] = np.where( tt0 < 1.000000e+03, h_low, h_high )

        h_high  = 2.926640e+00 + 1.487977e-03 * 0.50 * tt0 - 5.684760e-07 * self.one_third * tt1 + 1.009704e-10 * 0.25 * tt2 - 6.753351e-15 * 0.20 * tt3 - 9.227977e+02 * tt4
        h_low   = 3.298677e+00 + 1.408240e-03 * 0.50 * tt0 - 3.963222e-06 * self.one_third * tt1 + 5.641515e-09 * 0.25 * tt2 - 2.444854e-12 * 0.20 * tt3 - 1.020900e+03 * tt4
        h_RT[8] = np.where( tt0 < 1.000000e+03, h_low, h_high )

        return h0_RT

    def get_species_entropies_R(self, T):

        tt0 = T
        tt1 = T * tt0
        tt2 = T * tt1
        tt3 = T * tt2
        tt4 = np.power( T, -1.0 )
        tt5 = tt4 * tt4
        tt6 = np.log(tt0)

        nz,ny,nx = T.shape
        s0_R = np.zeros( self.num_species )

        s_high  = 3.337279e+00 * tt6 - 4.940247e-05 * tt0 + 4.994568e-07 * 0.50 * tt1 - 1.795664e-10 * self.one_third * tt2 + 2.002554e-14 * 0.25 * tt3 - 3.205023e+00
        s_low   = 2.344331e+00 * tt6 + 7.980521e-03 * tt0 - 1.947815e-05 * 0.50 * tt1 + 2.015721e-08 * self.one_third * tt2 - 7.376118e-12 * 0.25 * tt3 + 6.830102e-01
        s0_R[0] = np.where( tt0 < 1.000000e+03, s_low, s_high )

        s_high  = 2.500000e+00 * tt6 - 2.308430e-11 * tt0 + 1.615619e-14 * 0.50 * tt1 - 4.735152e-18 * self.one_third * tt2 + 4.981974e-22 * 0.25 * tt3 - 4.466829e-01
        s_low   = 2.500000e+00 * tt6 + 7.053328e-13 * tt0 - 1.995920e-15 * 0.50 * tt1 + 2.300816e-18 * self.one_third * tt2 - 9.277323e-22 * 0.25 * tt3 - 4.466829e-01
        s0_R[1] = np.where( tt0 < 1.000000e+03, s_low, s_high )

        s_high  = 3.282538e+00 * tt6 + 1.483088e-03 * tt0 - 7.579667e-07 * 0.50 * tt1 + 2.094706e-10 * self.one_third * tt2 - 2.167178e-14 * 0.25 * tt3 + 5.453231e+00
        s_low   = 3.782456e+00 * tt6 - 2.996734e-03 * tt0 + 9.847302e-06 * 0.50 * tt1 - 9.681295e-09 * self.one_third * tt2 + 3.243728e-12 * 0.25 * tt3 + 3.657676e+00
        s0_R[2] = np.where( tt0 < 1.000000e+03, s_low, s_high )

        s_high  = 2.569421e+00 * tt6 - 8.597411e-05 * tt0 + 4.194846e-08 * 0.50 * tt1 - 1.001778e-11 * self.one_third * tt2 + 1.228337e-15 * 0.25 * tt3 + 4.784339e+00
        s_low   = 3.168267e+00 * tt6 - 3.279319e-03 * tt0 + 6.643064e-06 * 0.50 * tt1 - 6.128066e-09 * self.one_third * tt2 + 2.112660e-12 * 0.25 * tt3 + 2.051933e+00
        s0_R[3] = np.where( tt0 < 1.000000e+03, s_low, s_high )

        s_high  = 2.864729e+00 * tt6 + 1.056504e-03 * tt0 - 2.590828e-07 * 0.50 * tt1 + 3.052187e-11 * self.one_third * tt2 - 1.331959e-15 * 0.25 * tt3 + 5.701641e+00
        s_low   = 4.125306e+00 * tt6 - 3.225449e-03 * tt0 + 6.527647e-06 * 0.50 * tt1 - 5.798536e-09 * self.one_third * tt2 + 2.062374e-12 * 0.25 * tt3 - 6.904330e-01
        s0_R[4] = np.where( tt0 < 1.000000e+03, s_low, s_high )

        s_high  = 4.017211e+00 * tt6 + 2.239820e-03 * tt0 - 6.336581e-07 * 0.50 * tt1 + 1.142464e-10 * self.one_third * tt2 - 1.079085e-14 * 0.25 * tt3 + 3.785102e+00
        s_low   = 4.301798e+00 * tt6 - 4.749121e-03 * tt0 + 2.115829e-05 * 0.50 * tt1 - 2.427639e-08 * self.one_third * tt2 + 9.292251e-12 * 0.25 * tt3 + 3.716662e+00
        s0_R[5] = np.where( tt0 < 1.000000e+03, s_low, s_high )

        s_high  = 4.165003e+00 * tt6 + 4.908317e-03 * tt0 - 1.901392e-06 * 0.50 * tt1 + 3.711860e-10 * self.one_third * tt2 - 2.879083e-14 * 0.25 * tt3 + 2.916157e+00
        s_low   = 4.276113e+00 * tt6 - 5.428224e-04 * tt0 + 1.673357e-05 * 0.50 * tt1 - 2.157708e-08 * self.one_third * tt2 + 8.624544e-12 * 0.25 * tt3 + 3.435051e+00
        s0_R[6] = np.where( tt0 < 1.000000e+03, s_low, s_high )

        s_high  = 3.033992e+00 * tt6 + 2.176918e-03 * tt0 - 1.640725e-07 * 0.50 * tt1 - 9.704199e-11 * self.one_third * tt2 + 1.682010e-14 * 0.25 * tt3 + 4.966770e+00
        s_low   = 4.198641e+00 * tt6 - 2.036434e-03 * tt0 + 6.520402e-06 * 0.50 * tt1 - 5.487971e-09 * self.one_third * tt2 + 1.771978e-12 * 0.25 * tt3 - 8.490322e-01
        s0_R[7] = np.where( tt0 < 1.000000e+03, s_low, s_high )

        s_high  = 2.926640e+00 * tt6 + 1.487977e-03 * tt0 - 5.684760e-07 * 0.50 * tt1 + 1.009704e-10 * self.one_third * tt2 - 6.753351e-15 * 0.25 * tt3 + 5.980528e+00
        s_low   = 3.298677e+00 * tt6 + 1.408240e-03 * tt0 - 3.963222e-06 * 0.50 * tt1 + 5.641515e-09 * self.one_third * tt2 - 2.444854e-12 * 0.25 * tt3 + 3.950372e+00
        s0_R[8] = np.where( tt0 < 1.000000e+03, s_low, s_high )

        return s0_R

    def get_species_gibbs_RT(self, T):

        h0_RT = self.get_enthalpies_RT( T )
        s0_R  = self.get_entropies_R( T )
        g0_RT = h0_RT - s0_R

        return g0_RT

    def get_equilibrium_constants(self, T):

        RT = self.gas_constant * T
        C0 = self.one_atm * np.power( RT, -1.0 )

        k_eq = np.zeros( self.num_reactions )

        g0_RT = self.get_gibbs_RT( T )

        k_eq[0] = ( g0_RT[3] + g0_RT[4] ) - ( g0_RT[1] + g0_RT[2] )
        k_eq[1] = ( g0_RT[1] + g0_RT[4] ) - ( g0_RT[0] + g0_RT[3] )
        k_eq[2] = ( g0_RT[1] + g0_RT[7] ) - ( g0_RT[0] + g0_RT[4] )
        k_eq[3] = ( g0_RT[4] + g0_RT[4] ) - ( g0_RT[7] + g0_RT[3] )
        k_eq[4] =  C0 + ( g0_RT[0] ) - ( g0_RT[1] + g0_RT[1] )
        k_eq[5] =  C0 + ( g0_RT[7] ) - ( g0_RT[1] + g0_RT[4] )
        k_eq[6] =  C0 + ( g0_RT[2] ) - ( g0_RT[3] + g0_RT[3] )
        k_eq[7] =  C0 + ( g0_RT[4] ) - ( g0_RT[1] + g0_RT[3] )
        k_eq[8] =  C0 + ( g0_RT[5] ) - ( g0_RT[3] + g0_RT[4] )
        k_eq[9] =  C0 + ( g0_RT[5] ) - ( g0_RT[1] + g0_RT[2] )
        k_eq[10] = ( g0_RT[4] + g0_RT[4] ) - ( g0_RT[1] + g0_RT[5] )
        k_eq[11] = ( g0_RT[0] + g0_RT[2] ) - ( g0_RT[1] + g0_RT[5] )
        k_eq[12] = ( g0_RT[7] + g0_RT[3] ) - ( g0_RT[1] + g0_RT[5] )
        k_eq[13] = ( g0_RT[2] + g0_RT[4] ) - ( g0_RT[5] + g0_RT[3] )
        k_eq[14] = ( g0_RT[7] + g0_RT[2] ) - ( g0_RT[5] + g0_RT[4] )
        k_eq[15] = ( g0_RT[7] + g0_RT[2] ) - ( g0_RT[5] + g0_RT[4] )
        k_eq[16] =  C0 + ( g0_RT[6] ) - ( g0_RT[4] + g0_RT[4] )
        k_eq[17] = ( g0_RT[6] + g0_RT[2] ) - ( g0_RT[5] + g0_RT[5] )
        k_eq[18] = ( g0_RT[6] + g0_RT[2] ) - ( g0_RT[5] + g0_RT[5] )
        k_eq[19] = ( g0_RT[0] + g0_RT[5] ) - ( g0_RT[1] + g0_RT[6] )
        k_eq[20] = ( g0_RT[7] + g0_RT[4] ) - ( g0_RT[1] + g0_RT[6] )
        k_eq[21] = ( g0_RT[7] + g0_RT[5] ) - ( g0_RT[6] + g0_RT[4] )
        k_eq[22] = ( g0_RT[7] + g0_RT[5] ) - ( g0_RT[6] + g0_RT[4] )
        k_eq[23] = ( g0_RT[5] + g0_RT[4] ) - ( g0_RT[6] + g0_RT[3] )

        return k_eq

    def get_temperature(self, H_or_E, T_guess, Y, T):

        num_iter = 500
        tol = 1.0e-6
        RT  = self.gas_constant * T_guess
        T_i = T_guess
        dT = 1.0
        F  = H_or_E
        J  = 0.0
        he_k = np.zeros( self.num_species )
        c_k  = np.zeros( self.num_species )

        for iter in range( 0, num_iter):
            RT   = self.gas_constant * T_i
            c_k  = self.get_species_specific_heats_R( T_i )
            he_k = self.get_species_enthalpies_RT( T_i )
            c_k  -= 1.0
            he_k -= 1.0
            c_k  *= self.gas_constant * self.iwts
            he_k *= RT * self.iwts
            F    -= np.dot( he_k, Y )
            J    -= np.dot( c_k,  Y )
            dT    = - F / J
            T_i  += dT
            if np.abs( dT ) < tol:
                T = T_i
                break
            F = H_or_E
            J = 0.0

        T = T_i

        return

    def get_falloff_rates(self, T, C, k_fwd):

        TROE = 110
        log_T = np.log( T )
        inv_T = 1.0 / T
        k_hi = np.zeros( self.num_falloff )
        k_lo = np.zeros( self.num_falloff )
        pr   = np.zeros( self.num_falloff )
        work = np.zeros( self.num_falloff )
        falloff_type = 100 * np.ones( self.num_falloff, dtype = int )

        k_hi[0] = np.exp(2.226013e+01 + 4.400000e-01 * log_T)
        k_lo[0] = np.exp(3.168281e+01 - 1.400000e+00 * log_T)

        k_hi[1] = np.exp(2.528239e+01 - 2.700000e-01 * log_T)
        k_lo[1] = np.exp(4.476435e+01 - 3.200000e+00 * log_T)

        pr[0] = 2.50e+00 * C[0] + 1.00e+00 * C[1] + 1.00e+00 * C[2] + 1.00e+00 * C[3] + 1.00e+00 * C[4] + 1.00e+00 * C[5] + 1.00e+00 * C[6] + 1.60e+01 * C[7] + 1.00e+00 * C[8] 
        pr[1] = 2.50e+00 * C[0] + 1.00e+00 * C[1] + 1.00e+00 * C[2] + 1.00e+00 * C[3] + 1.00e+00 * C[4] + 1.00e+00 * C[5] + 1.00e+00 * C[6] + 6.00e+00 * C[7] + 1.00e+00 * C[8] 

        for i_falloff in range( 0, self.num_falloff ):
            pr[i_falloff] *= ( k_lo[i_falloff] / k_hi[i_falloff] )

        falloff_type[0] = 110
        falloff_type[1] = 110

        work[0] = (1.0 - 5.000000e-01) * std::exp( - 1.000000e+30 * T) + 5.000000e-01 * std::exp( - 1.000000e-30 * T) + std::exp( - 1.000000e-17 * invT);
        work[1] = (1.0 - 4.300000e-01) * std::exp( - 1.000000e-30 * T) + 4.300000e-01 * std::exp( - 1.000000e+30 * T) + std::exp( - 1.000000e-17 * invT);

        for i_falloff in range( 0, self.num_falloff ):
            lpr = np.log10(pr[i_falloff])
            if falloff_type[i_falloff] == TROE:
                cc = -0.40 - 0.67 * np.log10(work[i_falloff])
                nn =  0.75 - 1.27 * np.log10(work[i_falloff])
                f1 =  (lpr + cc)/(nn - 0.14 * (lpr + cc))
                work[i_falloff] = np.log10(work[i_falloff])/(1 + f1 * f1)
                work[i_falloff] = 10.0 ** work[i_falloff]
             work[i_falloff] = (pr[i_falloff] * work[i_falloff])/(1 + pr[i_falloff])

        k_fwd[9] = k_hi[0] * work[0]
        k_fwd[16] = k_hi[1] * work[1]

        return

    def get_rate_coefficients(self, T, C):

        log_T = np.log( T )
        inv_T = 1.0 / T
        k_eq  = self.get_equilibrium_constants( T )
        k_fwd = np.zeros( self.num_reactions )
        k_rev = np.zeros( self.num_reactions )

        k_fwd[0] = std::exp(3.119207e+01 - 7.000000e-01 * log_T - 8.589852e+03 * inv_T)
        k_fwd[1] = std::exp(3.923952e+00 + 2.670000e+00 * log_T - 3.165568e+03 * inv_T)
        k_fwd[2] = std::exp(1.397251e+01 + 1.300000e+00 * log_T - 1.829343e+03 * inv_T)
        k_fwd[3] = std::exp(6.551080e+00 + 2.330000e+00 * log_T - 7.320978e+03 * inv_T)
        k_fwd[4] = std::exp(2.789339e+01 - 1.000000e+00 * log_T)
        k_fwd[5] = std::exp(3.822766e+01 - 2.000000e+00 * log_T)
        k_fwd[6] = std::exp(2.254296e+01 - 5.000000e-01 * log_T)
        k_fwd[7] = std::exp(2.918071e+01 - 1.000000e+00 * log_T)
        k_fwd[8] = std::exp(2.280271e+01)
        k_fwd[10] = std::exp(2.498312e+01 - 1.484161e+02 * inv_T)
        k_fwd[11] = std::exp(2.353267e+01 - 4.140977e+02 * inv_T)
        k_fwd[12] = std::exp(2.415725e+01 - 8.659610e+02 * inv_T)
        k_fwd[13] = std::exp(2.371900e+01)
        k_fwd[14] = std::exp(2.683251e+01 - 5.500055e+03 * inv_T)
        k_fwd[15] = std::exp(2.411777e+01 + 2.501665e+02 * inv_T)
        k_fwd[17] = std::exp(1.908337e+01 + 7.090055e+02 * inv_T)
        k_fwd[18] = std::exp(2.535799e+01 - 5.556583e+03 * inv_T)
        k_fwd[19] = std::exp(2.385876e+01 - 4.000619e+03 * inv_T)
        k_fwd[20] = std::exp(2.302585e+01 - 1.804085e+03 * inv_T)
        k_fwd[21] = std::exp(2.505268e+01 - 3.659888e+03 * inv_T)
        k_fwd[22] = std::exp(2.127715e+01 - 1.599622e+02 * inv_T)
        k_fwd[23] = std::exp(9.172639e+00 + 2.000000e+00 * log_T - 2.008548e+03 * inv_T)

        k_fwd[4] *= ( 2.500e+00 * C[0] + 1.000e+00 * C[1] + 1.000e+00 * C[2] + 1.000e+00 * C[3] + 1.000e+00 * C[4] + 1.000e+00 * C[5] + 1.000e+00 * C[6] + 1.200e+01 * C[7] + 1.000e+00 * C[8] ) 
        k_fwd[5] *= ( 2.500e+00 * C[0] + 1.000e+00 * C[1] + 1.000e+00 * C[2] + 1.000e+00 * C[3] + 1.000e+00 * C[4] + 1.000e+00 * C[5] + 1.000e+00 * C[6] + 1.200e+01 * C[7] + 1.000e+00 * C[8] ) 
        k_fwd[6] *= ( 2.500e+00 * C[0] + 1.000e+00 * C[1] + 1.000e+00 * C[2] + 1.000e+00 * C[3] + 1.000e+00 * C[4] + 1.000e+00 * C[5] + 1.000e+00 * C[6] + 1.200e+01 * C[7] + 1.000e+00 * C[8] ) 
        k_fwd[7] *= ( 2.500e+00 * C[0] + 1.000e+00 * C[1] + 1.000e+00 * C[2] + 1.000e+00 * C[3] + 1.000e+00 * C[4] + 1.000e+00 * C[5] + 1.000e+00 * C[6] + 1.200e+01 * C[7] + 1.000e+00 * C[8] ) 
        k_fwd[8] *= ( 2.500e+00 * C[0] + 1.000e+00 * C[1] + 1.000e+00 * C[2] + 1.000e+00 * C[3] + 1.000e+00 * C[4] + 1.000e+00 * C[5] + 1.000e+00 * C[6] + 1.200e+01 * C[7] + 1.000e+00 * C[8] ) 

        get_falloff_rates(T, C, k_fwd)

        for i in range( 0, self.num_reactions ):
            if k_eq[i] > self.big_number:
                k_eq[i] = self.big_number
            k_rev[i] = k_fwd[i] * np.exp( k_eq[i] )

        return k_fwd, k_eq

    def get_net_rates_of_progress(self, T, C):

        R_fwd = np.zeros( self.num_reactions )
        R_rev = np.zeros( self.num_reactions )
        R_net = np.zeros( self.num_reactions )
        k_fwd, k_rev = self.get_rate_coefficients( T, C )

        R_fwd[0] = k_fwd[0] * C[1] * C[2]
        R_rev[0] = k_rev[0] * C[3] * C[4]

        R_fwd[1] = k_fwd[1] * C[0] * C[3]
        R_rev[1] = k_rev[1] * C[1] * C[4]

        R_fwd[2] = k_fwd[2] * C[0] * C[4]
        R_rev[2] = k_rev[2] * C[1] * C[7]

        R_fwd[3] = k_fwd[3] * C[7] * C[3]
        R_rev[3] = k_rev[3] * C[4] * C[4]

        R_fwd[4] = k_fwd[4] * C[1] * C[1]
        R_rev[4] = k_rev[4] * C[0]

        R_fwd[5] = k_fwd[5] * C[1] * C[4]
        R_rev[5] = k_rev[5] * C[7]

        R_fwd[6] = k_fwd[6] * C[3] * C[3]
        R_rev[6] = k_rev[6] * C[2]

        R_fwd[7] = k_fwd[7] * C[1] * C[3]
        R_rev[7] = k_rev[7] * C[4]

        R_fwd[8] = k_fwd[8] * C[3] * C[4]
        R_rev[8] = k_rev[8] * C[5]

        R_fwd[9] = k_fwd[9] * C[1] * C[2]
        R_rev[9] = k_rev[9] * C[5]

        R_fwd[10] = k_fwd[10] * C[1] * C[5]
        R_rev[10] = k_rev[10] * C[4] * C[4]

        R_fwd[11] = k_fwd[11] * C[1] * C[5]
        R_rev[11] = k_rev[11] * C[0] * C[2]

        R_fwd[12] = k_fwd[12] * C[1] * C[5]
        R_rev[12] = k_rev[12] * C[7] * C[3]

        R_fwd[13] = k_fwd[13] * C[5] * C[3]
        R_rev[13] = k_rev[13] * C[2] * C[4]

        R_fwd[14] = k_fwd[14] * C[5] * C[4]
        R_rev[14] = k_rev[14] * C[7] * C[2]

        R_fwd[15] = k_fwd[15] * C[5] * C[4]
        R_rev[15] = k_rev[15] * C[7] * C[2]

        R_fwd[16] = k_fwd[16] * C[4] * C[4]
        R_rev[16] = k_rev[16] * C[6]

        R_fwd[17] = k_fwd[17] * C[5] * C[5]
        R_rev[17] = k_rev[17] * C[6] * C[2]

        R_fwd[18] = k_fwd[18] * C[5] * C[5]
        R_rev[18] = k_rev[18] * C[6] * C[2]

        R_fwd[19] = k_fwd[19] * C[1] * C[6]
        R_rev[19] = k_rev[19] * C[0] * C[5]

        R_fwd[20] = k_fwd[20] * C[1] * C[6]
        R_rev[20] = k_rev[20] * C[7] * C[4]

        R_fwd[21] = k_fwd[21] * C[6] * C[4]
        R_rev[21] = k_rev[21] * C[7] * C[5]

        R_fwd[22] = k_fwd[22] * C[6] * C[4]
        R_rev[22] = k_rev[22] * C[7] * C[5]

        R_fwd[23] = k_fwd[23] * C[6] * C[3]
        R_rev[23] = k_rev[23] * C[5] * C[4]

        for i in range( 0, self.num_reactions ):
            R_net[i] = R_fwd[i] - R_rev[i]

        return R_net

    def get_net_production_rates(self, p, T, Y):

        W   = np.reciprocal( np.dot( self.iwts, Y ) )
        RT  = self.gas_constant * T
        rho = p * W / RT
        C   = self.iwts * rho * Y
        R_net = self.get_net_rates_of_progress( T, C )

        omega[0] =  - R_net[1] - R_net[2] + R_net[4] + R_net[11] + R_net[19]
        omega[1] =  - R_net[0] + R_net[1] + R_net[2] - R_net[4] - R_net[4] - R_net[5] - R_net[7] - R_net[9] - R_net[10] - R_net[11] - R_net[12] - R_net[19] - R_net[20]
        omega[2] =  - R_net[0] + R_net[6] - R_net[9] + R_net[11] + R_net[13] + R_net[14] + R_net[15] + R_net[17] + R_net[18]
        omega[3] =  + R_net[0] - R_net[1] - R_net[3] - R_net[6] - R_net[6] - R_net[7] - R_net[8] + R_net[12] - R_net[13] - R_net[23]
        omega[4] =  + R_net[0] + R_net[1] - R_net[2] + R_net[3] + R_net[3] - R_net[5] + R_net[7] - R_net[8] + R_net[10] + R_net[10] + R_net[13] - R_net[14] - R_net[15] - R_net[16] - R_net[16] + R_net[20] - R_net[21] - R_net[22] + R_net[23]
        omega[5] =  + R_net[8] + R_net[9] - R_net[10] - R_net[11] - R_net[12] - R_net[13] - R_net[14] - R_net[15] - R_net[17] - R_net[17] - R_net[18] - R_net[18] + R_net[19] + R_net[21] + R_net[22] + R_net[23]
        omega[6] =  + R_net[16] + R_net[17] + R_net[18] - R_net[19] - R_net[20] - R_net[21] - R_net[22] - R_net[23]
        omega[7] =  + R_net[2] - R_net[3] + R_net[5] + R_net[12] + R_net[14] + R_net[15] + R_net[20] + R_net[21] + R_net[22]

        return omega

