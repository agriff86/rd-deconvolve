#ifndef _FAST_DETECTOR_MODEL_HPP
#define _FAST_DETECTOR_MODEL_HPP


 /* Interface function */
int integrate_radon_detector(int N_times,
                              double timestep,
                              int interpolation_mode,
                              double *external_radon_conc,
                              double *internal_airt_history,
                              double *initial_state,
                              double *state_history,
                              double *parameters);


/* for testing linear interpolation class */
double linear_interpolation(double xi, int N, double timestep, const double *y, const int mode);

// constants (despite appearances, only correct to about 2 dec places but
//            written like this to match the python version)
const double lamrn = 2.1001405267111005e-06;
const double lama = 0.0037876895112565318;
const double lamb = 0.00043106167945270227;
const double lamc = 0.00058052527685087548;

// state vector:  Nrnd, Nrn, Fa, Fb, Fc, Acc_counts
const int NUM_STATE_VARIABLES = 6;

const int IDX_Nrnd = 0;
const int IDX_Nrn = 1;
const int IDX_Fa = 2;
const int IDX_Fb = 3;
const int IDX_Fc = 4;
const int IDX_Acc_counts = 5;


// model parameters
const int NUM_PARAMETERS = 15;


#endif
