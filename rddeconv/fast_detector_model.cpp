/*
 * Driver for ODE integration
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <boost/numeric/odeint.hpp>

#include "fast_detector_model.hpp"

/* The type of container used to hold the state vector */
typedef std::vector< double > state_type;

/* the type returned by std::vector.size() */
typedef std::vector< double >::size_type T_vecsize;

/* copy a carray into a vector */
std::vector< double >
carray_to_vector(const double *data, int N)
{
    std::vector< double > vec;
    vec.assign(data, data+N);
    return vec;
}


/* TODO: doc */
struct linear_interpolator {
    double m_timestep;
    std::vector< double > m_interpolation_values;
    int m_mode;
    T_vecsize N;

    linear_interpolator( double timestep, std::vector< double > intvals, int mode) :
                                    m_timestep(timestep),
                                    m_interpolation_values(intvals),
                                    m_mode(mode)
    {
        N = intvals.size();
        /*
        std::cout << "linear interpolator init\n";
        for(int ii=0; ii<10; ii++)
        {
            std::cout << ii << '\t' << intvals[ii] << std::endl;
        }
        */

    }

    double operator() ( const double t )
    {
        double p, i1, i2, w1, w2, r;
        T_vecsize ii1, ii2;
        //linear interpolation
        p = t/m_timestep;
        if (p <= 0)
        {
            r = m_interpolation_values[0];
        }
        else if (p >= N-1)
        {
            r = m_interpolation_values[N-1];
        }
        else
        {
            /* piecewise-constant - return Y0 for t < 0 between t0 and t1
                                           Y1 for t0 <= t < t1 , etc*/
            if(m_mode == 0)
            {
                i1 = floor(p);
                ii1 = (T_vecsize)i1 + 1;
                r = m_interpolation_values[ii1];
            }
            /* linear interpolation */
            else if (m_mode == 1)
            {
                i1 = floor(p);
                i2 = ceil(p);
                w2 = (p-i1);
                w1 = 1.0-w2;
                ii1 = (T_vecsize)i1;
                ii2 = (T_vecsize)i2;
                /*
                std::cout << "p:" << p << " t:" << t << "i1:" << i1 << " i2:" << i2 << std::endl;
                std::cout << w1 << " " << w2 << std::endl;
                std::cout << m_interpolation_values[ii1] << " " << m_interpolation_values[ii2] << std::endl;
                */
                //std::cout << ii1 << " " << ii2 << std::endl;
                r = w1*m_interpolation_values[ii1] + w2*m_interpolation_values[ii2];
            }
            /* invalid mode TODO: how to flag the error? */
            else
            {
                assert(false);
                r = INFINITY;
            }
        }
        return r;
    }
};


// calculate Na, Nb, Nc from generated code (copied from generated.c)
void calc_NaNbNc(double t, double Nrn, double lamp,
                 double &Na, double &Nb, double &Nc)
{
    using namespace std; //for math functions pow, exp
    Na = Nrn*lamrn/(lama + lamp) - Nrn*lamrn*exp(-t*(lama + lamp))/(lama +
            lamp);
    Nb = Nrn*lama*lamrn/(lama*lamb + pow(lamp, 2) + lamp*(lama + lamb)) -
            Nrn*lama*lamrn*exp(-t*(lamb + lamp))/(lama*lamb - pow(lamb, 2) +
            lamp*(lama - lamb)) + Nrn*lama*lamrn*exp(-t*(lama + lamp))/
            (pow(lama, 2) - lama*lamb + lamp*(lama - lamb));
    Nc = Nrn*lama*lamb*lamrn/(lama*lamb*lamc + pow(lamp, 3) + pow(lamp, 2)*
            (lama + lamb + lamc) + lamp*(lama*lamb + lamc*(lama + lamb))) -
            Nrn*lama*lamb*lamrn*exp(-t*(lamc + lamp))/(lama*lamb*lamc +
            pow(lamc, 3) - pow(lamc, 2)*(lama + lamb) + lamp*(lama*lamb +
            pow(lamc, 2) - lamc*(lama + lamb))) + Nrn*lama*lamb*lamrn*
            exp(-t*(lamb + lamp))/(lama*pow(lamb, 2) - pow(lamb, 3) -
            lamc*(lama*lamb - pow(lamb, 2)) + lamp*(lama*lamb - pow(lamb, 2)
            - lamc*(lama - lamb))) - Nrn*lama*lamb*lamrn*exp(-t*(lama + lamp))
            /(pow(lama, 3) - pow(lama, 2)*lamb - lamc*(pow(lama, 2) - lama*lamb)
             + lamp*(pow(lama, 2) - lama*lamb - lamc*(lama - lamb)));
    return;
}

// Compute the steady-state solution for the state variable Y
// returns Yss : state_type( [Nrnd, Nrn, Fa, Fb, Fc] )

state_type calc_steady_state(double Nrn, double Q, double rs, double lamp,
                      double V_tank, double recoil_prob, double eff)
{
    double tt, Na, Nb, Nc, Fa, Fb, Fc;
    state_type Yss;
    double Acc_counts;
    // transit time assuming plug flow in the tank
    tt = V_tank / Q;
    calc_NaNbNc(tt, Nrn, lamp, Na, Nb, Nc);

    // expressions based on these lines from detector_state_rate_of_change
    // dFadt = Q*rs*Na - Fa*lama
    // dFbdt = Q*rs*Nb - Fb*lamb + Fa*lama * (1.0-recoil_prob)
    // dFcdt = Q*rs*Nc - Fc*lamc + Fb*lamb
    Fa = Na*Q*rs/lama;
    Fb = (Q*rs*Nb + Fa*lama * (1.0-recoil_prob)) / lamb;
    Fc = (Q*rs*Nc + Fb*lamb) / lamc;

    // accumulated counts, in this case over one second
    Acc_counts = eff*(Fa*lama + Fc*lamc);
    // pack output into vector
    Yss.push_back(Nrn);
    Yss.push_back(Nrn);
    Yss.push_back(Nrn);
    Yss.push_back(Fa);
    Yss.push_back(Fb);
    Yss.push_back(Fc);
    Yss.push_back(Acc_counts);
    return Yss;
}



//[ rhs_class
/* The rhs of x' = f(x) defined as a class */
struct two_filter_detector {

    linear_interpolator m_bc;
    linear_interpolator m_airt;
    // parameters
    double Q;
    double rs;
    double lamp;
    double eff;
    double Q_external;
    double V_delay;
    double V_delay_2;
    double V_tank;
    double t_delay;
    double recoil_prob;
    double cal_source_strength;
    double cal_begin;
    double cal_duration;
    double inj_source_strength;
    double inj_begin;
    double inj_duration;

    two_filter_detector( linear_interpolator bc,
                         linear_interpolator airt,
                         double * parameters ) : m_bc(bc), m_airt(airt)
    {
        // store parameters
        Q = *parameters++;
        rs = *parameters++;
        lamp = *parameters++;
        eff = *parameters++;
        Q_external = *parameters++;
        V_delay = *parameters++;
        V_delay_2 = *parameters++;
        V_tank = *parameters++;
        t_delay = *parameters++;
        recoil_prob = *parameters++;
        cal_source_strength = *parameters++;
        cal_begin = *parameters++;
        cal_duration = *parameters++;
        inj_source_strength = *parameters++;
        inj_begin = *parameters++;
        inj_duration = *parameters++;

        /*
        std::cout << "Q: " << Q << std::endl
                  << "rs: " << rs << std::endl
                  << "lamp: " << lamp << std::endl
                  << "eff: " << eff << std::endl
                  << "Q_external: " << Q_external << std::endl
                  << "V_delay: " << V_delay << std::endl
                  << "V_delay_2: " << V_delay_2 << std::endl
                  << "V_tank: " << V_tank << std::endl
                  << "t_delay: " << t_delay << std::endl
                  << "recoil_prob: " << recoil_prob << std::endl;
        */
        /*
        std::cout << "t/60\tT\tdTdt\tRn\n";
        for (double t=-60; t<600; t+=60)
        {
            const double delt = 60;
            const double T = m_airt(t);
            double dTdt = (T - m_airt(t-delt)) / delt;
            std::cout << t/60.0 << '\t' << T << '\t' << dTdt << '\t'
                      << m_bc(t)
                      << std::endl;
        }
        */
    }

    void operator() ( const state_type &x , state_type &dxdt , const double t )
    {
        double Nrnd, Nrnd2, Nrn, Fa, Fb, Fc;
        double Nrn_inj; //radon concentration from source injected at inlet
        double Nrn_cal; //radon concentration from calibration source
        double dNrnddt, dNrnd2dt, dNrndt, dFadt, dFbdt, dFcdt, dAcc_countsdt;
        double tt, Na, Nb, Nc;
        // boundary condition
        double Nrn_ext = m_bc(t - t_delay);
        // temperature and temperature change
        double T = m_airt(t - t_delay);
        // for computing temerature rate of change by finite-difference
        const double delt = 60;
        double dTdt = (T - m_airt(t-t_delay-delt)) / delt;
        // copied from theoretical_model.py:detector_state_rate_of_change...
        // unpack state vector
        Nrnd = x[IDX_Nrnd];
        Nrnd2 = x[IDX_Nrnd2];
        Nrn = x[IDX_Nrn];
        Fa = x[IDX_Fa];
        Fb = x[IDX_Fb];
        Fc = x[IDX_Fc];
        // The radon concentration flowing into the inlet needs to be
        // bumped up if the injection source is active
        bool inj_is_active = inj_source_strength > 0.0
                                && (t - t_delay) > inj_begin
                                && (t - t_delay) <= inj_begin+inj_duration;
        if(inj_is_active)
        {
            Nrn_inj = inj_source_strength / Q_external;
        }
        else
        {
            Nrn_inj = 0;
        }

        // The radon concentration flowing into the main tank needs to be
        // bumped up if the calibration source is active
        bool cal_is_active = cal_source_strength > 0.0
                                && (t - t_delay) > cal_begin
                                && (t - t_delay) <= cal_begin+cal_duration;
        if(cal_is_active)
        {
            Nrn_cal = cal_source_strength / Q_external;
        }
        else
        {
            Nrn_cal = 0;
        }

        // make sure that we can't have V_delay_2 > 0 when V_delay == 0
        if (V_delay == 0 && V_delay_2 > 0)
        {
          V_delay = V_delay_2;
          V_delay_2 = 0;
        }

        // effect of delay and tank volumes (allow V_delay to be zero)
        if (V_delay == 0.0) // no delay tanks
        {
            dNrndt = Q_external / V_tank * (Nrn_ext + Nrn_cal + Nrn_inj - Nrn)
                     - Nrn*lamrn;
            // Nrnd,Nrnd2 become unimportant, but we need to do something with them
            // so just apply the same equation as for Nrn
            dNrnddt = Q_external / V_tank * (Nrn_ext + Nrn_cal + Nrn_inj - Nrnd)
                      - Nrnd*lamrn;
            dNrnd2dt = Q_external / V_tank * (Nrn_ext + Nrn_cal + Nrn_inj - Nrnd2)
                                - Nrnd2*lamrn;
        }
        else if (V_delay > 0.0 && V_delay_2 == 0.0) //one delay tank
        {
          dNrnddt = Q_external / V_delay * (Nrn_ext + Nrn_inj - Nrnd)
                                                                - Nrnd*lamrn;
          dNrndt = Q_external / V_tank * (Nrnd + Nrn_cal - Nrn) - Nrn*lamrn;
          // unused, but apply same eqn as delay tank 1
          dNrnd2dt = Q_external / V_delay * (Nrn_ext + Nrn_inj - Nrnd2)
                                                               - Nrnd2*lamrn;
        }
        else // two delay tanks
        {
            dNrnddt = Q_external / V_delay * (Nrn_ext + Nrn_inj - Nrnd)
                                                                  - Nrnd*lamrn;
            dNrnd2dt = Q_external / V_delay_2 * (Nrnd - Nrnd2) - Nrnd2*lamrn;
            dNrndt = Q_external / V_tank * (Nrnd2 + Nrn_cal - Nrn) - Nrn*lamrn;
        }


        // effect of temperature changes causing the tank to 'breathe'
        dNrndt -= Nrnd * dTdt/T;
        // Na, Nb, Nc from steady-state in tank
        // transit time assuming plug flow in the tank
        tt = V_tank / Q;
        calc_NaNbNc(tt, Nrn, lamp, Na, Nb, Nc);
        // compute rate of change of each state variable
        dFadt = Q*rs*Na - Fa*lama;
        dFbdt = Q*rs*Nb - Fb*lamb + Fa*lama * (1.0-recoil_prob);
        dFcdt = Q*rs*Nc - Fc*lamc + Fb*lamb;
        dAcc_countsdt = eff*(Fa*lama + Fc*lamc);
        // pack into dxdt
        dxdt[IDX_Nrnd] = dNrnddt;
        dxdt[IDX_Nrnd2] = dNrnd2dt;
        dxdt[IDX_Nrn] = dNrndt;
        dxdt[IDX_Fa] = dFadt;
        dxdt[IDX_Fb] = dFbdt;
        dxdt[IDX_Fc] = dFcdt;
        dxdt[IDX_Acc_counts] = dAcc_countsdt;
        return;
    }
};
//]


//[ integrate_observer
class push_back_state_and_time
{
    std::vector< state_type >& m_states;
    std::vector< double >& m_times;

    public:
    push_back_state_and_time( std::vector< state_type > &states ,
                              std::vector< double > &times )
    : m_states( states ) , m_times( times ) { }

    void operator()( const state_type &x , double t )
    {
        m_states.push_back( x );
        m_times.push_back( t );
        // std::cout << t << std::endl;
    }

    double getval(T_vecsize idx_time, T_vecsize idx_statevec)
    {
        return m_states[idx_time][idx_statevec];
    }
};
//]


/* Interface function */
int integrate_radon_detector(int N_times,
                             double timestep,
                             int interpolation_mode,
                             double *external_radon_conc,
                             double *airt,
                             double *initial_state,
                             double *state_history,
                             double *parameters)
{
    using namespace boost::numeric::odeint;

    // initialise the boundary conditions object
    std::vector< double > intvals;
    intvals = carray_to_vector(external_radon_conc, N_times);
    linear_interpolator boundary_conditions(timestep, intvals, interpolation_mode);

    // initialise the air temperature object (always use linear intrpolation)
    std::vector< double > airt_intvals;
    airt_intvals = carray_to_vector(airt, N_times);
    linear_interpolator boundary_conditions_airt(timestep, airt_intvals, 1);

    // copy initial state into a state_type object
    state_type x0 = carray_to_vector(initial_state, NUM_STATE_VARIABLES);
    double t0 = 0.0;
    double t1 = timestep*N_times;

    // initialise the RHS of the system of equations
    two_filter_detector system_of_equations(boundary_conditions,
                                            boundary_conditions_airt,
                                            parameters);

    // initialise the stepper (integrator)
    // ref: http://www.boost.org/doc/libs/1_57_0/libs/numeric/odeint/doc/html/boost_numeric_odeint/odeint_in_detail/steppers.html
    // (it would be nicer to use 'auto' for the type, but we'd need extra compiler flags)

    // stepper options: http://www.boost.org/doc/libs/1_57_0/libs/numeric/odeint/doc/html/boost_numeric_odeint/odeint_in_detail/steppers.html#boost_numeric_odeint.odeint_in_detail.steppers.stepper_overview

    //// OPTION 1: dense output stepper (step size can be larger than output grid)
    //typedef boost::numeric::odeint::result_of::make_dense_output<
    //    runge_kutta_dopri5< state_type > >::type dense_stepper_type;
    //dense_stepper_type stepper = make_dense_output( 1.0e-6 , 1.0e-5 ,
    //                                    runge_kutta_dopri5< state_type >() );

    // OPTION 2: controlled stepper (error is controlled, but dt must be chosen
    // such that the stepper ends up at exactly the endpoint)
    typedef boost::numeric::odeint::result_of::make_controlled<
        runge_kutta_dopri5< state_type > >::type controlled_stepper_type;
    controlled_stepper_type stepper = make_controlled( 1.0e-6 , 1.0e-5 ,
                                        runge_kutta_dopri5< state_type >() );



    //initialise observer (i.e. container for recording output at each step)
    std::vector< state_type > obs_x_vec;
    std::vector< double > obs_times_vec;
    obs_x_vec.reserve(N_times);
    obs_times_vec.reserve(N_times);
    push_back_state_and_time observer(obs_x_vec, obs_times_vec);

    // integrate the system of odes
    // ref: http://www.boost.org/doc/libs/1_57_0/libs/numeric/odeint/doc/html/boost_numeric_odeint/odeint_in_detail/integrate_functions.html
    boost::numeric::odeint::integrate_const(stepper, system_of_equations,
                                            x0, t0, t1, timestep, observer);

    //copy the integration history from the observer into the output array
    double *state_iter = state_history;
    for (int ii=0; ii<N_times; ii++){
        for (int jj=0; jj<NUM_STATE_VARIABLES; jj++){
            *state_iter = observer.getval(ii,jj);
            state_iter++;
        }
    }

    return 0;
}



/* Function for testing the linear interpolation class */
double linear_interpolation(double xi, int N, double timestep, const double *y, const int mode)
{
    assert(mode == 0 || mode == 1);
    std::vector< double > intvals;
    intvals = carray_to_vector(y, N);
    /*
    for(int ii=0; ii<N; ii++){
        std::cout << intvals[ii] << " ";
    }
    std::cout << std::endl;
    */
    linear_interpolator li(timestep, intvals, mode);
    return li(xi);
}
