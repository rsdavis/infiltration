
#include "model.hpp"

namespace model
{
    /** 
    The model namespace includes variable definitions and data indices.
    It enables the parameters to be used 'globally' within this model file.
    These variables are referenced using model::variable.
    */

    int phi0; 
    int phi1;
    int phi2;

    double dx;
    double dt;

    double A0;
    double A1;
    double A2;
    double D;

    double Wp0;
    double Wp1;
    double Wp2;
}


/** model-specific functions can be included here */


void preprocess(double ** phase,  // order parameter data
                int * dims,       // system dimensions
                std::map<std::string, std::string> params, // input file parameters
                std::map<std::string, int> phase_index)     // phase indices
{

    /**
    Preprocess is only called once, before the time-stepping begins.
    It is used primarily for unpacking parameters from the input file and
    getting the index to access each order parameter.
    Preprocess can also be used to allocate additional data storage and 
    calculate model-specific data that will be constant throughout the simulation.
    */

    // unpack phase index

    unpack(phase_index, "phi0", model::phi0);
    unpack(phase_index, "phi1", model::phi1);
    unpack(phase_index, "phi2", model::phi2);

    // unpack model parameters

    unpack(params, "dx", model::dx);
    unpack(params, "dt", model::dt);

    unpack(params, "A0", model::A0);
    unpack(params, "A1", model::A1);
    unpack(params, "A2", model::A2);
    unpack(params, "D", model::D);

    unpack(params, "Wp0", model::Wp0);
    unpack(params, "Wp1", model::Wp1);
    unpack(params, "Wp2", model::Wp2);
}

void kernel(double ** phase, double ** chem_pot, double ** mobility, int * dims)
{
    /**
    The kernel is run at every timestep and should include all calculations 
    required by the model in order to integrate the order parameters.

    All of the order parameter data is stored in the phase array.
    It is accessed using the phase index, and the ijk index ("ndx").

    Space is already allocated for storing chemical potential (chem_pot) and
    mobilities (mobility) values for each order parameter. 
    Their use is optional. They are only included for convenience.

    Looping is handled by using the for_loop_ijk(x) macro 
    where "x" is the number of ghost rows to be included in the loop.
    The calc_ijk_index macro calculates an "ndx" which indicates the position
    within the system at every iteration.
    These two macros are required in order to make the implementation independent of 
    system dimensionality.

    The stencil object includes common finite differencing operations. 
    A setup routine must be called in order to determine indexing values.
    Any operations involving neighbor grid points should be implemented in Stencil.

    Don't modify i,j,k variables, they are used for looping.
    */

    Stencil stencil;
    stencil.setup(dims, model::dx);
    double h = 0.036;
    double inv_h = 1.0/h;

    for_loop_ijk(1)
    {
        int ndx = calc_ijk_index();

        double p0 = phase[model::phi0][ndx];
        double p1 = phase[model::phi1][ndx];
        double p2 = phase[model::phi2][ndx];

        double p0sq = p0*p0;
        double p1sq = p1*p1;
        double p2sq = p2*p2;

        double grad_sq_1 = stencil.grad_sq(phase[model::phi1], ndx);

        double tanh_1 = tanh(grad_sq_1*inv_h);

        double laplace_1 = stencil.laplacian_h2(phase[model::phi1], ndx);
        double laplace_2 = stencil.laplacian_h2(phase[model::phi2], ndx);
        
        chem_pot[model::phi1][ndx] = model::A1 * (2*p1-6*p1sq+4*p1sq*p1)
                                    + model::D * (2*p0sq*p1 + 2*p1*p2sq - 2*(1-p0)*(1-p0)*(1-p1)*(1-p2)*(1-p2))
                                    - model::Wp1 * laplace_1;

        mobility[model::phi1][ndx] = tanh_1; 


        chem_pot[model::phi2][ndx] = model::A2 * (2*p2-6*p2sq+4*p2sq*p2)
                                    + model::D * (2*p0sq*p2 + 2*p1sq*p2 - 2*(1-p0)*(1-p0)*(1-p1)*(1-p1)*(1-p2))
                                    - model::Wp2 * laplace_2;
    }

    for_loop_ijk(0)
    {
        int ndx = calc_ijk_index();
        phase[model::phi1][ndx] += model::dt * stencil.div_A_grad_B(mobility[model::phi1], chem_pot[model::phi1], ndx);
        //phase[model::phi1][ndx] += model::dt * stencil.laplacian_h2(chem_pot[model::phi1], ndx);
        phase[model::phi2][ndx] += model::dt * stencil.laplacian_h2(chem_pot[model::phi2], ndx);
    }
}


void postprocess(double ** phase, double ** chem_pot, double ** mobility, int * dims)
{

    /**
    This routine is run only once after the simulation has completed.
    It is primarily used for free'ing resources allocated in the preprocess routine.
    */
}

