// n2p2 - A neural network potential package
// Copyright (C) 2018 Andreas Singraber (University of Vienna)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#ifndef MOLECULARDYNAMICS_H
#define MOLECULARDYNAMICS_H

#include "Updater.h"
#include <cstddef> // std::size_t
#include <string>  // std::string
#include <vector>  // std::vector
#include <random>  // std::default_random_engine & std::normal_distribution

namespace nnp
{

/// Weight updates based on artificial MD in weight parameter space.
class MolecularDynamics : public Updater
{
public:
    /// Enumerate different gradient descent variants.
    enum DynamicsType
    {
        /// Verlet algorithm.
        DT_VERLET,
        /// Velocity Verlet algorithm.
        DT_VELOCITYVERLET,
        /// Langevin VRORV algorithm.
        DT_LANGEVIN
    };

    /** %MolecularDynamics class constructor.
     *
     * @param[in] sizeState Number of neural network connections (weights
     *                      and biases).
     * @param[in] type Dynamics type used.
     */
    MolecularDynamics(std::size_t const sizeState, DynamicsType const type);
    /** Destructor
     */
    virtual ~MolecularDynamics() {};
    /** Set pointer to current state.
     *
     * @param[in,out] state Pointer to state vector (weights vector), will be
     *                      changed in-place upon calling update().
     */
    void                     setState(double* state);
    /** Set pointer to current error vector.
     *
     * @param[in] error Pointer to error (difference between reference and
     *                  neural network potential output).
     * @param[in] size Number of error vector entries.
     */
    void                     setError(double const* const error,
                                      std::size_t const   size = 1);
    /** Set pointer to current Jacobi matrix.
     *
     * @param[in] jacobian Derivatives of error with respect to weights.
     * @param[in] columns Number of gradients provided.
     *
     * @note
     * If there are @f$m@f$ errors and @f$n@f$ weights, the Jacobi matrix
     * is a @f$n \times m@f$ matrix stored in column-major order.
     */
    void                     setJacobian(double const* const jacobian,
                                         std::size_t const   columns = 1);
    /** Pre-update step required by some MD integrators.
     */
    void                     preUpdateMD();
    /** Perform connection update.
     *
     * Update the connections via steepest descent method.
     */
    void                     update();
    /** Set parameters for Verlet algorithm.
     *
     * @param[in] dt Time step size for Verlet algorithm.
     * @param[in] m Mass of artificial weight particles.
     */
    void                     setParametersVerlet(double const dt,
                                                 double const m);
    /** Set parameters for Velocity Verlet algorithm.
     *
     * @param[in] dt Time step size for Velocity Verlet algorithm.
     * @param[in] m Mass of artificial weight particles.
     */
    void                     setParametersVelocityVerlet(double const dt,
                                                         double const m);
    /** Set parameters for Langevin VRORV algorithm.
     *
     * @param[in] dt Time step size for Langevin VRORV algorithm.
     * @param[in] m Mass of artificial weight particles.
     * @param[in] gamma Collision rate of particles with heat bath.
     * @param[in] T Temperature of heat bath.
     */
    void                     setParametersLangevin(double const dt,
                                                   double const m,
                                                   double const gamma,
                                                   double const T);

    /** Status report.
     *
     * @param[in] epoch Current epoch.
     *
     * @return Line with current status information.
     */
    std::string              status(std::size_t epoch) const;
    /** Header for status report file.
     *
     * @return Vector with header lines.
     */
    std::vector<std::string> statusHeader() const;
    /** Information about gradient descent settings.
     *
     * @return Vector with info lines.
     */
    std::vector<std::string> info() const;

private:
    /// Current integration step.
    std::size_t         n;
    /// Selected dynamics type
    DynamicsType        type;
    /// Timestep for user.
    double              dt;
    /// Mass.
    double              m;
    /// Collision rate.
    double              gamma;
    /// Temperature of heat bath.
    double              T;
    /// Constant for velocity in O-step.
    double              a1;
    /// Constant for random number in O-step.
    double              b1;
    /// Pointer to previous state.
    std::vector<double> state_prev;
    /// State vector pointer.
    double*             state;
    /// Dummy for caching actual state.
    double              state_dummy;
    /// Velocity vector.
    std::vector<double> velo;
    /// Random number generator.
    std::default_random_engine generator;
    /// Standard normal distribution.
    std::normal_distribution<double> distribution;
    /// Error pointer (single double value).
    double const*       error;
    /// Gradient vector pointer.
    double const*       gradient;
};

}

#endif
