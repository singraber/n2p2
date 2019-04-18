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

#include "MolecularDynamics.h"
#include "utility.h"
#include <cstddef>
#include <cmath>

using namespace std;
using namespace nnp;

MolecularDynamics::MolecularDynamics(size_t const       sizeState,
                                     DynamicsType const type) :
    Updater (sizeState),
    n       (0          ),
    type    (type),
    dt      (0.0        ),
    m       (0.0        ),
    state   (NULL       ),
    error   (NULL       ),
    gradient(NULL       )
{
    if (type != DT_VERLET)
    {
        throw runtime_error("ERROR: Unknown MolecularDynamics type.\n");
    }

    if (sizeState < 1)
    {
        throw runtime_error("ERROR: Wrong MolecularDynamics dimensions.\n");
    }

    if (type == DT_VERLET)
    {
        state_prev.resize(sizeState, 0.0);
    }
}

void MolecularDynamics::setState(double* state)
{
    this->state = state;

    if (n == 0)
    {
        for(size_t i = 0; i < sizeState; ++i)
        {
            state_prev[i] = state[i];
        }
    }

    return;
}

void MolecularDynamics::setError(double const* const error,
                                 size_t const /* size */)
{
    this->error = error;

    return;
}

void MolecularDynamics::setJacobian(double const* const jacobian,
                                    size_t const /* columns*/)
{
    this->gradient = jacobian;

    return;
}

void MolecularDynamics::preUpdateMD()
{
    return;
}

void MolecularDynamics::update()
{
    if (type == DT_VERLET)
    {
        for (std::size_t i = 0; i < sizeState; ++i)
        {
            // Cache the actual state at t
            state_dummy = state[i];

            // Verlet: generating new state at (t + dt)
            state[i] = 2 * state[i] - state_prev[i] - gradient[i] * dt * dt / m;

            // Set the cached state as the previous state at (t - dt)
            state_prev[i] = state_dummy;
        }
    }

    ++n;

    return;
}

void MolecularDynamics::setParametersVerlet(double const dt, double const m)
{
    this->dt = dt;
    this->m = m;

    return;
}

string MolecularDynamics::status(size_t epoch) const
{
    string s = strpr("%10zu %10zu %16.8E", epoch, n, error[0]);

    s += '\n';

    return s;
}

vector<string> MolecularDynamics::statusHeader() const
{
    vector<string> header;

    vector<string> title;
    vector<string> colName;
    vector<string> colInfo;
    vector<size_t> colSize;
    title.push_back("Molecular dynamics status report.");
    colSize.push_back(10);
    colName.push_back("epoch");
    colInfo.push_back("Training epoch.");
    colSize.push_back(10);
    colName.push_back("n");
    colInfo.push_back("Integration step.");
    colSize.push_back(16);
    colName.push_back("loss");
    colInfo.push_back("Loss function: MSE of energies (and forces).");

    header = createFileHeader(title, colSize, colName, colInfo);

    return header;
}

vector<string> MolecularDynamics::info() const
{
    vector<string> v;

    if (type == DT_VERLET)
    {
        v.push_back(strpr("MolecularDynamicsType::DT_VERLET (%d)\n", type));
        v.push_back(strpr("sizeState       = %zu\n", sizeState));
        v.push_back(strpr("dt              = %12.4E\n", dt));
        v.push_back(strpr("m               = %12.4E\n", m));
    }

    return v;
}
