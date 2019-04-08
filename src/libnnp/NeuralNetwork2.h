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

#ifndef NEURALNETWORK2_H
#define NEURALNETWORK2_H

#include <vector>
#include <Eigen/Core>

namespace nnp
{

/// This class implements a feed-forward neural network.
class NeuralNetwork2
{
public:
    /// List of available activation function types.
    enum ActivationType
    {
        /// @f$f_a(x) = x@f$
        AT_IDENTITY,
        /// @f$f_a(x) = \tanh(x)@f$
        AT_TANH,
        /// @f$f_a(x) = 1 / (1 + \mathrm{e}^{-x})@f$
        AT_LOGISTIC,
        /// @f$f_a(x) = \ln (1 + \mathrm{e}^x)@f$
        AT_SOFTPLUS,
        /// @f$f_a(x) = \max(0, x)@f$
        AT_RELU
    };

    /// All neurons in contiguous memory.
    Eigen::VectorXd neurons;
    /// All connections in contiguous memory.
    Eigen::VectorXd connections;

    /** Constructor, requires neural network topology.
     *
     * @param[in] neuronsPerLayer Number of neurons per layer.
     * @param[in] activationPerLayer Activation function type per layer.
     */
    NeuralNetwork2(std::vector<std::size_t>    neuronsPerLayer,
                   std::vector<ActivationType> activationPerLayer);
};

}

#endif
