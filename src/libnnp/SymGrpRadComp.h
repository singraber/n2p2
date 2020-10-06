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

#ifndef SYMGRPRADCOMP_H
#define SYMGRPRADCOMP_H

#include "SymGrp.h"
#include <cstddef> // std::size_t
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

struct Atom;
class ElementMap;
class SymFnc;
class SymFncRadComp;

/** Radial symmetry function group (type 2)
 *
 * @f[
 * G^2_i = \sum_{j \neq i} \mathrm{e}^{-\eta(r_{ij} - r_\mathrm{s})^2}
 *         f_c(r_{ij}) 
 * @f]
 * Common features:
 * - element of central atom
 * - element of neighbor atom
 * - cutoff type
 * - @f$r_c@f$
 * - @f$\alpha@f$
 */
class SymGrpRadComp : public SymGrp
{
public:
    /** Constructor, sets type = 2
     */
    SymGrpRadComp(ElementMap const& elementMap);
    /** Overload == operator.
     */
    bool operator==(SymGrp const& rhs) const;
    /** Overload != operator.
     */
    bool operator!=(SymGrp const& rhs) const;
    /** Overload < operator.
     */
    bool operator<(SymGrp const& rhs) const;
    /** Overload > operator.
     */
    bool operator>(SymGrp const& rhs) const;
    /** Overload <= operator.
     */
    bool operator<=(SymGrp const& rhs) const;
    /** Overload >= operator.
     */
    bool operator>=(SymGrp const& rhs) const;
    /** Potentially add a member to group.
     *
     * @param[in] symmetryFunction Candidate symmetry function.
     * @return If addition was successful.
     *
     * If symmetry function is compatible with common feature list its pointer
     * will be added to #members.
     */
    bool addMember(SymFnc const* const symmetryFunction);
    /** Sort member symmetry functions.
     *
     * Also allocate and precalculate additional stuff.
     */
    void sortMembers();
    /** Fill #scalingFactors with values from member symmetry functions.
     */
    void setScalingFactors();
    /** Calculate all symmetry functions of this group for one atom.
     *
     * @param[in,out] atom Atom for which symmetry functions are caluclated.
     * @param[in] derivatives If also symmetry function derivatives will be
     *                        calculated and saved.
     */
    void calculate(Atom& atom, bool const derivatives) const;
    /** Give symmetry function group parameters on multiple lines.
     *
     * @return Vector of string containing symmetry function parameters lines.
     */
    std::vector<std::string>
         parameterLines() const;

private:
    /// Element index of neighbor atom (common feature).
    std::size_t                       e1;
    /// Vector of all group member pointers.
    std::vector<SymFncRadComp const*> members;
    /// Minimum radius within group
    double                            rl;
};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline bool SymGrpRadComp::operator!=(SymGrp const& rhs) const
{
    return !((*this) == rhs);
}

inline bool SymGrpRadComp::operator>(SymGrp const& rhs) const
{
    return rhs < (*this);
}

inline bool SymGrpRadComp::operator<=(SymGrp const& rhs) const
{
    return !((*this) > rhs);
}

inline bool SymGrpRadComp::operator>=(SymGrp const& rhs) const
{
    return !((*this) < rhs);
}

}

#endif