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

#include "SymFncBaseComp.h"
#include "utility.h"
#include <string>

using namespace std;
using namespace nnp;

vector<string> SymFncBaseComp::parameterInfo() const
{
    vector<string> v = SymFnc::parameterInfo();
    string s;
    size_t w = sfinfoWidth;

    s = "subtype";
    v.push_back(strpr((pad(s, w) + "%s"    ).c_str(), subtype.c_str()));
    s = "rl";
    v.push_back(strpr((pad(s, w) + "%14.8E").c_str(), rl / convLength));

    return v;
}

void SymFncBaseComp::setCompactFunction(string subtype)
{
    if (subtype.size() < 2 || subtype.size() > 3)
    {
        throw runtime_error(strpr("ERROR: Invalid compact function type "
                                  "specification: \"%s\".\n",
                                  subtype.c_str()));
    }

    // Check for polynomials.
    if (subtype.front() == 'p')
    {
        using CFT = CoreFunction::Type;
        if      (subtype.at(1) == '1') cr.setCoreFunction(CFT::POLY1);
        else if (subtype.at(1) == '2') cr.setCoreFunction(CFT::POLY2);
        else if (subtype.at(1) == '3') cr.setCoreFunction(CFT::POLY3);
        else if (subtype.at(1) == '4') cr.setCoreFunction(CFT::POLY4);
        else
        {
            throw runtime_error(strpr("ERROR: Invalid polynom type: \"%s\".\n",
                                      subtype.c_str()));
        }
        if (subtype.size() == 3 && subtype.at(2) == 'a') asymmetric = true;
        else
        {
            throw runtime_error(strpr("ERROR: Invalid polynom specifier: "
                                      "\"%s\".\n", subtype.c_str()));
        }
    }

    return;
}

SymFncBaseComp::SymFncBaseComp(size_t type,
                               ElementMap const& elementMap) :
    SymFnc(type, elementMap),
    asymmetric(false),
    rl        (0.0),
    subtype   ("p2")
{
    // Add polynomial-related parameter IDs to set.
    parameters.insert("rs/rl");
    parameters.insert("subtype");
}