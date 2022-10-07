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

#include "Dataset.h"
#include "mpi-extra.h"
#include "utility.h"
#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace std;
using namespace nnp;

int main(int argc, char* argv[])
{
    bool                normalize       = false;
    int                 numProcs        = 0;
    int                 myRank          = 0;
    string              fileName;
    ofstream            fileLatentData;
    ofstream            myLog;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    Dataset dataset;
    if (myRank != 0) dataset.log.writeToStdout = false;
    myLog.open(strpr("nnp-latent.log.%04d", myRank).c_str());
    dataset.log.registerStreamPointer(&myLog);
    dataset.setupMPI();
    dataset.initialize();
    dataset.loadSettingsFile();
    dataset.setupGeneric();
    normalize = dataset.useNormalization();
    dataset.setupSymmetryFunctionScaling();
    dataset.setupSymmetryFunctionStatistics(false, false, true, false);
    dataset.setupNeuralNetworkWeights();
    dataset.distributeStructures(false);
    if (normalize) dataset.toNormalizedUnits();

    dataset.log << "\n";
    dataset.log << "*** LATENT SPACE DATA COLLECTION ********"
                   "**************************************\n";
    dataset.log << "\n";

    // Open latent space data file.
    fileName = strpr("latent-space.data.%04d", myRank);
    fileLatentData.open(fileName.c_str());
    if (myRank == 0)
    {
        // File header.
        vector<string> title;
        vector<string> colName;
        vector<string> colInfo;
        vector<size_t> colSize;
        title.push_back("Latent space data (neuron values from layer before "
                        "the output neuron).");
        colSize.push_back(10);
        colName.push_back("index_s");
        colInfo.push_back("Structure index.");
        colSize.push_back(10);
        colName.push_back("index_a");
        colInfo.push_back("Atom index.");
        colSize.push_back(3);
        colName.push_back("ie");
        colInfo.push_back("Element index of atom.");
        colSize.push_back(2);
        colName.push_back("e");
        colInfo.push_back("Element string of atom.");
        colSize.push_back(6);
        colName.push_back("in");
        colInfo.push_back("Neuron number.");
        colSize.push_back(24);
        colName.push_back("neuron_value");
        colInfo.push_back("Neural value.");
        appendLinesToFile(fileLatentData,
                          createFileHeader(title, colSize, colName, colInfo));
    }

    for (vector<Structure>::iterator it = dataset.structures.begin();
         it != dataset.structures.end(); ++it)
    {
        it->calculateNeighborList(dataset.getMaxCutoffRadius());
#ifdef N2P2_NO_SF_GROUPS
        dataset.calculateSymmetryFunctions((*it), false);
#else
        dataset.calculateSymmetryFunctionGroups((*it), false);
#endif
        auto neuron_values = dataset.calculateLatentSpace((*it));
        // Write latent space data.
        for (size_t i = 0; i < it->atoms.size(); ++i)
        {
            for (size_t j = 0; j < neuron_values.at(i).size(); ++j)
            {
                fileLatentData << strpr("%10zu %10zu %3zu %2s %6zu %24.16E\n",
                                        it->index + 1,
                                        it->atoms.at(i).index + 1,
                                        it->atoms.at(i).element,
                                        dataset.elementMap[
                                            it->atoms.at(i).element].c_str(),
                                        j,
                                        neuron_values.at(i).at(j));
            }
        }
    }

    fileLatentData.close();
    MPI_Barrier(MPI_COMM_WORLD);

    if (myRank == 0)
    {
        fileName = "latent-space.data";
        dataset.combineFiles(fileName);
    }

    if (myRank == 0)
    {
        dataset.log << "Latent space data written to \"latent-space.data\"\n";
    }

    dataset.log << "*****************************************"
                   "**************************************\n";

    myLog.close();

    MPI_Finalize();

    return 0;
}
