#ifndef EXAMPLE_NNP_TRAIN_H
#define EXAMPLE_NNP_TRAIN_H

#include "Example_nnp.h"
#include "BoostDataContainer.h"

struct Example_nnp_train : public Example_nnp
{
    std::size_t lastEpoch;
    double      rmseEnergyTrain;
    double      rmseEnergyTest;
    double      rmseForcesTrain;
    double      rmseForcesTest;

    Example_nnp_train(std::string name) : Example_nnp("nnp-train", name) {};
};

template<>
void BoostDataContainer<Example_nnp_train>::setup()
{
    Example_nnp_train* e = nullptr;

    examples.push_back(Example_nnp_train("LJ"));
    e = &(examples.back());
    e->lastEpoch = 10;
    e->rmseEnergyTrain = 6.52345439E-04;
    e->rmseEnergyTest  = 7.11819625E-04;
    e->rmseForcesTrain = 2.09901211E-02;
    e->rmseForcesTest  = 1.57340602E-02;

    //examples.push_back(Example_nnp_train("H2O_RPBE-D3"));
    //e = &(examples.back());
    //e->lastEpoch = 10;
    //e->rmseEnergyTrain = 1.76506540E-04;
    //e->rmseEnergyTest  = 3.22755877E-04;
    //e->rmseForcesTrain = 1.24590860E-02;
    //e->rmseForcesTest  = 8.59577651E-03;

    return;
}

#endif
