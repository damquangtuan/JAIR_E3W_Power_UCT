#include "simulator.h"
#include "rocksample.h"
using namespace std;
using namespace UTILS;

SIMULATOR::KNOWLEDGE::KNOWLEDGE()
    : TreeLevel(PURE),
    RolloutLevel(PURE),
    SmartTreeCount(10),
    SmartTreeValue(1.0)
{
}

SIMULATOR::STATUS::STATUS()
    : Phase(TREE),
    Particles(CONSISTENT)
{
}

SIMULATOR::SIMULATOR()
    : Discount(1.0),
    NumActions(0),
    NumObservations(0),
    RewardRange(1.0)
{
}

SIMULATOR::SIMULATOR(int numActions, int numObservations, double discount)
    : NumActions(numActions),
    NumObservations(numObservations),
    Discount(discount)
{
    assert(discount > 0 && discount <= 1);

   
}

SIMULATOR::~SIMULATOR()
{
}

void SIMULATOR::Validate(const STATE& state) const
{
}

float SIMULATOR::value(STATE& state, float* data) const
{
    return 0.0;
}

float SIMULATOR::value2(STATE& state, int action) const
{
    return 0.0;
}


bool SIMULATOR::LocalMove(STATE& state, const HISTORY& history,
    int stepObs, const STATUS& status) const
{
    return true;
}

void SIMULATOR::GenerateLegal(const STATE& state, const HISTORY& history,
    std::vector<int>& actions, const STATUS& status) const
{
    for (int a = 0; a < NumActions; ++a)
        actions.push_back(a);
}

void SIMULATOR::GeneratePreferred(const STATE& state, const HISTORY& history,
    std::vector<int>& actions, const STATUS& status) const
{
}

int SIMULATOR::SelectRandom(const STATE& state, const HISTORY& history,
    const STATUS& status) const
{
    static vector<int> actions;

    if (Knowledge.RolloutLevel >= KNOWLEDGE::SMART)
    {
        actions.clear();
        GeneratePreferred(state, history, actions, status);
        if (!actions.empty())
            return actions[Random(actions.size())];
    }

    if (Knowledge.RolloutLevel >= KNOWLEDGE::LEGAL)
    {
        actions.clear();
        GenerateLegal(state, history, actions, status);
        if (!actions.empty())
            return actions[Random(actions.size())];
    }

    return Random(NumActions);
}

void SIMULATOR::Prior(const STATE* state, const HISTORY& history,
    VNODE* vnode, const STATUS& status) const
{
    static vector<int> actions;

    if (Knowledge.TreeLevel == KNOWLEDGE::PURE || state == 0)
    {
        vnode->SetChildren(0, 0);
        return;
    }
    else
    {
        vnode->SetChildren(+LargeInteger, -Infinity);
    }

    if (Knowledge.TreeLevel >= KNOWLEDGE::LEGAL)
    {
        actions.clear();
        GenerateLegal(*state, history, actions, status);

        for (vector<int>::const_iterator i_action = actions.begin(); i_action != actions.end(); ++i_action)
        {
            int a = *i_action;
            QNODE& qnode = vnode->Child(a);
            qnode.Value.Set(0, 0, UTILS::getInitStd());

        }
    }

    if (Knowledge.TreeLevel >= KNOWLEDGE::SMART)
    {
        actions.clear();
        GeneratePreferred(*state, history, actions, status);

        for (vector<int>::const_iterator i_action = actions.begin(); i_action != actions.end(); ++i_action)
        {
            int a = *i_action;
            QNODE& qnode = vnode->Child(a);
            
           
           
            qnode.Value.Set(0.0, 0.0, getInitStd());
           

        }
    }
}

void SIMULATOR::saveVisits() const {

}

bool SIMULATOR::HasAlpha() const
{
    return false;
}

void SIMULATOR::AlphaValue(const QNODE& qnode, double& q, int& n) const
{
}

void SIMULATOR::UpdateAlpha(QNODE& qnode, const STATE& state) const
{
}

void SIMULATOR::DisplayBeliefs(const BELIEF_STATE& beliefState,
    ostream& ostr) const
{
}

void SIMULATOR::DisplayState(const STATE& state, ostream& ostr) const
{
}

void SIMULATOR::DisplayAction(int action, ostream& ostr) const
{
    ostr << "Action " << action << endl;
}

void SIMULATOR::DisplayObservation(const STATE& state, int observation, ostream& ostr) const
{
    ostr << "Observation " << observation << endl;
}

void SIMULATOR::DisplayReward(double reward, std::ostream& ostr) const
{
    ostr << "Reward " << reward << endl;
}

double SIMULATOR::GetHorizon(double accuracy, int undiscountedHorizon) const
{
    if (Discount == 1)
        return undiscountedHorizon;
    return log(accuracy) / log(Discount);
}

double SIMULATOR::getNNPrediction(STATE* state, int action) const {
    return 0.0;
}


std::vector<double> SIMULATOR::f_lake_priors = { 0.10944165, 0.08072938, 0.07125261, 0.08479619,
                       0.04184463, 0.0453524,  0.03531314, 0.08886708,
                       0.06795874, 0.04894549, 0.03939226, 0.07444809,
                       0.05747822, 0.0281393,  0.01907414, 0.02643916,
                       0.11198432, 0.06483005, 0.06179734, 0.05962203,
                       0., 0., 0., 0.,
                       0.09109917, 0.08654694 ,0.09083652, 0.09102637,
                       0., 0., 0., 0.,
                       0.08130991, 0.10635473, 0.0985738,  0.16939236,
                       0.208456  , 0.27315834, 0.22843903 ,0.10645811,
                       0.22748257, 0.29938 ,   0.25959805 ,0.13080376,
                       0. ,0., 0., 0.,
                       0., 0., 0., 0.,
                       0.18530077, 0.33003727, 0.39165398, 0.24647015,
                       0.3350309 , 0.56958294, 0.6181232 , 0.5360135,
                       0.29863796, 0.6018661,  0.6463813,  0.5730382 };