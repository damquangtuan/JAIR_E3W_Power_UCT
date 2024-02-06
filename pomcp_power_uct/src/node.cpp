#include "node.h"
#include "history.h"
#include "utils.h"

using namespace std;

//-----------------------------------------------------------------------------

int QNODE::NumChildren = 0;

void QNODE::Initialise(const UTILS::PARAMS params)
{
    assert(NumChildren);
    Children.resize(NumChildren);
    for (int observation = 0; observation < QNODE::NumChildren; observation++)
        Children[observation] = 0;
    AlphaData.AlphaSum.clear();

    this->exploration_const = params.ExplorationConstant;
    this->p = params.Power;
}

void QNODE::DisplayValue(HISTORY& history, int maxDepth, ostream& ostr) const
{
    history.Display(ostr);
    ostr << ": " << Value.GetValue() << " (" << Value.GetCount() << ")\n";
    if (history.Size() >= maxDepth)
        return;

    for (int observation = 0; observation < NumChildren; observation++)
    {
        if (Children[observation])
        {
            history.Back().Observation = observation;
            Children[observation]->DisplayValue(history, maxDepth, ostr);
        }
    }
}

void QNODE::DisplayPolicy(HISTORY& history, int maxDepth, ostream& ostr) const
{
    history.Display(ostr);
    ostr << ": " << Value.GetValue() << " (" << Value.GetCount() << ")\n";
    if (history.Size() >= maxDepth)
        return;

    for (int observation = 0; observation < NumChildren; observation++)
    {
        if (Children[observation])
        {
            history.Back().Observation = observation;
            Children[observation]->DisplayPolicy(history, maxDepth, ostr);
        }
    }
}

void QNODE::set_parent(VNODE* parent) {
    this->Parent = parent;
}

double QNODE::get_value() {
    return this->q;
}
void QNODE::set_value(double q) {
    this->q = q;
}

void QNODE::update(double reward) {
    this->reward += reward;
}

double QNODE::selection_strategy(double exploration_const) {
    if (this->Value.GetCount() == 0) {
        return +INFINITY;
    } else {
        double exploration_bonus = exploration_const * sqrt(log((this->Parent)->Value.GetCount()) / this->Value.GetCount());
        return this->Value.GetValue() + exploration_bonus;
    }
}

//-----------------------------------------------------------------------------

MEMORY_POOL<VNODE> VNODE::VNodePool;

int VNODE::NumChildren = 0;

void VNODE::Initialise(const UTILS::PARAMS params)
{
    assert(NumChildren);
    Children.resize(VNODE::NumChildren);
    for (int action = 0; action < VNODE::NumChildren; action++) {
        Children[action].Initialise(params);
        Children[action].set_parent(this);
    }
}

VNODE* VNODE::Create(const UTILS::PARAMS params)
{
    VNODE* vnode = VNodePool.Allocate();
    vnode->Initialise(params);
    return vnode;
}

void VNODE::Free(VNODE* vnode, const SIMULATOR& simulator)
{
    vnode->BeliefState.Free(simulator);
    VNodePool.Free(vnode);
    for (int action = 0; action < VNODE::NumChildren; action++)
        for (int observation = 0; observation < QNODE::NumChildren; observation++)
            if (vnode->Child(action).Child(observation))
                Free(vnode->Child(action).Child(observation), simulator);
}

void VNODE::FreeAll()
{
	VNodePool.DeleteAll();
}

void VNODE::SetChildren(int count, double value)
{
    for (int action = 0; action < NumChildren; action++)
    {
        QNODE& qnode = Children[action];
        qnode.Value.Set(count, value);
        qnode.AMAF.Set(count, value);
    }
}

void VNODE::DisplayValue(HISTORY& history, int maxDepth, ostream& ostr) const
{
    if (history.Size() >= maxDepth)
        return;

    for (int action = 0; action < NumChildren; action++)
    {
        history.Add(action);
        Children[action].DisplayValue(history, maxDepth, ostr);
        history.Pop();
    }
}

void VNODE::DisplayPolicy(HISTORY& history, int maxDepth, ostream& ostr) const
{
    if (history.Size() >= maxDepth)
        return;

    double bestq = -Infinity;
    int besta = -1;
    for (int action = 0; action < NumChildren; action++)
    {
        if (Children[action].Value.GetValue() > bestq)
        {
            besta = action;
            bestq = Children[action].Value.GetValue();
        }
    }

    if (besta != -1)
    {
        history.Add(besta);
        Children[besta].DisplayPolicy(history, maxDepth, ostr);
        history.Pop();
    }
}

int VNODE::sampling_rule(double exploration_const) {
    std::vector<double> qvalues(this->NumChildren);
    for (int index = 0; index < this->NumChildren; index+=1) {
        qvalues[index] = this->Children[index].selection_strategy(exploration_const);
    }

    return UTILS::arg_max(qvalues);
}

//-----------------------------------------------------------------------------
