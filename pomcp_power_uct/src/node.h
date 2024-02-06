#ifndef NODE_H
#define NODE_H

#include "beliefstate.h"
#include "utils.h"
#include <iostream>

class HISTORY;
class SIMULATOR;
class QNODE;
class VNODE;

//-----------------------------------------------------------------------------
// Efficient computation of value from alpha vectors
// Only used for explicit POMDPs
struct ALPHA
{
    std::vector<double> AlphaSum;
    double MaxValue;
};

//-----------------------------------------------------------------------------

template<class COUNT>
class VALUE
{
public:

    void Set(double count, double value, double std=0)
    {
        Count = count;
        Total = value * count;
        std = std;
    }

    void Add(double totalReward)
    {
        Count += 1.0;
        Total += totalReward;
    }

    void Add(double totalReward, COUNT weight)
    {
        Count += weight;
        Total += totalReward * weight;
    }

    double GetValue() const
    {
        return Count == 0 ? Total : Total / Count;
    }

    COUNT GetCount() const
    {
        return Count;
    }

private:

    COUNT Count;
    double Total;
};

//-----------------------------------------------------------------------------

class QNODE
{
public:

    VALUE<int> Value;
    VALUE<double> AMAF;

    void Initialise(const UTILS::PARAMS params);

    VNODE*& Child(int c) { return Children[c]; }
    VNODE* Child(int c) const { return Children[c]; }
    ALPHA& Alpha() { return AlphaData; }
    const ALPHA& Alpha() const { return AlphaData; }


    void DisplayValue(HISTORY& history, int maxDepth, std::ostream& ostr) const;
    void DisplayPolicy(HISTORY& history, int maxDepth, std::ostream& ostr) const;

    VNODE* get_parent() const { return this->Parent; }
    void set_parent(VNODE* parent);
    double get_value();
    void set_value(double q);
    void update(double reward);
    double selection_strategy(double exploration_const);

    static int NumChildren;
    double q = 0.0;
    double reward = 0.0;
    double exploration_const = 1.0;
    int p = 1;//default is average mean

    std::vector<VNODE*> Children;

private:
    VNODE* Parent;
    ALPHA AlphaData;

friend class VNODE;
};

//-----------------------------------------------------------------------------

class VNODE : public MEMORY_OBJECT
{
public:

    VALUE<int> Value;

    void Initialise(const UTILS::PARAMS params);
    static VNODE* Create(const UTILS::PARAMS params);
    static void Free(VNODE* vnode, const SIMULATOR& simulator);
    static void FreeAll();

    QNODE& Child(int c) { return Children[c]; }
    const QNODE& Child(int c) const { return Children[c]; }
    BELIEF_STATE& Beliefs() { return BeliefState; }
    const BELIEF_STATE& Beliefs() const { return BeliefState; }

    void SetChildren(int count, double value);

    void DisplayValue(HISTORY& history, int maxDepth, std::ostream& ostr) const;
    void DisplayPolicy(HISTORY& history, int maxDepth, std::ostream& ostr) const;

    static int NumChildren;

    int sampling_rule(double exploration_const);

    std::vector<QNODE> Children;

private:
    BELIEF_STATE BeliefState;
    static MEMORY_POOL<VNODE> VNodePool;
};

#endif // NODE_H
