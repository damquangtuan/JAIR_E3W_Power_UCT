#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <random>
#include "iostream"
#include "coord.h"
#include "memorypool.h"
#include <algorithm>
#include "chrono"
#include "boost/random.hpp"
#include "boost/random/normal_distribution.hpp"

#define LargeInteger 1000000
#define Infinity 1e+10
#define Tiny 1e-10

#ifdef DEBUG
#define safe_cast dynamic_cast
#else
#define safe_cast static_cast
#endif

using namespace std;

namespace UTILS
{
    struct PARAMS
    {
        PARAMS();

        int Verbose;
        int MaxDepth;
        int NumSimulations;
        int NumStartStates;
        bool UseTransforms;
        int NumTransforms;
        int MaxAttempts;
        int ExpandCount;
        double ExplorationConstant;
        bool UseRave;
        double RaveDiscount;
        double RaveConstant;
        bool DisableTree;
        int NumberAtoms;
        double Power;
    };

    struct data {
        double p;
        double std;
        double C;
        bool useWstein;
        std::string visits_file;
        bool useOS;
        data() {

        }
        data(double p_, double std_, double C_) {
            p = p_;
            std = std_;
            C = C_;
        }
        void setP(double p_) {
            p = p_;
        }
        void setC(double C_) {
            C = C_;
        }
        void setStd(double std_) {
            std = std_;
        }
        void setUseWstein(bool b) {
            useWstein = b;
        }
        void setVisitString(std::string s) {
            visits_file = s;
        }
        void setUseOS(bool b) {
            useOS = b;
        }
    };
    extern data dat;
    inline static bool getUSeWstein() {
        return dat.useWstein;
    }
    inline static double getC() {
        return dat.C;
    }

    inline static double getInitStd() {
        return dat.std;
    }
    inline static double getP() {
        return dat.p;
    }
    inline static std::string getVisitsString() {
        return dat.visits_file;
    }

    static  boost::mt19937 rng(time(0));
    inline double Gaussian(double mean, double std) {
        boost::normal_distribution<> nd(mean, std);
        boost::variate_generator<boost::mt19937&,
                boost::normal_distribution<> > var_nor(rng, nd);
        double q = var_nor();
        return q;
    }
    inline double UniformDist(double min, double max) {
        boost::uniform_real<> ud(min, max);
        boost::variate_generator<boost::mt19937&,
                boost::uniform_real<> > var_nor(rng, ud);
        double q = var_nor();
        return q;
    }

inline int Sign(int x)
{
    return (x > 0) - (x < 0);
}

inline int Random(int max)
{
    return rand() % max;
}

inline int Random(int min, int max)
{
    return rand() % (max - min) + min;
}

inline double RandomDouble(double min, double max)
{
    return (double) rand() / RAND_MAX * (max - min) + min;
}

inline void RandomSeed(int seed)
{
    srand(seed);
}

inline bool Bernoulli(double p)
{
    return rand() < p * RAND_MAX;
}

inline bool Near(double x, double y, double tol)
{
    return fabs(x - y) <= tol;
}

inline bool CheckFlag(int flags, int bit) { return (flags & (1 << bit)) != 0; }

inline void SetFlag(int& flags, int bit) { flags = (flags | (1 << bit)); }

template<class T>
inline bool Contains(std::vector<T>& vec, const T& item)
{
    return std::find(vec.begin(), vec.end(), item) != vec.end();
}

template<typename T>
inline T sample_uniform(const std::vector<T> &values) {
    std::mt19937 gen;
    std::uniform_real_distribution<double> dis;

    uint32_t n = values.size();
    double prob = 1. / ((double) n);
    double v = dis(gen);
    uint32_t count = 0;
    double acc = prob;
    while (v > acc) {
        count += 1;
        acc += prob;
    }

    return count;
}

inline int sample_uniform_indices(const std::vector<double> &probabilities) {
    std::discrete_distribution<int> distribution(cbegin(probabilities), cend(probabilities));

    const int outputSize = 1;

    std::vector<decltype(distribution)::result_type> indices;
    std::default_random_engine generator;
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
    indices.reserve(outputSize); // reserve to prevent reallocation
    // use a generator lambda to draw random indices based on distribution
    std::generate_n(back_inserter(indices), outputSize,
                    [distribution = std::move(distribution), // could also capture by reference (&) or construct in the capture list
                            generator  //pseudo random. Fixed seed! Always same output.
                    ]() mutable { // mutable required for generator
                        return distribution(generator);
                    });
    return indices[0];
}

inline double erf(double x)
{
    double y = 1.0 / ( 1.0 + 0.3275911 * x);
    return 1 - (((((
            + 1.061405429  * y
            - 1.453152027) * y
            + 1.421413741) * y
            - 0.284496736) * y
            + 0.254829592) * y)
            * exp (-x * x);
}

// Probability density function
inline double pdf(const double& x, const double& mu, const double& sigma) {
    //Constants
    static const double pi = 3.14159265;
    return exp( -1 * (x - mu) * (x - mu) / (2 * sigma * sigma)) / (sigma * sqrt(2 * pi));
}

inline double normalCDF(const double& x, const double& mu, const double& sigma) // Phi(-∞, x) aka N(x)
{
    return 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2.))));
}

inline double sampleNormalDistribution(const double& mu, const double& sigma) {
    unsigned seed = chrono::steady_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(mu,sigma);
    return distribution(generator);
}

// Cumulative density function
inline double cdf(const double& x) {
    double k = 1.0/(1.0 + 0.2316419*x);
    double k_sum = k*(0.319381530 + k*(-0.356563782 + k*(1.781477937 + k*(-1.821255978 + 1.330274429*k))));

    if (x >= 0.0) {
        return (1.0 - (1.0/(pow(2*M_PI,0.5)))*exp(-0.5*x*x) * k_sum);
    } else {
        return 1.0 - cdf(-x);
    }
}

// Cumulative density function
inline double cdf(const double& x, const double& mean, const double& std) {
    double X = (x - mean)/std;
    double k = 1.0/(1.0 + 0.2316419*X);
    double k_sum = k*(0.319381530 + k*(-0.356563782 + k*(1.781477937 + k*(-1.821255978 + 1.330274429*k))));

    if (X >= 0.0) {
        return (1.0 - (1.0/std*(pow(2*M_PI,0.5)))*exp(-0.5*X*X) * k_sum);
    } else {
        return 1.0 - cdf(-X);
    }
}

template <typename T, typename A>
inline int arg_max(std::vector<T, A> const& vec) {
    unsigned seed = chrono::steady_clock::now().time_since_epoch().count();
    double bestq = -Infinity;
    vector<int> besta;
    for (int action = 0; action < vec.size(); action++) {
        double q = vec[action];
        if (q >= bestq) {
            if (q > bestq)
                besta.clear();
            bestq = q;
            besta.push_back(action);
        }
    }

    assert(!besta.empty());
    srand(seed);
    return besta[Random(besta.size())];
//    return static_cast<int>(std::distance(vec.begin(), max_element(vec.begin(), vec.end())));
}

template <typename T, typename A>
inline int arg_min(std::vector<T, A> const& vec) {
    unsigned seed = chrono::steady_clock::now().time_since_epoch().count();
    double bestq = +Infinity;
    vector<int> besta;
    for (int action = 0; action < vec.size(); action++) {
        double q = vec[action];
        if (q <= bestq) {
            if (q < bestq)
                besta.clear();
            bestq = q;
            besta.push_back(action);
        }
    }

    assert(!besta.empty());
    RandomSeed(seed);
    return besta[Random(besta.size())];

//    return static_cast<int>(std::distance(vec.begin(), min_element(vec.begin(), vec.end())));
}

template<typename T>
std::vector<double> linspace(T start_in, T end_in, int num_in)
{

    std::vector<double> linspaced;

    double start = static_cast<double>(start_in);
    double end = static_cast<double>(end_in);
    double num = static_cast<double>(num_in);

    if (num == 0) { return linspaced; }
    if (num == 1)
    {
        linspaced.push_back(start);
        return linspaced;
    }

    double delta = (end - start) / (num - 1);

    for(int i=0; i < num-1; ++i)
    {
        linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end); // I want to ensure that start and end
    // are exactly the same as the input
    return linspaced;
}



void UnitTest();

}

#endif // UTILS_H
