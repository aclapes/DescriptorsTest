//
//  xtmath.cpp
//  descriptors_test
//
//  Created by Albert Clapés on 06/10/14.
//
//

#include "xtmath.h"
#include <limits>

template<typename T>
void xtl::sum(std::vector<std::vector<std::vector<T> > > vw, std::vector<std::vector<T> >& w)
{
    w.clear();
    for (int m = 0; m < vw.size(); m++)
    {
        if (w.size() == 0) w.resize(vw[m].size());
        for (int i = 0; i < vw[m].size(); i++)
        {
            if (w[i].size() == 0) w[i].resize(vw[m][i].size(), 0);
            for (int j = 0; j < vw[m][i].size(); j++)
            {
                w[i][j] += vw[m][i][j];
            }
        }
    }
}

template<typename T>
void xtl::avg(std::vector<std::vector<std::vector<T> > > vw, std::vector<std::vector<T> >& w)
{
    w.clear();
    
    xtl::sum(vw, w);
    
    for (int i = 0; i < w.size(); i++) for (int j = 0; j < w[i].size(); j++)
    {
        w[i][j] /= vw.size();
    }
}

template<typename T>
void xtl::collapse(std::vector<std::vector<T> > w, int dim, std::vector<float>& v, int operation)
{
    // if dim == 0, then collapse rows into single row
    // else collapse cols into a single col
    v.clear();
    v.resize((dim == 0) ? w[0].size() : w.size(), 0);
    
    for (int i = 0; i < ((dim == 0) ? w[0].size() : w.size()); i++)
    {
        float sum = 0.f;
        for (int j = 0; j < ((dim == 0) ? w.size(): w[0].size()); j++)
        {
            sum += ((dim == 0) ? w[j][i] : w[i][j]);
        }
        
        if (operation == xtl::COLLAPSE_SUM)
            v[i] = sum;
        else if (operation == xtl::COLLAPSE_AVG)
            v[i] = sum / ((dim == 0) ? w.size() : w[0].size());
    }
}

template<typename T>
T xtl::mean(std::vector<T> v)
{
    float sum = 0.f;
    
    for (int i = 0; i < v.size(); i++)
        sum += v[i];
    
    
    return sum / v.size();
}

template<typename T>
T xtl::stddev(std::vector<T> v)
{
    float u = xtl::mean(v);
    
    float sumsqdiffs = 0.f;
    for (int i = 0; i < v.size(); i++)
        sumsqdiffs += powf(v[i] - u, 2);
    
    return sqrtf(sumsqdiffs/v.size());
}

template<typename T>
std::pair<int,T> xtl::max(std::vector<T> v)
{
    std::pair<int,T> max;
    max.second = -std::numeric_limits<T>::max();
    for (int i = 0; i < v.size(); i++)
    {
        if (v[i] > max.second)
        {
            max.first = i;
            max.second = v[i];
        }
    }
    
    return max;
}

template<typename T>
std::pair<int,T> xtl::min(std::vector<T> v)
{
    std::pair<int,T> min;
    min.second = std::numeric_limits<T>::max();
    for (int i = 0; i < v.size(); i++)
    {
        if (v[i] < min.second)
        {
            min.first = i;
            min.second = v[i];
        }
    }
    
    return min;
}


//
// Template instantiation
//

template void xtl::sum(std::vector<std::vector<std::vector<int> > > vw, std::vector<std::vector<int> >& w);
template void xtl::sum(std::vector<std::vector<std::vector<float> > > vw, std::vector<std::vector<float> >& w);
template void xtl::sum(std::vector<std::vector<std::vector<double> > > vw, std::vector<std::vector<double> >& w);

template void xtl::avg(std::vector<std::vector<std::vector<int> > > vw, std::vector<std::vector<int> >& w);
template void xtl::avg(std::vector<std::vector<std::vector<float> > > vw, std::vector<std::vector<float> >& w);
template void xtl::avg(std::vector<std::vector<std::vector<double> > > vw, std::vector<std::vector<double> >& w);

template void xtl::collapse(std::vector<std::vector<int> > w, int dim, std::vector<float>& v, int operation);
template void xtl::collapse(std::vector<std::vector<float> > w, int dim, std::vector<float>& v, int operation);

template float xtl::mean(std::vector<float> v);
template double xtl::mean(std::vector<double> v);

template float xtl::stddev(std::vector<float> v);
template double xtl::stddev(std::vector<double> v);

template std::pair<int,int> xtl::max(std::vector<int> v);
template std::pair<int,float> xtl::max(std::vector<float> v);

template std::pair<int,int> xtl::min(std::vector<int> v);
template std::pair<int,float> xtl::min(std::vector<float> v);
