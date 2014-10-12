//
//  xtlcommon.cpp
//  descriptors_test
//
//  Created by Albert Clapés on 08/10/14.
//
//

#include "xtlcommon.h"

template<typename T>
void xtl::filterByValue(std::vector<T> u, std::vector<T> v, std::vector<T>& o, std::vector<int>& s, bool bPositive)
{
    std::vector<T> _o;
    _o.reserve(u.size());

    std::vector<int> _s;
    _s.reserve(u.size());
    
    for (int i = 0; i < u.size(); i++)
    {
        bool bFound = false;
        for (int j = 0; j < v.size() && !bFound; j++)
            bFound = (u[i] == v[j]);
        
        if ((bPositive && bFound) || (!bPositive && !bFound))
        {
            _o.push_back(u[i]);
            _s.push_back(i);
        }
    }
    
    o = _o;
    s = _s;
}

template<typename T>
void xtl::filterByValue(std::vector<T> u, std::vector<T> v, std::vector<T>& o, bool bPositive)
{
    std::vector<int> s;
    xtl::filterByValue(u,v,o,s,bPositive);
}

template<typename T>
void xtl::filterByIndex(std::vector<std::vector<T> > u, std::vector<int> x, std::vector<int> y, std::vector<std::vector<T> >& o, bool bPositiveX, bool bPositiveY)
{
    std::vector<std::vector<T> > _o;
    _o.reserve(u.size());
    for (int i = 0; i < u.size(); i++)
    {
        bool bFound = false;
        for (int j = 0; j < x.size() && !bFound; j++)
            bFound = (i == x[j]);
        
        if ((bPositiveX && bFound) || (!bPositiveX && !bFound))
            _o.push_back(u[i]);
    }
    
    for (int i = 0; i < _o.size(); i++)
    {
        xtl::filterByIndex(_o[i], y, _o[i], bPositiveY);
    }
    
    o = _o;
}

template<typename T>
void xtl::filterByIndex(std::vector<T> u, std::vector<int> x, std::vector<T>& o, bool bPositive)
{
    std::vector<T> _o;
    _o.reserve(u.size());
    for (int i = 0; i < u.size(); i++)
    {
        bool bFound = false;
        for (int j = 0; j < x.size() && !bFound; j++)
            bFound = (i == x[j]);
        
        if ((bPositive && bFound) || (!bPositive && !bFound))
            _o.push_back(u[i]);
    }
    
    o = _o;
}

std::vector<bool> xtl::indicesToLogicals(int n, std::vector<int> indices)
{
    std::vector<bool> logicals (n);
    for (int i = 0; i < indices.size(); i++)
    {
        logicals[indices[i]] = true;
    }
    return logicals;
}

std::vector<int> xtl::logicalsToIndices(std::vector<bool> logicals)
{
    std::vector<int> indices;
    indices.reserve(indices.size());
    for (int i = 0; i < logicals.size(); i++)
        if (logicals[i]) indices.push_back(i);
    
    return indices;
}

template<typename T>
std::vector<T> xtl::getCol(std::vector<std::vector<T> > w, int i)
{
    std::vector<T> v (w.size());
    for (int j = 0; j < w.size(); j++)
    {
        v[j] = w[j][i];
    }
    
    return v;
}

template<typename T>
std::vector<T> xtl::getRow(std::vector<std::vector<T> > w, int i)
{
    return w[i];
}

//
// Template function instantiations
//

template void xtl::filterByValue(std::vector<int> u, std::vector<int> v, std::vector<int>& o, std::vector<int>& z, bool bPositive);
template void xtl::filterByValue(std::vector<float> u, std::vector<float> v, std::vector<float>& o, std::vector<int>& z, bool bPositive);

template void xtl::filterByValue(std::vector<int> u, std::vector<int> v, std::vector<int>& o, bool bPositive);
template void xtl::filterByValue(std::vector<float> u, std::vector<float> v, std::vector<float>& o, bool bPositive);

template void xtl::filterByIndex(std::vector<int> u, std::vector<int> x, std::vector<int>& o, bool bPositive);
template void xtl::filterByIndex(std::vector<float> u, std::vector<int> x, std::vector<float>& o, bool bPositive);

template void xtl::filterByIndex(std::vector<std::vector<float> > u, std::vector<int> x, std::vector<int> y, std::vector<std::vector<float> >& o, bool bPositiveX, bool bPositiveY);

template std::vector<int> xtl::getCol(std::vector<std::vector<int> > w, int i);
template std::vector<float> xtl::getCol(std::vector<std::vector<float> > w, int i);

template std::vector<int> xtl::getRow(std::vector<std::vector<int> > w, int i);
template std::vector<float> xtl::getRow(std::vector<std::vector<float> > w, int i);

