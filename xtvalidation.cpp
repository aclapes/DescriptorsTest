//
//  xtvalidation.cpp
//  descriptors_test
//
//  Created by Albert Clapés on 06/10/14.
//
//

#include "xtvalidation.h"
#include <map>

//
// Functions
//

float xtl::computeAccuracy(std::vector<int> groundtruth, std::vector<int> predictions)
{
    std::map<std::string,int> instancesMap;
    
    for (int i = 0; i < groundtruth.size(); ++i)
        instancesMap[std::to_string(groundtruth[i])]++;
    
    std::map<std::string,int> hitsMap;
    
    std::map<std::string,int>::iterator instancesIt;
    for (instancesIt = instancesMap.begin(); instancesIt != instancesMap.end(); instancesIt++)
        hitsMap[instancesIt->first] = 0;
    
    for (int i = 0; i < groundtruth.size(); ++i)
        if (groundtruth[i] == predictions[i])
            hitsMap[std::to_string(groundtruth[i])]++;
    
    instancesIt = instancesMap.begin();
    std::map<std::string,int>::iterator hitsIt = hitsMap.begin();
    
    float accSum = 0.f;
    while (instancesIt != instancesMap.end())
    {
        accSum += ((float) hitsIt->second) / instancesIt->second;
        instancesIt++;
        hitsIt++;
    }
    
    return accSum / instancesMap.size();
}

void xtl::computeConfusion(std::vector<int> groundtruth, std::vector<int> predictions, std::vector<std::vector<int> >& confusions)
{
    std::vector<int> nHits;
    std::vector<int> nInstances;
    
    for (int i = 0; i < groundtruth.size(); ++i)
    {
        while (groundtruth[i] >= nInstances.size())
        {
            nHits.push_back(0);
            nInstances.push_back(0);
        }
        
        nInstances[groundtruth[i]]++;
        
        if (predictions[i] == groundtruth[i])
            nHits[groundtruth[i]]++;
    }
    
    confusions.resize(nInstances.size());
    for (int i = 0; i < confusions.size(); i++)
        confusions[i].resize(nInstances.size(), 0);
    
    for (int i = 0; i < groundtruth.size(); i++)
    {
        if (predictions[i] >= 0)
            confusions[groundtruth[i]][predictions[i]]++;
    }
}

std::vector<float> xtl::linspace(float a, float b, float step, bool bExtremaIn)
{
    if (!bExtremaIn)
    {
        a += step;
        b -= step;
    }
    
    std::vector<float> points ((b-a)/step + 1);
    
    for (int i = 0; i < points.size(); i++)
        points[i] = a + i * step;
    
    return points;
}

// ===========================================================================//
// CLASSES
//============================================================================//

//
// Cross validation
//

xtl::CvPartition::CvPartition(int n, int k, int seed)
{
    m_NumOfPartitions = k;
    
    std::default_random_engine g(seed);
    
    m_Partitions.resize(n);
    for (int i = 0; i < m_Partitions.size(); i++)
        m_Partitions[i] = i % k;
    
    std::shuffle(m_Partitions.begin(), m_Partitions.end(), g);
}

xtl::CvPartition::CvPartition(std::vector<int> groups, int k, int seed)
{
    m_NumOfPartitions = k;
    
    std::map<std::string,std::vector<int> > map;
    for (int i = 0; i < groups.size(); i++)
    {
        std::string key = std::to_string(groups[i]);
        map[key].push_back(i);
    }
    
    std::vector<int> labels;
    std::vector<std::vector<int> > indices;
    
    std::map<std::string,std::vector<int> >::iterator it;
    for (it = map.begin(); it != map.end(); it++)
    {
        labels.push_back(stoi(it->first));
        
        std::vector<int> labelIndices = it->second;
        indices.push_back(labelIndices);
    }
    
    // Generate random partitions
    
    m_Partitions.resize(groups.size());
    
    std::default_random_engine g(seed);
    
    for (int i = 0; i < labels.size(); i++)
    {
        std::vector<int> v (indices[i].size());
        for (int j = 0; j < indices[i].size(); j++)
            v[j] = j % k;
        
        std::shuffle(v.begin(), v.end(), g);
        
        for (int j = 0; j < indices[i].size(); j++)
            m_Partitions[indices[i][j]] = v[j];
    }
}

xtl::CvPartition::CvPartition(const CvPartition& rhs)
{
    *this = rhs;
}

xtl::CvPartition& xtl::CvPartition::operator=(const CvPartition& rhs)
{
    if (this != &rhs)
    {
        m_NumOfPartitions = rhs.m_NumOfPartitions;
        m_Partitions = rhs.m_Partitions;
    }
    
    return *this;
}

int xtl::CvPartition::getNumOfPartitions()
{
    return m_NumOfPartitions;
}

std::vector<int> xtl::CvPartition::getPartitions()
{
    return m_Partitions;
}

std::vector<int> xtl::CvPartition::getPartition(int p, bool bLogicalIndexation)
{
    std::vector<int> v;
    if (bLogicalIndexation)
    {
        v.resize(m_Partitions.size());
        
        for (int i = 0; i < m_Partitions.size(); i++)
            v[i] = (m_Partitions[i] == p);
    }
    else
    {
        for (int i = 0; i < m_Partitions.size(); i++)
            if (m_Partitions[i] == p)
                v.push_back(i);
    }
    
    return v;
}

template<typename T>
void xtl::CvPartition::getTrainTest(std::vector<T> data, int p, std::vector<T>& train, std::vector<T>& test)
{
    train.clear();
    test.clear();
    for (int i = 0; i < data.size(); i++)
    {
        (m_Partitions[i] == p) ? test.push_back(data[i]) : train.push_back(data[i]);
    }
}

template<typename T>
void xtl::CvPartition::getTrain(std::vector<T> data, int p, std::vector<T>& train)
{
    train.clear();
    for (int i = 0; i < data.size(); i++)
    {
        if (m_Partitions[i] != p) train.push_back(data[i]);
    }
}

template<typename T>
void xtl::CvPartition::getTest(std::vector<T> data, int p, std::vector<T>& test)
{
    test.clear();
    for (int i = 0; i < data.size(); i++)
    {
        if (m_Partitions[i] == p) test.push_back(data[i]);
    }
}

template void xtl::CvPartition::getTrainTest(std::vector<int> data, int p, std::vector<int>& train, std::vector<int>& test);
template void xtl::CvPartition::getTrainTest(std::vector<float> data, int p, std::vector<float>& train, std::vector<float>& test);
template void xtl::CvPartition::getTrainTest(std::vector<std::vector<float> > data, int p, std::vector<std::vector<float> >& train, std::vector<std::vector<float> >& test);

template void xtl::CvPartition::getTrain(std::vector<int> data, int p, std::vector<int>& train);
template void xtl::CvPartition::getTrain(std::vector<float> data, int p, std::vector<float>& train);
template void xtl::CvPartition::getTrain(std::vector<std::vector<float> > data, int p, std::vector<std::vector<float> >& train);

template void xtl::CvPartition::getTest(std::vector<int> data, int p, std::vector<int>& test);
template void xtl::CvPartition::getTest(std::vector<float> data, int p, std::vector<float>& test);
template void xtl::CvPartition::getTest(std::vector<std::vector<float> > data, int p, std::vector<std::vector<float> >& test);

//
// Leave-one-out cross validation
//

xtl::LoocvPartition::LoocvPartition(int n, int seed)
{
    std::default_random_engine g(seed);
    
    m_Partitions.resize(n);
    for (int i = 0; i < m_Partitions.size(); i++)
        m_Partitions[i] = i;
    
    std::shuffle(m_Partitions.begin(), m_Partitions.end(), g);
}

xtl::LoocvPartition::LoocvPartition(const LoocvPartition& rhs)
{
    *this = rhs;
}

xtl::LoocvPartition& xtl::LoocvPartition::operator=(const LoocvPartition& rhs)
{
    if (this != &rhs)
    {
        m_Partitions = rhs.m_Partitions;
    }
    
    return *this;
}

int xtl::LoocvPartition::getNumOfPartitions()
{
    return m_Partitions.size();
}

std::vector<int> xtl::LoocvPartition::getPartitions()
{
    return m_Partitions;
}

std::vector<int> xtl::LoocvPartition::getPartition(int p, bool bLogicalIndexation)
{
    std::vector<int> v;
    if (bLogicalIndexation)
    {
        v.resize(m_Partitions.size());
        
        for (int i = 0; i < m_Partitions.size(); i++)
            v[i] = (m_Partitions[i] == p);
    }
    else
    {
        for (int i = 0; i < m_Partitions.size(); i++)
            if (m_Partitions[i] == p)
                v.push_back(i);
    }
    
    return v;
}

template<typename T>
void xtl::LoocvPartition::getTrainTest(std::vector<T> data, int p, std::vector<T>& train, std::vector<T>& test)
{
    train.clear();
    test.clear();
    for (int i = 0; i < data.size(); i++)
    {
        (m_Partitions[i] == p) ? test.push_back(data[i]) : train.push_back(data[i]);
    }
}

template<typename T>
void xtl::LoocvPartition::getTrain(std::vector<T> data, int p, std::vector<T>& train)
{
    train.clear();
    for (int i = 0; i < data.size(); i++)
    {
        if (m_Partitions[i] != p) train.push_back(data[i]);
    }
}

template<typename T>
void xtl::LoocvPartition::getTest(std::vector<T> data, int p, std::vector<T>& test)
{
    test.clear();
    for (int i = 0; i < data.size(); i++)
    {
        if (m_Partitions[i] == p) test.push_back(data[i]);
    }
}

template void xtl::LoocvPartition::getTrainTest(std::vector<int> data, int p, std::vector<int>& train, std::vector<int>& test);
template void xtl::LoocvPartition::getTrainTest(std::vector<float> data, int p, std::vector<float>& train, std::vector<float>& test);
template void xtl::LoocvPartition::getTrainTest(std::vector<std::vector<float> > data, int p, std::vector<std::vector<float> >& train, std::vector<std::vector<float> >& test);

template void xtl::LoocvPartition::getTrain(std::vector<int> data, int p, std::vector<int>& train);
template void xtl::LoocvPartition::getTrain(std::vector<float> data, int p, std::vector<float>& train);
template void xtl::LoocvPartition::getTrain(std::vector<std::vector<float> > data, int p, std::vector<std::vector<float> >& train);

template void xtl::LoocvPartition::getTest(std::vector<int> data, int p, std::vector<int>& test);
template void xtl::LoocvPartition::getTest(std::vector<float> data, int p, std::vector<float>& test);
template void xtl::LoocvPartition::getTest(std::vector<std::vector<float> > data, int p, std::vector<std::vector<float> >& test);