//
//  CvPartition.hpp
//  descriptors_test
//
//  Created by Albert Clapés on 04/10/14.
//
//


#include <algorithm>
#include <random>
#include <iostream>

class CvPartition
{
public:
    
    CvPartition(int n, int k)
    {
        m_NumOfPartitions = k;

        std::default_random_engine g(74);
        
        m_Partitions.resize(n);
        for (int i = 0; i < m_Partitions.size(); i++)
            m_Partitions[i] = i % k;
            
        std::shuffle(m_Partitions.begin(), m_Partitions.end(), g);
    }
    
    CvPartition(vector<int> groups, int k)
    {
        m_NumOfPartitions = k;
        
        std::map<string,std::vector<int> > map;
        for (int i = 0; i < groups.size(); i++)
        {
            string key = to_string(groups[i]);
            map[key].push_back(i);
        }
        
        std::vector<int> labels;
        std::vector<std::vector<int> > indices;
        
        std::map<string,std::vector<int> >::iterator it;
        for (it = map.begin(); it != map.end(); it++)
        {
            labels.push_back(stoi(it->first));
            
            std::vector<int> labelIndices = it->second;
            assert (k <= labelIndices.size());
            indices.push_back(labelIndices);
        }
        
        assert (k <= labels.size());

        // Generate random partitions
        
        m_Partitions.resize(groups.size());
        
        std::default_random_engine g(74);

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
    
    int getNumOfPartitions()
    {
        return m_NumOfPartitions;
    }
    
    std::vector<int> getPartitions()
    {
        return m_Partitions;
    }
    
    std::vector<int> getPartition(int p, bool bLogicalIndexation = true)
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
    
private:
    
    int m_NumOfPartitions;
    std::vector<int> m_Partitions;
    
};