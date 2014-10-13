//
//  xtvalidation.h
//  descriptors_test
//
//  Created by Albert Clapés on 06/10/14.
//
//

#ifndef __descriptors_test__xtvalidation__
#define __descriptors_test__xtvalidation__

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

namespace xtl
{
    //
    // Functions
    //
    
    float computeAccuracy(std::vector<int> groundtruth, std::vector<int> predictions);
    void computeConfusion(std::vector<int> groundtruth, std::vector<int> predictions, std::vector<std::vector<float> >& confusions, bool bNormalize = false);
    
    std::vector<float> linspace(float a, float b, float step, bool bExtremaIn);
    
    //
    // Classes
    //
    
    class CvPartition
    {
    public:
        
        CvPartition(int n, int k, int seed = 74);
        CvPartition(std::vector<int> groups, int k, int seed = 74);
        CvPartition(const CvPartition& rhs);
        
        CvPartition& operator=(const CvPartition& rhs);
        
        int getNumOfPartitions();
        std::vector<int> getPartitions();
        std::vector<int> getPartition(int p, bool bLogicalIndexation = true);
        
        template<typename T>
        void getTrainTest(std::vector<T> data, int p, std::vector<T>& train, std::vector<T>& test);
        template<typename T>
        void getTrain(std::vector<T> data, int p, std::vector<T>& train);
        template<typename T>
        void getTest(std::vector<T> data, int p, std::vector<T>& test);
        
    private:
        
        int m_NumOfPartitions;
        std::vector<int> m_Partitions;
    };
    
    class LoocvPartition
    {
    public:
        
        LoocvPartition(int n, int seed = 74);
        LoocvPartition(const LoocvPartition& rhs);
        
        LoocvPartition& operator=(const LoocvPartition& rhs);
        
        int getNumOfPartitions();
        std::vector<int> getPartitions();
        std::vector<int> getPartition(int p, bool bLogicalIndexation = true);
        
        template<typename T>
        void getTrainTest(std::vector<T> data, int p, std::vector<T>& train, std::vector<T>& test);
        template<typename T>
        void getTrain(std::vector<T> data, int p, std::vector<T>& train);
        template<typename T>
        void getTest(std::vector<T> data, int p, std::vector<T>& test);
        
    private:
        
        std::vector<int> m_Partitions;
    };
}

#endif /* defined(__descriptors_test__xtvalidation__) */
