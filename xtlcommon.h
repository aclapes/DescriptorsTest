//
//  xtlcommon.h
//  descriptors_test
//
//  Created by Albert Clapés on 08/10/14.
//
//

#ifndef __descriptors_test__xtlcommon__
#define __descriptors_test__xtlcommon__

#include <iostream>
#include <vector>

namespace xtl
{
    template<typename T>
    void filterByValue(std::vector<T> u, std::vector<T> v, std::vector<T>& o, std::vector<int>& a, bool bPositive = true);
    template<typename T>
    void filterByValue(std::vector<T> u, std::vector<T> v, std::vector<T>& o, bool bPositive = true);
    
    template<typename T>
    void filterByIndex(std::vector<T> u, std::vector<int> x, std::vector<T>& o, bool bPositive = true);
    template<typename T>
    void filterByIndex(std::vector<std::vector<T> > u, std::vector<int> x, std::vector<int> y, std::vector<std::vector<T> >& o, bool bPositiveX = true, bool bPositiveY = true);
    
    std::vector<bool> indicesToLogicals(int n, std::vector<int> indices);
    std::vector<int> logicalsToIndices(std::vector<bool> logicals);
    
    template<typename T>
    std::vector<T> getCol(std::vector<std::vector<T> > w, int i);
    template<typename T>
    std::vector<T> getRow(std::vector<std::vector<T> > w, int i);
}

#endif /* defined(__descriptors_test__xtlcommon__) */
