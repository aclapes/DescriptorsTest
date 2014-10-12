//
//  xtmath.h
//  descriptors_test
//
//  Created by Albert Clapés on 06/10/14.
//
//

#ifndef __descriptors_test__xtmath__
#define __descriptors_test__xtmath__

#include <iostream>
#include <vector>
#include <math.h>

namespace xtl
{
    // Element-wise arithmetic functions
    
    template<typename T>
    void sum(std::vector<std::vector<std::vector<T> > > vw, std::vector<std::vector<T> >& w);
    
    template<typename T>
    void avg(std::vector<std::vector<std::vector<T> > > vw, std::vector<std::vector<T> >& w);
    
    enum {COLLAPSE_SUM, COLLAPSE_AVG};
    template<typename T>
    void collapse(std::vector<std::vector<T> > w, int dim, std::vector<float>& v, int operation = COLLAPSE_SUM);
    
    template<typename T>
    T mean(std::vector<T> v);
    
    template<typename T>
    T stddev(std::vector<T> v);
    
    template<typename T>
    std::pair<int,T> max(std::vector<T> v);
    
    template<typename T>
    std::pair<int,T> min(std::vector<T> v);
}

#endif /* defined(__descriptors_test__xtmath__) */
