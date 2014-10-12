//
//  xtio.h
//  descriptors_test
//
//  Created by Albert Clapés on 06/10/14.
//
//

#ifndef __descriptors_test__xtio__
#define __descriptors_test__xtio__

#include <iostream>
#include <vector>

namespace xtl
{
    // Print functions
    
    template<typename T>
    void print(std::vector<T> v);
    
    template<typename T>
    void print(std::vector<std::vector<T> > w);
}

#endif /* defined(__descriptors_test__xtended__) */
