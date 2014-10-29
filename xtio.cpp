//
//  xtio.cpp
//  descriptors_test
//
//  Created by Albert Clapés on 06/10/14.
//
//

#include "xtio.h"
#include <iterator>
#include <string>

template<typename T>
void xtl::print(std::vector<T> v)
{
    std::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, "\t"));
    std::cout << std::endl;
}

template<typename T>
void xtl::print(std::vector<std::vector<T> > w)
{
    for (int i = 0; i < w.size(); i++)
    {
        std::copy(w[i].begin(), w[i].end(), std::ostream_iterator<T>(std::cout, "\t"));
        std::cout << std::endl;;
    }
    std::cout << std::endl;
}


//
// Template instantiation
//

template void xtl::print(std::vector<int> v);
template void xtl::print(std::vector<float> v);
template void xtl::print(std::vector<double> v);
template void xtl::print(std::vector<char> v);
template void xtl::print(std::vector<std::string> v);

template void xtl::print(std::vector<std::vector<int> > w);
template void xtl::print(std::vector<std::vector<float> > w);
template void xtl::print(std::vector<std::vector<double> > w);
template void xtl::print(std::vector<std::vector<char> > w);
template void xtl::print(std::vector<std::vector<std::string> > w);