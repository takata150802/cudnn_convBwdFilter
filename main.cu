#include <iostream>
#include <vector>
#include <cudnn.h>

#define CHECK(call)                                                  \
{                                                                    \
    const cudaError_t error = call;                                  \
    std::cout << "CHECK cudaError_t: ";                              \
    if (error != cudaSuccess)                                        \
    {                                                                \
        std::cout << __FILE__                                        \
                  << "("                                             \
                  << __LINE__                                        \
                  << ")"                                             \
                  << ": "                                            \
                  << "Error"                                         \
                  << std::endl;                                      \
        std::cout << "code: "                                        \
                  << error                                           \
                  << ", "                                            \
                  << "reason: "                                      \
                  << cudaGetErrorString(error)                       \
                  << std::endl;                                      \
        exit(1);                                                     \
    }                                                                \
    else                                                             \
    {                                                                \
        std::cout << __FILE__                                        \
                  << "("                                             \
                  << __LINE__                                        \
                  << ")"                                             \
                  << ": "                                            \
                  << "cudaSuccess"                                   \
                  << std::endl;                                      \
    }                                                                \
}

int main(int argc, char *argv[]) {
    return 0;
}

