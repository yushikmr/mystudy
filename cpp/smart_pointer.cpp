#include <iostream>
#include <memory>
#include <vector>
#include <cmath>

int main() {
    std::unique_ptr<int> x(new int(100));

    std::cout << *x << std::endl;

    // copyはできないので、std::moveを使用
    std::unique_ptr<int> y =std::move(x);

    std::cout << *y << std::endl;

    int n = 10;
    std::unique_ptr<double[]> z(new double[n]);
    for (size_t i = 0; i < n; i++)
    {
        z[i] = 0.;
    }

    z[2] = 1.;
    
    return 0;
}