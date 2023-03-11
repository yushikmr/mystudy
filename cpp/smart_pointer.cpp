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

    return 0;
}