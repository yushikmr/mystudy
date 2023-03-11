#include <iostream>
#include <memory>
#include <vector>
#include <cmath>

// function template
template<typename T>
T add(T x1, T x2){
    return x1 + x2;
}

// method template
template<typename T>
class Myclass{
    private:
        T value1;
    public:
        Myclass(T v1){value1 = v1;}
        void print_value(){std::cout << value1 << std::endl;}
};

int main(){
    Myclass<int> myc1(0);      // value1をintとして定義 
    Myclass<double> myc2(0.1); // value1をdoubleとして定義 

    myc1.print_value();
    myc2.print_value();
}




