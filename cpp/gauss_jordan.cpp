#include <iostream>
#include <memory>
#include <vector>
#include <cmath>

#include "matrix.cpp"


template<typename T>
void print_array2d(const T *vec, int r, int c){
    for (size_t i = 0; i < r; i++)
    {
       for (size_t j = 0; j < c; j++)
       {
            std::cout << " " << vec[i][j] ;
       }
    }
    std::cout << " " << std::endl;
}


class AufCoefMatrix: public virtual Matrix{
    private:
        bool resolved;
        void normalize_row(const int i){
            double scale = double(value[i][i]);
            for (size_t j = 0; j < c; j++)
            {
                value[i][j] /= scale;
            }
        }

        void sweep_out(const int i){
            normalize_row(i);
            for (size_t k = 0; k < r; k++)
            {
                if (k != i)
                {
                    double scale = double(value[k][i]);
                    for (size_t j = 0; j < c; j++)
                    {
                        value[k][j] -= value[i][j] * scale;
                    }
                }
            }
        }

        int get_valid_index(const int i){
            int valid_index = -1;
            for(size_t k = i; k < r; k++)
            {
                if(value[k][i] != 0){
                valid_index = k;
                return valid_index;
            }}
            return valid_index;
        }

    public:

        AufCoefMatrix(std::string filepath): Matrix(filepath){}

        std::unique_ptr<double[]> gauss_jordan(){

            std::unique_ptr<double[]> sol = std::make_unique<double[]>(c);
            for (size_t i = 0; i < r; i++)
            {
                // 0で割らないように行を入れ替え
                int valid_index = get_valid_index(i);
                if (valid_index == -1){
                    // 解を持たない場合
                    resolved = false;
                    std::unique_ptr<double[]> ensol = std::make_unique<double[]>(c);
                    return ensol;
                }
                else{
                    swap_row(i, valid_index);
                    sweep_out(i);
                }
            }
            resolved = true;
            // extract solution
            for (size_t i = 0; i < r; i++)
            {
                sol[i] = double(value[i][c-1]);
            }
            return sol;
        }

        void swap_row(int i1, int i2){
            std::swap(value[i1], value[i2]);
        }
};


int main(){

    AufCoefMatrix mat1("data/sample01.txt");
    mat1.gauss_jordan();
    mat1.showmat();


    AufCoefMatrix mat2("data/sample02.txt");
    mat2.gauss_jordan();
    mat2.showmat();

    AufCoefMatrix mat3("data/sample03.txt");
    mat3.gauss_jordan();
    mat3.showmat();
}