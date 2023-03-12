#include<iostream>
#include<fstream>
#include <memory>


class Matrix{
    protected:
        int r, c;
        std::unique_ptr< std::unique_ptr<double[]>[]> value;
    public:
        Matrix(const std::string filepath){
            std::ifstream  inputfile(filepath);
            inputfile >> r;
            inputfile >> c;
            value =  std::make_unique<std::unique_ptr<double[]>[]>(r);
            for (size_t i = 0; i < r; i++)
            {
                value[i] = std::make_unique<double[]>(c);
            }

            for (size_t i = 0; i < r; i++)
            {
                for (size_t j = 0; j < c; j++)
                {
                    inputfile >> value[i][j];
                }
            }
        }
        void showmat(){
            std::cout << "Matrix: " << " row = " << r << ", col= " 
            << c << std::endl;
            
            for (size_t j = 0; j < c; j++)
            {
                std::cout << "---";
            }
            std::cout << std::endl;

            for (size_t i = 0; i < r; i++)
            {
                for (size_t j = 0; j < c; j++)
                {
                    std::cout << " " << value[i][j];
                }
                std::cout << std::endl;
            }
            for (size_t j = 0; j < c; j++)
            {
                std::cout << "---";
            }
            std::cout << std::endl;
            
        }
        int getnumrow(){
            return r;
        }
        int getnumcol(){
            return c;
        }
        std::unique_ptr< std::unique_ptr<double[]>[]> assignment_value(){
            return std::move(value);
        }
        

};
