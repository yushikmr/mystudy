#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <cmath>

class ConstDirichlet{
    public:
        double xlow = 0. , ylow = 0.;
        double xup = 1., yup = 1.;
        ConstDirichlet(){}
};

class LaplacesEq{
    private:
        int r, c;
        double xmin=0., xmax=1., ymin=0., ymax=1.;
        double dx, dy;
        ConstDirichlet diric;
        std::unique_ptr< std::unique_ptr<double[]>[]> field;
    public:
        LaplacesEq(int row, int col, ConstDirichlet dirichlet): r(row), c(col){
            dx = (xmax - xmin) / c;
            dy = (ymax - ymin) / r;
            diric = dirichlet;

            field =  std::make_unique<std::unique_ptr<double[]>[]>(r);

            for (size_t i = 0; i < r; i++)
            {
                field[i] = std::make_unique<double[]>(c);
                for (size_t j = 0; j < c; j++)
                {
                    field[i][j] = 0.;
                }
            }
        }

        double assignment(const int i, const int j){
            double old = double(field[i][j]);
            if (i == 0)
            {
                field[i][j] = diric.ylow;
            }
            else if (j == 0){
                field[i][j] = diric.xlow;
            }

            else if (i == r - 1)
            {
                field[i][j] =  diric.yup;
            }

            else if (j == c -1)
            {
                field[i][j] =  diric.xup;
            }

            else{
                field[i][j] = 
                    (1. / 4.) * (field[i+1][j] + field[i-1][j] 
                                + field[i][j+1] + field[i][j-1]);
            }
            double diff = abs(old - double(field[i][j]));
            return diff;
        }

        double self_consistent(){
            double d=0.;
            for (size_t i = 0; i < r; i++)
            {
                for (size_t j = 0; j < c; j++)
                {
                    d += assignment(i, j);
                }
            }
            return d; 
        }

        void solve(int niter){
            double d;
            for (size_t i = 0; i < niter; i++)
            {
                d = self_consistent();
                std::cout << "iter "<< i+1 <<": diff :  "<< d << std::endl;
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
                    std::cout << " " << field[i][j];
                }
                std::cout << std::endl;
            }
            for (size_t j = 0; j < c; j++)
            {
                std::cout << "---";
            }
            std::cout << std::endl;
            
        }

        void save_sol(std::string savepath){
            std::ofstream outfile(savepath);
            outfile << r << std::endl;
            outfile << c << std::endl;
            for (size_t i = 0; i < r; i++)
            {
                for (size_t j = 0; j < c; j++)
                {
                    outfile << field[i][j] << " ";
                }
                outfile << std::endl;
                
            }
            
        }


};

int main(){
    ConstDirichlet dir;
    LaplacesEq eq(100, 100, dir);
    // eq.showmat();
    eq.solve(10000);

    eq.save_sol("data/output01.txt");

    // eq.showmat();
    return 0;
}
