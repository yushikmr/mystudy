#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <cmath>

class ConstDirichlet{
    public:
        const double xlow, ylow;
        const double xup, yup;
        ConstDirichlet()
            :xlow(0), xup(0), ylow(0), yup(0){}
        ConstDirichlet(double xl, double xu, double yl, double yu):
        xlow(xl), xup(xu), ylow(yl), yup(yu){}
        ConstDirichlet(const ConstDirichlet &diric):
            xlow(diric.xlow), xup(diric.xup), ylow(diric.ylow), yup(diric.yup){}
};

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

        Matrix(int row, int col): r(row), c(col){
            value =  std::make_unique<std::unique_ptr<double[]>[]>(r);
            for (size_t i = 0; i < r; i++)
            {
                value[i] = std::make_unique<double[]>(c);
                for (size_t j = 0; j < c; j++)
                {
                    value[i][j] = 0.;
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
        int getnumrow() const {
            return r;
        }
        int getnumcol() const {
            return c;
        }
        std::unique_ptr< std::unique_ptr<double[]>[]> assignment_value(){
            return std::move(value);
        }

        double getv(int i, int j) const {
            return double(value[i][j]);
        }

        void update(int i, int j, double v){
            value[i][j] = v;
        }
};

class ForceField: public Matrix{
    public:
        ForceField(int row, int col): Matrix(row, col){}
        ForceField(std::string filepath):Matrix(filepath){}
        ForceField(const ForceField &ff):Matrix(ff.r, ff.c){
            for (size_t i = 0; i < r; i++)
            {
                for (size_t j = 0; j < c; j++)
                {
                    value[i][j] = ff.getv(i, j);
                }
            }
        }
};

class LaplacesEq{
    private:
        int r, c;
        double xmin=0., xmax=1., ymin=0., ymax=1.;
        double dx, dy;
        ConstDirichlet diric;
        Matrix field;
        ForceField force;
    public:
        LaplacesEq(int row, int col, ConstDirichlet &dirichlet, ForceField &ff):
         r(row), c(col), diric(dirichlet), field(row, col), force(ff){
            dx = (xmax - xmin) / c;
            dy = (ymax - ymin) / r;
        }

        double assignment(const int i, const int j){
            double old = field.getv(i, j);
            if (i == 0)
            {
                field.update(i, j, diric.ylow);
            }
            else if (j == 0){
                field.update(i, j, diric.xlow);
            }

            else if (i == r - 1)
            {
                field.update(i, j, diric.yup);
            }

            else if (j == c -1)
            {
                field.update(i, j, diric.xup);
            }

            else{
                field.update(i, j, 
                            (1. / 4.) * (field.getv(i+1, j) + field.getv(i-1, j) 
                                        + field.getv(i, j+1) + field.getv(i, j-1)) 
                                        + dx * dy * force.getv(i, j));
            }
            double diff = abs(old - field.getv(i, j));
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


        void save_sol(std::string savepath){
            std::ofstream outfile(savepath);
            outfile << r << std::endl;
            outfile << c << std::endl;
            for (size_t i = 0; i < r; i++)
            {
                for (size_t j = 0; j < c; j++)
                {
                    outfile << field.getv(i, j) << " ";
                }
                outfile << std::endl;
                
            }
            
        }


};

int main(){

    

    ConstDirichlet dir(0, 0, 0, 0.03);
    ForceField ff(100, 100);
    ff.update(30, 30, -100);
    ff.update(70, 70, 100);
    LaplacesEq eq(100, 100, dir, ff);
    eq.solve(3000);

    eq.save_sol("data/output03.txt");
    return 0;
}
