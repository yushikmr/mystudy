#include <iostream>
#include <memory>
#include <vector>
#include <cmath>


#include "matrix.cpp"

# define EPS 0.0001
# define MAXITER 100

/*! @class GaussSaidel
    @brief  Gauss Saidel iterative method.
*/
class GaussSaidel{

    protected:
        int numvar;
        std::unique_ptr<double[]> variables;
        Matrix augcoefmx;
        const double eps;
        const int maxiter;
        bool resolved;

        /**
         * @fn sort_diagonal
         * 対角要素が大きくなるようにsort
         * @detail 
         */
        void sort_diagonal(){
            int n =  augcoefmx.getnumrow();
            for (size_t i = 0; i < n; i++)
            {
                double diagonal = abs(augcoefmx.getv(i, i));
                for (size_t k = i+1; k < n; k++)
                {   
                    if (abs(augcoefmx.getv(k, i)) > diagonal){
                        diagonal = abs(augcoefmx.getv(k, i));
                        augcoefmx.swap_row(i, k);
                    }
                }
            }
        }

    public:
        GaussSaidel(std::string filepath, double e, int mi): 
                    augcoefmx(filepath), eps(e), maxiter(mi){
            numvar = augcoefmx.getnumcol() - 1 ;
            variables = std::make_unique<double[]>(numvar);
            for (size_t j = 0; j < numvar; j++)
            {
                variables[j] = 1.;
            }
            
        }

        /**
         * @fn self_consistent
         * 自己無撞着な試行
         * @brief 解を代入して新しい解を得る
         * @return 試行における誤差
         * @detail 
         */
        double self_consistent(){
            // 収束させるために対角要素が大きくなるように行を入れ替え
            sort_diagonal();
            double residual = 0.;
            int col = augcoefmx.getnumcol();
            
            for (size_t i = 0; i < numvar; i++)
            {
                double v = 0.;
                double scale = augcoefmx.getv(i, i);
                for (size_t j = 0; j < col - 1; j++)
                {
                    if(j != i){
                        v -= (augcoefmx.getv(i, j) / scale) *  variables[j] ;
                    }
                }
                v += (augcoefmx.getv(i, col -1) / scale);
                // update (i)th variable
                residual += pow(v -  variables[i], 2);
                variables[i] = v;
            }
            return residual;
        }

        /**
         * @fn solve
         * gauss saidel によえり連立方程式の解を算出
         * @brief 
         * @return 反復が完了した時点での反復回数と誤差
         * @detail 
         */
        std::pair<int, double> solve(){
            std::pair<int, double> result(0, 0.);
            double res;
            for (size_t i = 0; i < maxiter; i++)
            {
                res = self_consistent();
                if (res < eps){
                    resolved = true;
                    return std::make_pair(i+1, res);
                }
            }
            resolved = false;
            return std::make_pair(maxiter, res); 
        }

        void showmat(){
            augcoefmx.showmat();
        }
        /**
         * @fn showvar
         * 解を標準出力
         * @brief メンバ変数 variablesを標準出力.
         * @detail 
         */
        void showvar(){
            for (size_t j = 0; j < numvar; j++)
            {
                std::cout << " " << variables[j];
            }
            std::cout << std::endl;  
        }
};

int main(){
    GaussSaidel gs1("data/sample05.txt", EPS, MAXITER);
    gs1.showmat();
    
    std::pair<int, double> result;
    result = gs1.solve();
    std::cout << "iter = " << result.first << 
        ", resudual =  " << result.second << std::endl; 
    gs1.showvar();

}