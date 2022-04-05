#include <bits/stdc++.h>
#include <omp.h>

using namespace std;

int main() {
    vector<int>sizes;
    
    int curr = 1024;
    for(int i = 0; i<6; i++){
         sizes.push_back(curr);
         curr *= 2;
    }

    string base = "A1_Q2_inputs/A1_Q2_input";
    curr = 1;

    ofstream in_fp, out_fp;
    
    for (int size : sizes) {
        string in_file = base + to_string(curr) + ".txt";
        string out_file = "A1_Q2_outputs/A1_Q2_output" + to_string(curr) + ".txt";
        
        in_fp.open(in_file);
        out_fp.open(out_file);
        in_fp << size << '\n';

        double*x =(double*)malloc(size*sizeof(double));
        double*y =(double*)malloc(size*sizeof(double));

        double**L = (double**)malloc(sizeof(double*));
        
        int n = size;

        for (int i = 0; i < n; i++) {
            L[i] = (double*)malloc((i+1)*sizeof(double));
        }

        // Parallelise the zeroing of the L matrix
        #pragma omp parallel for num_threads(8)
        for(int i =0 ; i<n; i++){
            for(int j = 0; j<=i; j++){
                L[i][j] = 0;
            }
        }

        // Parallelise the generation of lower triangular matrix L
        #pragma omp parallel for num_threads(8)
        for(int i = 1; i<=n; i++){
            for(int j = 1; j<i; j++){
                L[i-1][j-1] = (1.0+j+i)/(j*i + 4);
            }
            L[i-1][i-1] = (1.0*i)/(i + 7);
        }

        // Initialise the y values
        for(int i = 0; i<n; i++){
            y[i] = (1.0*i+10)/(i*2 + 5) + 3.25/(i+1);
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                in_fp << L[i][j] << ' ';
            }
            in_fp << '\n';
        }

        for(int i=0;i<size ; i++){
            in_fp << y[i] << ' ';
        }

        in_fp << '\n';

        for (int i = 0; i < size; i++) {
            in_fp << x[i] << ' ';
        }

        in_fp.close();
        out_fp.close();
        curr++;    
    }

    cout << "Done printing\n";
}
