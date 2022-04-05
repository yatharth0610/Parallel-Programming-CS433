#include<bits/stdc++.h>
using namespace std;

int main(int argc, char**argv){
    int num  = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    string x;
    if(num == 0){
        x = "PTHREAD_BARRIER";
    }
    else if(num == 1){
        x = "SENSE_REVERSAL";
    }
    else if(num == 2){
        x = "TREE_BUSY_WAIT";
    }
    else if(num == 3){
        x = "CENTRALISED_CONDITIONAL";
    }
    else if(num == 4){
        x = "TREE_CONDITIONAL";
    }

    string compile_comm = "g++ -O3 -pthread -D"+x+" pthread_main_barrier_cpy.cpp -o pthread_main_barrier";
    system(compile_comm.c_str());

    string execute = "./pthread_main_barrier " + to_string(num_threads) + " > output.txt";
    system(execute.c_str());

    ifstream indata;
    indata.open("output.txt");
    int num1, num2;

    for(int i = 0; i<1000000; i++){
        vector<bool> v(num_threads, false);
        for(int j=0; j <num_threads; j++){
            indata >> num1 >> num2;
            assert(i == num2);
            assert(v[num1] == false);
            v[num1] = true;
        }
    }
    indata.close();
    //system("rm output.txt");


   


}
