#include<iostream>
#include <sys/time.h>
using namespace std;

const int threadsPerBlock = 512;
const int N               = (1 <<20)-3;
const int blocksPerGrid   = (N + threadsPerBlock - 1)/threadsPerBlock;
const int iters           = 100;

__global__ void kernel1(float* arr, float* out, int N){
    __shared__ float s_data[threadsPerBlock];
    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x; 
    if(i < N){
        s_data[tid] = arr[i];
    }
    __syncthreads();

    for(int s = 1; s < blockDim.x; s*=2){
        if(tid % (2*s) == 0 && i + s <N){
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0){
        out[blockIdx.x] = s_data[0];
    }
}

__global__ void kernel2(float* arr, float* out, int N){
    __shared__ float s_data[threadsPerBlock];
    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x; 
    if(i < N){
        s_data[tid] = arr[i];
    }
    __syncthreads();

    for(int s = 1; s < blockDim.x; s*=2){
        int index = tid * 2 * s;       // 原来是每个线程对应一个位置，第一轮循环，只有0、2、4、6这些线程在执行，1、3、5线程闲置，同一个warp内有一半线程没有用上
        if((index + s) < blockDim.x && (blockIdx.x * blockDim.x + index + s) < N){   // 现在是tid号线程处理处理tid*2*s位置的任务，第一轮循环0123456线程都在线，warp利用率高
            s_data[index] += s_data[index + s];
        }
        __syncthreads();
    }

    if(tid == 0){
        out[blockIdx.x] = s_data[0];
    }
}

__global__ void kernel3(float* arr, float* out, int N){
    __shared__ float s_data[threadsPerBlock];
    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x; 
    if(i < N){
        s_data[tid] = arr[i];
    }
    __syncthreads();

    for(int s = blockDim.x/2; s > 0; s>>=1){   // 2的访问share memory的方式，存在share memory bank conflit
        if(tid < s && i + s < N){
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0){
        out[blockIdx.x] = s_data[0];
    }
}


void varifyOutput(float* predict, float* arr, int N){
    float pred = 0.0;
    for(int i=0;i<blocksPerGrid;i++){
        pred += predict[i];
    }

    float result = 0.0;
    struct timeval s;
    struct timeval e; 
    gettimeofday(&s,NULL);
    for(int t=0;t<iters;t++){
        result = 0.0;
        for(int i=0;i<N;i++){
            result += arr[i];
        }
    }
    gettimeofday(&e,NULL); 
    cout << "CPU Elapse time: " << ((e.tv_sec-s.tv_sec)*1000000+(e.tv_usec-s.tv_usec)) / iters / 1000.0 << " ms" << endl; 
    cout << "predict: " << pred << endl << "result: " << result << endl;
}

int main(){
    float* a_host, *r_host;
    float* a_device, *r_device;

    cudaMallocHost(&a_host, N * sizeof(float));
    cudaMallocHost(&r_host, blocksPerGrid * sizeof(float));

    cudaMalloc(&a_device, N * sizeof(float));
    cudaMalloc(&r_device, blocksPerGrid * sizeof(float));

    for(int i=0;i<N;i++){
        a_host[i] = 1;
    }
    for(int i=0;i<blocksPerGrid;i++){
        r_host[i] = 0.0;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(a_device, a_host, N * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(r_device, r_host, blocksPerGrid * sizeof(float), cudaMemcpyHostToDevice, stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for(int i=0;i<iters;i++){
        kernel3<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(a_device, r_device, N);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "GPU Elapse time: " << elapsedTime / iters << " ms" << endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(r_host, r_device, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    varifyOutput(r_host, a_host, N);

    cudaFree(r_device);
    cudaFree(a_device);
    cudaFreeHost(r_host);
    cudaFreeHost(a_host);
    
    return 0;
}