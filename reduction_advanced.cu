#include<iostream>
#include <sys/time.h>
using namespace std;

const int threadsPerBlock = 256;
const int N               = (1 <<20) -3;
const int blocksPerGrid   = (N + threadsPerBlock * 2 - 1)/ (threadsPerBlock * 2);  // 维持block数量不变
const int iters           = 100;

__global__ void kernel4(float* arr, float* out, int N){
    __shared__ float s_data[threadsPerBlock];
    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockIdx.x * (blockDim.x * 2); // 3的第一轮迭代，有一半的线程是idle的，现在把一个block的大小缩小一半
    if(i < N){
        s_data[tid] = arr[i] + arr[i + blockDim.x];  // 单独执行原来的第一轮迭代，后面代码不用变
    }
    __syncthreads();

    for(int s = blockDim.x/2; s > 0; s>>=1){
        if(tid < s && i + s < N){
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0){
        out[blockIdx.x] = s_data[0];
    }
}

__device__ void warpRecude(volatile float* s_data, int tid){ // volatile 关键字很重要，保证s_data从相应的内存单元取出，这里应该指gpu内存
    s_data[tid] += s_data[tid + 32];
    s_data[tid] += s_data[tid + 16];
    s_data[tid] += s_data[tid + 8];
    s_data[tid] += s_data[tid + 4];
    s_data[tid] += s_data[tid + 2];
    s_data[tid] += s_data[tid + 1];
}

__global__ void kernel5(float* arr, float* out, int N){
    __shared__ float s_data[threadsPerBlock];
    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockIdx.x * (blockDim.x * 2); // 3的第一轮迭代，有一半的线程是idle的，现在把一个block的大小缩小一半
    if(i < N){
        s_data[tid] = arr[i] + arr[i + blockDim.x];  // 单独执行原来的第一轮迭代，后面代码不用变
    }else{
        s_data[tid] = 0;
    }
    __syncthreads();

    for(int s = blockDim.x/2; s > 32; s>>=1){
        if(tid < s && i + s < N){
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    if(tid < 32){
        warpRecude(s_data, tid);
    }

    if(tid == 0){
        out[blockIdx.x] = s_data[0];
    }
}

template<unsigned int blockSize>
__device__ void warpRecude2(volatile float* s_data, int tid){ // volatile 关键字很重要，保证s_data从相应的内存单元取出，这里应该指gpu内存
    if(blockSize >= 64) s_data[tid] += s_data[tid + 32];   // if 是防止blockSize小于64，比如blockSize为16，那么会直接到下面
    if(blockSize >= 32) s_data[tid] += s_data[tid + 16];
    if(blockSize >= 16) s_data[tid] += s_data[tid + 8];
    if(blockSize >= 8) s_data[tid] += s_data[tid + 4];
    if(blockSize >= 4) s_data[tid] += s_data[tid + 2];
    if(blockSize >= 2) s_data[tid] += s_data[tid + 1];
}


template<unsigned int blockSize>
__global__ void reduce(float* arr, float* out, int N){
    __shared__ float s_data[threadsPerBlock];
    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockIdx.x * (blockDim.x * 2); // 3的第一轮迭代，有一半的线程是idle的，现在把一个block的大小缩小一半
    if(i < N){
        s_data[tid] = arr[i] + arr[i + blockDim.x];  // 单独执行原来的第一轮迭代，后面代码不用变
    }else{
        s_data[tid] = 0;
    }
    __syncthreads();

    if(blockSize >= 1024){
        if(tid < 512){
            s_data[tid] += s_data[tid+512];
        }
        __syncthreads();
    }
    if(blockSize >= 512){
        if(tid < 256){
            s_data[tid] += s_data[tid+256];
        }
        __syncthreads();
    }
    if(blockSize >= 256){
        if(tid < 128){
            s_data[tid] += s_data[tid+128];
        }
        __syncthreads();
    }
    if(blockSize >= 128){
        if(tid < 64){
            s_data[tid] += s_data[tid+64];
        }
        __syncthreads();
    }

    if(tid < 32){
        warpRecude2<blockSize>(s_data, tid);
    }

    if(tid == 0){
        out[blockIdx.x] = s_data[0];
    }
}

void kernel6(float* arr, float* out, int N, cudaStream_t &stream){   // 展开所有的循环，去除循环
    switch(threadsPerBlock){
        case 1024:
            reduce<1024><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 512:
            reduce<512><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 256:
            reduce<256><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 128:
            reduce<128><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 64:
            reduce<64><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 32:
            reduce<32><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 16:
            reduce<16><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 8:
            reduce<8><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 4:
            reduce<4><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 2:
            reduce<2><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 1:
            reduce<1><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
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
        // kernel5<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(a_device, r_device, N);
        kernel6(a_device, r_device, N, stream);
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