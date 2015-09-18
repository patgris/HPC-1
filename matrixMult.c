#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define SIZEx 3.0
#define SIZEy 3.0
#define BLOCKSIZE 32.0
#define TILE_WIDTH 16

__global__ void vecAdd(int *A, int *B, int *C, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //int i = threadIdx.x;

            //blockIdx.x;
    if (i < n){
        C[i] = A[i] + B[i];
        // printf("%d. %d + %d = %d\n",i, A[i], B[i], C[i]);
    }
}

__global__ void MatrixSumKernel(int *d_M, int *d_N, int *d_P, int Width){
    int Row = blockIdx.x * blockDim.x + threadIdx.x;
    int Col = blockIdx.y * blockDim.y + threadIdx.y;
    if ((Row <= Width) && (Col <= Width)){
        d_P[Row*Width + Col] = d_M[Row*Width +  Col] + d_N[Row*Width +  Col];
        // printf("%d\n", d_P[Row*Width + Col]);
    }
  
}

__global__ void MatrixMulKernel(int *d_M, int *d_N, int *d_P, int Width){
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((Row < Width) && (Col < Width)){
        int Pvalue = 0;
        int k;
        for (k =0; k<Width; ++k){
            Pvalue += d_M[Row*Width+k]*d_N[k*Width+Col];
        }
        d_P[Row*Width+Col] = Pvalue;
    }
}

__global__ void MatrixMulKernelSec(int *d_M, int *d_N, int *d_P, int Width){
    __shared__ int Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ int Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the d_P element to work on

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;

    // Loop over the d_M and d_N tiles required to compute d_P element
    int m;
    for (m = 0; m < Width/TILE_WIDTH; ++m)
    {
        
        // Coolaborative loading of d_M and d_N tiles into shared memory

        Mds[ty][tx] = d_M[Row*Width + m*TILE_WIDTH + tx];
        Mds[ty][tx] = d_M[(m*TILE_WIDTH + ty)*Width + Col];
        __syncthreads();
        int k;
        for (k = 0; k < TILE_WIDTH; ++k)
        {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    d_P[Row*Width + Col] = Pvalue;

}

void MatrixSum(int *d_M, int *d_N, int *d_P, int Width){
    int i; 
  int n = sqrt(Width);
    for (i = 0; i < n; ++i)
    {
        /* code */
        int j;
        for (j = 0; j < n; ++j)
        {
            /* code */
            d_P[i*n + j] = d_M[i*n +  j] + d_N[i*n +  j];
        }
    }
  
}



int vectorAddGPU( int *A, int *B, int *C, int n){
    int size = n*sizeof(int);
    int *d_A, *d_B, *d_C;
    //Reservo Memoria en el dispositivo
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);
    //Copio los datos al device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    // Ejecuto el Kernel (del dispositivo)
    int dimGrid = ceil(SIZEx/BLOCKSIZE);
    printf(" DimGrid %d\n", dimGrid);
    vecAdd<<< dimGrid, BLOCKSIZE >>>(d_A, d_B, d_C, n);
    // vecAdd<<< DIMGRID, HILOS >>>
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
  
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}

int matrixAddGPU( int *A, int *B, int *C, int n){
    int size = n*sizeof(int);
    int *d_A, *d_B, *d_C; n=sqrt(n);
    //Reservo Memoria en el dispositivo
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);
    //Copio los datos al device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    // Ejecuto el Kernel (del dispositivo)
    //int dimGrid = ceil(SIZE/BLOCKSIZE);
  
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
    dim3 dimGrid(ceil(SIZEx/BLOCKSIZE),ceil(SIZEy/BLOCKSIZE) , 1);
  
    printf(" DimGrid %f\n", ceil(SIZEx/BLOCKSIZE));
    MatrixSumKernel<<< dimGrid, dimBlock >>>(d_A, d_B, d_C, n);
    // vecAdd<<< DIMGRID, HILOS >>>
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}

int matrixMulGPU( int *A, int *B, int *C, int n){
    int size = n*sizeof(int);
    int *d_A, *d_B, *d_C; n=sqrt(n);
    //Reservo Memoria en el dispositivo
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);
    //Copio los datos al device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    // Ejecuto el Kernel (del dispositivo)
    //int dimGrid = ceil(SIZE/BLOCKSIZE);
  
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
    dim3 dimGrid(ceil(n/BLOCKSIZE),ceil(n/BLOCKSIZE) , 1);
  
    printf(" DimGrid %f\n", ceil(SIZEx/BLOCKSIZE));
    MatrixMulKernel<<< dimGrid, dimBlock >>>(d_A, d_B, d_C, n);
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}

int vectorAddCPU( int *A, int *B, int *C, int n){
    int i;
    for(i=0;i< n; i++){
        C[i]=A[i]+B[i];

    }
    return 0;
}

int main(){
    int j; int imp, imp2;
    int SIZES[] = {4};
    for (j = 0; j < sizeof(SIZES)/sizeof(SIZES[0]); ++j)
    {
        int *A=(int *) malloc(SIZES[j]*sizeof(int));
        int *B=(int *) malloc(SIZES[j]*sizeof(int));
        int *C=(int *) malloc(SIZES[j]*sizeof(int));
        int Row = sqrt(SIZES[j]);
        int Col = sqrt(SIZES[j]);
        clock_t inicioCPU, inicioGPU,finCPU, finGPU;
        int i;
        printf("Tamano matriz %d\n", SIZES[j]);
        for(i=0;i< SIZES[j]; i++){
            A[i]=rand()%21;
            B[i]=rand()%21;
            // A[i]=srand(time(NULL));
            // B[i]=srand(time(NULL));

        }
        // Ejecuto por GPU
        inicioGPU=clock();
        // matrixAddGPU(A, B, C, SIZES[j]);
        matrixMulGPU(A, B, C, SIZES[j]);
        finGPU = clock();
        // Ejecuto por CPU
        inicioCPU=clock();
        // MatrixSum(A, B, C, SIZES[j]);
        finCPU=clock();

        printf("Size %d\n", SIZES[j]);
        printf("El tiempo GPU es: %f\n",(double)(finGPU - inicioGPU) / CLOCKS_PER_SEC);
        printf("El tiempo CPU es: %f\n",(double)(finCPU - inicioCPU) / CLOCKS_PER_SEC);
    
        
        for (imp = 0; imp < Row; ++imp)
        {
            for (imp2 = 0; imp2 < Col; ++imp2)
            {
                printf("%d ", A[imp*Row+imp2]);
            }
            printf("\n");
        }
        printf("\n");
        for (imp = 0; imp < Row; ++imp)
        {
            for (imp2 = 0; imp2 < Col; ++imp2)
            {
                printf("%d ", B[imp*Row+imp2]);
            }
            printf("\n");
        }
        printf("\n");

        for (imp = 0; imp < Row; ++imp)
        {
            for (imp2 = 0; imp2 < Col; ++imp2)
            {
                printf("%d ", C[imp*Row+imp2]);
            }
            printf("\n");
        }
    
        free(A);
        free(B);
    }
    return 0;
}