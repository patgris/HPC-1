void vectorAdd( int *A, int *B, int *C, int n){
	int size = n*sizeof(int);
	float *d_A, *d_B, *d_C;
	//Reservo Memoria en el dispositivo
	cudaMalloc((void **)&d_A, size);
	cudaMalloc((void **)&d_B, size);
	cudaMalloc((void **)&d_C, size);
	//Copio los datos al device
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
	// Ejecuto el Kernel (del dispositivo)
	vecAdd(<<< 1, n >>>)(d_A, d_B, d_C, n);
	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	return 0;
}