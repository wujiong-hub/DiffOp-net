//#include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include"cufft.h"

template<typename Real>
 __global__ void Gradient(
    Real* __restrict__ OutputImagex,
    Real* __restrict__ OutputImagey,
    Real* __restrict__ OutputImagez,
    const Real* __restrict__ InputImage,
    int ImageSizex,
    int ImageSizey,
    int ImageSizez)
{
    unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned idy = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned idz = blockIdx.z*blockDim.z+threadIdx.z;
    if(idx < ImageSizex && idy<ImageSizey &&idz<ImageSizez)
    {
        int xDirection[2];
        int yDirection[2];
        int zDirection[2];
        xDirection[0] = idx - 1;
        xDirection[1] = idx + 1;
        yDirection[0] = idy - 1;
        yDirection[1] = idy + 1;
        zDirection[0] = idz - 1;
        zDirection[1] = idz + 1;

        if(xDirection[0]<0 ) xDirection[0] = idx;
        if(yDirection[0]<0 ) yDirection[0] = idy;
        if(zDirection[0]<0 ) zDirection[0] = idz;
        if(xDirection[1]>ImageSizex-1 ) xDirection[1] = idx;
        if(yDirection[1]>ImageSizey-1 ) yDirection[1] = idy;
        if(zDirection[1]>ImageSizez-1 ) zDirection[1] = idz;

        OutputImagex[idx+idy*ImageSizex+idz*ImageSizex*ImageSizey] = (InputImage[xDirection[1]+idy*ImageSizex+idz*ImageSizex*ImageSizey]
                                                                            - InputImage[xDirection[0]+idy*ImageSizex+idz*ImageSizex*ImageSizey])/2;
        OutputImagey[idx+idy*ImageSizex+idz*ImageSizex*ImageSizey] = (InputImage[idx+yDirection[1]*ImageSizex+idz*ImageSizex*ImageSizey]
                                                                            - InputImage[idx+yDirection[0]*ImageSizex+idz*ImageSizex*ImageSizey])/2;
        OutputImagez[idx+idy*ImageSizex+idz*ImageSizex*ImageSizey] = (InputImage[idx+idy*ImageSizex+zDirection[1]*ImageSizex*ImageSizey]
                                                                            - InputImage[idx+idy*ImageSizex+zDirection[0]*ImageSizex*ImageSizey])/2;
    }
}

template<typename Real>
__global__ void ComputJacobian(
    Real* __restrict__ Jacobian,
    const Real* __restrict__ DisplaceFieldx,
    const Real* __restrict__ DisplaceFieldy,
    const Real* __restrict__ DisplaceFieldz,
    int ImageSizex,
    int ImageSizey,
    int ImageSizez)
{
    unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned idy = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned idz = blockIdx.z*blockDim.z+threadIdx.z;
    double Detxx, Detxy, Detxz, Detyx, Detyy, Detyz, Detzx, Detzy, Detzz;
    if(idx < ImageSizex && idy<ImageSizey &&idz<ImageSizez)
    {
        int xDirection[2];
        int yDirection[2];
        int zDirection[2];
        xDirection[0] = idx - 1;
        xDirection[1] = idx + 1;
        yDirection[0] = idy - 1;
        yDirection[1] = idy + 1;
        zDirection[0] = idz - 1;
        zDirection[1] = idz + 1;

        if(xDirection[0]<0 ) xDirection[0] = idx;
        if(yDirection[0]<0 ) yDirection[0] = idy;
        if(zDirection[0]<0 ) zDirection[0] = idz;
        if(xDirection[1]>ImageSizex-1 ) xDirection[1] = idx;
        if(yDirection[1]>ImageSizey-1 ) yDirection[1] = idy;
        if(zDirection[1]>ImageSizez-1 ) zDirection[1] = idz;

        Detxx = double(DisplaceFieldx[xDirection[1]+idy*ImageSizex+idz*ImageSizex*ImageSizey]- DisplaceFieldx[xDirection[0]+idy*ImageSizex+idz*ImageSizex*ImageSizey])/8+1;
        Detyx = double(DisplaceFieldx[idx+yDirection[1]*ImageSizex+idz*ImageSizex*ImageSizey]- DisplaceFieldx[idx+yDirection[0]*ImageSizex+idz*ImageSizex*ImageSizey])/8;
        Detzx = double(DisplaceFieldx[idx+idy*ImageSizex+zDirection[1]*ImageSizex*ImageSizey]- DisplaceFieldx[idx+idy*ImageSizex+zDirection[0]*ImageSizex*ImageSizey])/8;
        Detxy = double(DisplaceFieldy[xDirection[1]+idy*ImageSizex+idz*ImageSizex*ImageSizey]- DisplaceFieldy[xDirection[0]+idy*ImageSizex+idz*ImageSizex*ImageSizey])/8;
        Detyy = double(DisplaceFieldy[idx+yDirection[1]*ImageSizex+idz*ImageSizex*ImageSizey]- DisplaceFieldy[idx+yDirection[0]*ImageSizex+idz*ImageSizex*ImageSizey])/8+1;
        Detzy = double(DisplaceFieldy[idx+idy*ImageSizex+zDirection[1]*ImageSizex*ImageSizey]- DisplaceFieldy[idx+idy*ImageSizex+zDirection[0]*ImageSizex*ImageSizey])/8;
        Detxz = double(DisplaceFieldz[xDirection[1]+idy*ImageSizex+idz*ImageSizex*ImageSizey]- DisplaceFieldz[xDirection[0]+idy*ImageSizex+idz*ImageSizex*ImageSizey])/8;
        Detyz = double(DisplaceFieldz[idx+yDirection[1]*ImageSizex+idz*ImageSizex*ImageSizey]- DisplaceFieldz[idx+yDirection[0]*ImageSizex+idz*ImageSizex*ImageSizey])/8;
        Detzz = double(DisplaceFieldz[idx+idy*ImageSizex+zDirection[1]*ImageSizex*ImageSizey]- DisplaceFieldz[idx+idy*ImageSizex+zDirection[0]*ImageSizex*ImageSizey])/8+1;

        Jacobian[idx+idy*ImageSizex+idz*ImageSizex*ImageSizey] = Detxx*Detyy*Detzz + Detxy*Detyz*Detzx + Detxz*Detzy*Detyx
                                    - Detxz*Detyy*Detzx - Detxy*Detzz*Detyx - Detzy*Detxx*Detyz;
    }
}

template<typename Real>
__global__ void MultiplyJacobianAndConst(
    Real* __restrict__ resultx,
    Real* __restrict__ resulty,
    Real* __restrict__ resultz,
    Real* __restrict__ Jacobian,
    Real* __restrict__ MetricDerivativex,
    Real* __restrict__ MetricDerivativey,
    Real* __restrict__ MetricDerivativez,
    int ImageSizex,
    int ImageSizey,
    int ImageSizez,
    float constData)
{
    unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned idy = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned idz = blockIdx.z*blockDim.z+threadIdx.z;
    if(idx < ImageSizex && idy<ImageSizey &&idz<ImageSizez)
    {
        resultx[idx + idy*ImageSizex + idz*ImageSizex*ImageSizey] = Jacobian[idx + idy*ImageSizex + idz*ImageSizex*ImageSizey]*MetricDerivativex[idx + idy*ImageSizex + idz*ImageSizex*ImageSizey]*constData;
        resulty[idx + idy*ImageSizex + idz*ImageSizex*ImageSizey] = Jacobian[idx + idy*ImageSizex + idz*ImageSizex*ImageSizey]*MetricDerivativey[idx + idy*ImageSizex + idz*ImageSizex*ImageSizey]*constData;
        resultz[idx + idy*ImageSizex + idz*ImageSizex*ImageSizey] = Jacobian[idx + idy*ImageSizex + idz*ImageSizex*ImageSizey]*MetricDerivativez[idx + idy*ImageSizex + idz*ImageSizex*ImageSizey]*constData;
    }
}

template<typename Real>
__global__ void MultiplyKernel(cufftComplex *ComplexImage, cufftReal *Kernel, unsigned int vectorSize) {
	unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx < vectorSize){
		ComplexImage[idx].x = ComplexImage[idx].x*Kernel[idx];
		ComplexImage[idx].y = ComplexImage[idx].y*Kernel[idx];
	}
}

template<typename Real>
__global__ void Scale(cufftReal *data, unsigned vectorSize) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx < vectorSize){
		data[idx] = data[idx]/vectorSize;
	}
}

