//#include <torch/extension.h>
#include<torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

/*******************************************************************************************
 * 							The image transformation function
 *******************************************************************************************/
template<typename Real>
 __global__ void Kernel(
    Real* __restrict__ Lkernel,
    Real* __restrict__ Akernel,
    int ImageSizex,
    int ImageSizey,
    int ImageSizez,
    double alpha,
    double gamma)
{
    double A;
	double pi = 3.1415926;
    unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned idy = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned idz = blockIdx.z*blockDim.z+threadIdx.z;
    
    if(idx < ImageSizex && idy<ImageSizey &&idz<ImageSizez)
    {
        A = gamma + 2 * alpha * ( ImageSizex* ImageSizex) * (1.0 - cos(2 * pi*idx /  ImageSizex))
                  + 2 * alpha * ( ImageSizey* ImageSizey) * (1.0 - cos(2 * pi*idy /  ImageSizey))
                  + 2 * alpha * ( ImageSizez* ImageSizez) * (1.0 - cos(2 * pi*idz /  ImageSizez));

        Lkernel[idx + idy*ImageSizex + idz*ImageSizex*ImageSizey] = A;
        Akernel[idx + idy*ImageSizex + idz*ImageSizex*ImageSizey] = pow(A, -2);
    }
}

std::vector<at::Tensor> KernelInitialize_cuda(
    at::Tensor VFx,
    double alpha,
    double gamma){
        at::Tensor Lkernel = at::zeros_like(VFx);
        at::Tensor Akernel = at::zeros_like(VFx);

        const int uint = 4;
	    const dim3 gridvirtual((VFx.size(2) + uint - 1) / uint, (VFx.size(1)  + uint - 1) / uint, (VFx.size(0)  + uint - 1) / uint);
        const dim3 blockvirtual(uint, uint, uint);
        
        AT_DISPATCH_FLOATING_TYPES(VFx.type(), "kernel_initialize", ([&] {
            Kernel<scalar_t><<<gridvirtual, blockvirtual>>>(
                Lkernel.data<scalar_t>(),
                Akernel.data<scalar_t>(),
                VFx.size(2),
                VFx.size(1),
                VFx.size(0),
                alpha,
                gamma);
            }));

            
    return {Lkernel, Akernel};
}
