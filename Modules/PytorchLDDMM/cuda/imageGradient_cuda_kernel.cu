//#include <torch/extension.h>
#include<torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>


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

std::vector<at::Tensor> imageGradient_cuda(
    at::Tensor I)
{
    const int uint = 4;
	const dim3 gridvirtual((I.size(2) + uint - 1) / uint, (I.size(1)  + uint - 1) / uint, (I.size(0)  + uint - 1) / uint);
    const dim3 blockvirtual(uint, uint, uint);
    
    at::Tensor d_x = at::zeros_like(I);
    at::Tensor d_y = at::zeros_like(I);
    at::Tensor d_z = at::zeros_like(I);

    AT_DISPATCH_FLOATING_TYPES(I.type(), "gradient_of_image", ([&] {
        Gradient<scalar_t><<<gridvirtual, blockvirtual>>>(
            d_x.data<scalar_t>(),
            d_y.data<scalar_t>(),
            d_z.data<scalar_t>(),
            I.data<scalar_t>(),
            I.size(2),
            I.size(1),
            I.size(0));
        }));

    return {d_x, d_y, d_z};
}
