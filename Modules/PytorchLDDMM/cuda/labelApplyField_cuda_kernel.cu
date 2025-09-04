//#include <torch/extension.h>
#include<torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

/*******************************************************************************************
 * 							The image transformation function
 *******************************************************************************************/
 template<typename Real>
 __global__ void transformLabel(
    Real* __restrict__ OutputImage,
    Real* __restrict__ InputImage,
    const Real* __restrict__ DisplaceFieldx,
    const Real* __restrict__ DisplaceFieldy,
    const Real* __restrict__ DisplaceFieldz,
    int ImageSizex,
    int ImageSizey,
    int ImageSizez,
    int xsize,
    int ysize,
    int zsize,
    float ForwardImageOriginx,
    float ForwardImageOriginy,
    float ForwardImageOriginz,
    float MovingImageOriginx,
    float MovingImageOriginy,
    float MovingImageOriginz,
    float scale)
{
    //transform the point from index to physical
    #define MovingImageDimension 3
    unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned idy = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned idz = blockIdx.z*blockDim.z+threadIdx.z;
    if(idx<ImageSizex&&idy<ImageSizey&&idz<ImageSizez)
    {
        float PhysicalePoint[3];
        //transform the index to the fixed image space by ysing fixed image origin, i.e. forwardImgeOrigin.
        PhysicalePoint[0] = idx + ForwardImageOriginx;
        PhysicalePoint[1] = idy + ForwardImageOriginy;
        PhysicalePoint[2] = idz + ForwardImageOriginz;
        const int m_Neighbors = 1 << MovingImageDimension;
        int baseIndex[MovingImageDimension];
        float distance[MovingImageDimension];
        int neighIndex[MovingImageDimension];
        float StartContinuousIndex[MovingImageDimension] = {-0.5/scale + ForwardImageOriginx, -0.5/scale + ForwardImageOriginy, -0.5/scale + ForwardImageOriginz};
        float EndContinuousIndex[MovingImageDimension];
        float EndIndex[MovingImageDimension];
        float index[MovingImageDimension];
        double DisplaceValue[3] = {0,0,0};

        int EndImageIndex[MovingImageDimension];
        int StartImageIndex[MovingImageDimension];
        float ImageStartContinuousIndex[MovingImageDimension];
        float ImageEndContinuousIndex[MovingImageDimension];
        double index_Im[MovingImageDimension];

        //interpolation the image
        ImageEndContinuousIndex[0] = ImageSizex - 0.5;
        ImageEndContinuousIndex[1] = ImageSizey - 0.5;
        ImageEndContinuousIndex[2] = ImageSizez - 0.5;
        EndImageIndex[0] = ImageSizex - 1;
        EndImageIndex[1] = ImageSizey - 1;
        EndImageIndex[2] = ImageSizez - 1;
        for (unsigned int dim = 0; dim < MovingImageDimension; ++dim)
        {
            ImageStartContinuousIndex[dim] = -0.5;
            StartImageIndex[dim] = 0;
        }


        EndContinuousIndex[0] = (xsize-0.5)/scale + ForwardImageOriginx;
        EndContinuousIndex[1] = (ysize-0.5)/scale + ForwardImageOriginy;
        EndContinuousIndex[2] = (zsize-0.5)/scale + ForwardImageOriginz;
        EndIndex[0] = xsize -1;
        EndIndex[1] = ysize -1;
        EndIndex[2] = zsize -1;

        index[0] = PhysicalePoint[0];
        index[1] = PhysicalePoint[1];
        index[2] = PhysicalePoint[2];

        bool IsInsideBuffer = true;
        for (unsigned int dim = 0; dim<MovingImageDimension; dim++)
        {
            if(index[dim]<StartContinuousIndex[dim] || index[dim]>EndContinuousIndex[dim])
            {
                IsInsideBuffer = false;
            }
        }

        float StartIndex[MovingImageDimension] = {0,0,0};
        if(IsInsideBuffer)
        {
            index[0] = (index[0] - ForwardImageOriginx)*scale;
            index[1] = (index[1] - ForwardImageOriginy)*scale;
            index[2] = (index[2] - ForwardImageOriginz)*scale;
            //index[1] = 0;
            //index[2] = 0;
            for (unsigned int dim = 0; dim < MovingImageDimension; ++dim)
            {
                baseIndex[dim] = floor(index[dim]);
                distance[dim] = index[dim] - baseIndex[dim];
            }

            float totalOverlap = 0;
            for ( unsigned int counter = 0; counter < m_Neighbors; ++counter )
            {
                float overlap = 1.0;    // fraction overlap
                unsigned int upper = counter;  // each bit indicates upper/lower neighbour
                for (unsigned int dim = 0; dim < MovingImageDimension; ++dim)
                {
                    neighIndex[dim] = baseIndex[dim];
                }
                for ( unsigned int dim = 0; dim < MovingImageDimension; ++dim )
                {
                    if ( upper & 1 )
                    {

                    ++(neighIndex[dim]);
                    if ( neighIndex[dim] > EndIndex[dim] )
                    {
                        neighIndex[dim] = EndIndex[dim];
                    }
                        overlap *= distance[dim];
                    }
                    else
                    {
                        if ( neighIndex[dim] < StartIndex[dim] )
                        {
                            neighIndex[dim] = StartIndex[dim];
                        }
                            overlap *= 1.0 - distance[dim];
                    }

                    upper >>= 1;
                }

                if(overlap)
                {
                    DisplaceValue[0] += DisplaceFieldx[neighIndex[2]*xsize*ysize +  neighIndex[1]*xsize + neighIndex[0]] * overlap;
                    DisplaceValue[1] += DisplaceFieldy[neighIndex[2]*xsize*ysize +  neighIndex[1]*xsize + neighIndex[0]] * overlap;
                    DisplaceValue[2] += DisplaceFieldz[neighIndex[2]*xsize*ysize +  neighIndex[1]*xsize + neighIndex[0]] * overlap;
                    totalOverlap += overlap;
                }
                if (totalOverlap == 1.0)
                {
                    break;
                }
            }
        }
        //all of the transform are done in the physical space.
        index_Im[0] = PhysicalePoint[0] + DisplaceValue[0];
        index_Im[1] = PhysicalePoint[1] + DisplaceValue[1];
        index_Im[2] = PhysicalePoint[2] + DisplaceValue[2];

        //the function of following codes is transform the physical point to the Moving continuouse index spacing by using the moving image origin.
        index_Im[0] -= MovingImageOriginx;
        index_Im[1] -= MovingImageOriginy;
        index_Im[2] -= MovingImageOriginz;

        bool IsInsideBufferIm = true;
        float ImageValue = 0;
        double nindex[MovingImageDimension];
        for (unsigned int dim = 0; dim < MovingImageDimension; ++dim)
        {
            if(index_Im[dim] < ImageStartContinuousIndex[dim] || index_Im[dim] > ImageEndContinuousIndex[dim])
                IsInsideBufferIm = false;
        }
        if (!IsInsideBufferIm)
        {
            for ( unsigned int j = 0; j < MovingImageDimension; j++ )
            {
                nindex[j] = index_Im[j];
                float size = ImageEndContinuousIndex[j] - ImageStartContinuousIndex[j];
                while(nindex[j] > EndImageIndex[j])
                {
                    nindex[j] -= size;
                }
                while(nindex[j] < StartImageIndex[j])
                {
                    nindex[j] += size;
                }
            }
        }
        else
        {
            for ( unsigned int j = 0; j < MovingImageDimension; j++ )
            {
                nindex[j] = index_Im[j];
            }
        }
        
        //========================================================================
        for (unsigned int dim = 0; dim < MovingImageDimension; ++dim)
        {
            neighIndex[dim] = floor(nindex[dim] + 0.5);
            if ( neighIndex[dim] > EndImageIndex[dim] )
            {
                neighIndex[dim] = EndImageIndex[dim];
            }
            if ( neighIndex[dim] < StartIndex[dim] )
            {
            neighIndex[dim] = StartIndex[dim];
            }
        }
        ImageValue = InputImage[neighIndex[2]*ImageSizex*ImageSizey +  neighIndex[1]*ImageSizex + neighIndex[0]];
        OutputImage[idz*ImageSizex*ImageSizey+idy*ImageSizex+idx] = ImageValue;
    }
}


at::Tensor labelApplyField_cuda(
    torch::Tensor MI,
    torch::Tensor DFx,
    torch::Tensor DFy,
    torch::Tensor DFz,
    float ForwardImageOriginx,
    float ForwardImageOriginy,
    float ForwardImageOriginz,
    float MovingImageOriginx,
    float MovingImageOriginy,
    float MovingImageOriginz,
    float scale)
{
	const int uint = 8;
	const dim3 grid((MI.size(2)+uint-1)/uint,(MI.size(1)+uint-1)/uint,(MI.size(0)+uint-1)/uint);
	const dim3 block(uint,uint,uint);

    at::Tensor FI = at::zeros_like(MI);
    AT_DISPATCH_FLOATING_TYPES(MI.type(), "image apply field", ([&] {
        transformLabel<scalar_t><<<grid, block>>>(
            FI.data<scalar_t>(),
            MI.data<scalar_t>(),
			DFx.data<scalar_t>(),
			DFy.data<scalar_t>(),
			DFz.data<scalar_t>(),
			MI.size(2),
			MI.size(1),
			MI.size(0),
			DFx.size(2),
			DFx.size(1),
			DFx.size(0),
            ForwardImageOriginx,
            ForwardImageOriginy,
            ForwardImageOriginz,
            MovingImageOriginx,
            MovingImageOriginy,
            MovingImageOriginz,
            scale);
        }));

    return FI;
}
