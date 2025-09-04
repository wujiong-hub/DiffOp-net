//#include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template<typename Real>
__global__ void linear_interpolate_4D(
    Real* __restrict__ DisplaceFieldx,
    Real* __restrict__ DisplaceFieldy,
    Real* __restrict__ DisplaceFieldz,
    const Real* __restrict__ velocityFieldx,
    const Real* __restrict__ velocityFieldy,
    const Real* __restrict__ velocityFieldz,
    float LowerTimeBound,
    float DeltaTime,
    float TimeSpan,
    float TimeOrigion,
    int NumberOfIntegrationSteps,
    int xDsize,
    int yDsize,
    int xsize,
    int ysize,
    int zsize,
    int tsize,
    int Originx,
    int Originy,
    int Originz,
    float scale)
{
    #define ImageDimension 4
    unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned idy = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned idz = blockIdx.z*blockDim.z+threadIdx.z;

    if(idx<xsize&&idy<ysize&&idz<zsize)
    {
        float currentSpatialPoint[3];
        float timePoint = LowerTimeBound + 0.5*DeltaTime;
        currentSpatialPoint[0] = idx/scale + Originx;
        currentSpatialPoint[1] = idy/scale + Originy;
        currentSpatialPoint[2] = idz/scale + Originz;
        float index[ImageDimension];
        float totalDisplacement[3] = {0,0,0};
        for (unsigned int j = 0; j < NumberOfIntegrationSteps; j++, timePoint += DeltaTime)
        {
            float displacement[3]={0,0,0};
            float spatialPoint[3];
            for(unsigned int k = 0; k < 3; k++)spatialPoint[k] = currentSpatialPoint[k];

            for(unsigned int i = 0; i < 3; i++)
            {

                for(unsigned int k = 0; k < 3; k++)displacement[k] = displacement[k]*0.5;
                for(unsigned int k = 0; k < 3; k++)spatialPoint[k] = currentSpatialPoint[k] + displacement[k];
                for(unsigned int k = 0; k < 3; k++)index[k] = spatialPoint[k];
                index[3] = TimeSpan*timePoint + TimeOrigion;
                const int m_Neighbors = 1 << ImageDimension;
                float baseIndex[ImageDimension];
                float distance[ImageDimension];
                int neighIndex[ImageDimension];

                float StartContinuousIndex[ImageDimension] = {-0.5/scale+Originx, -0.5/scale+Originy, -0.5/scale+Originz,-0.5};
                float EndContinuousIndex[ImageDimension];
                float EndIndex[ImageDimension];
                float StartIndex[ImageDimension] = {0,0,0,0};
                EndIndex[0] = xsize -1;
                EndIndex[1] = ysize -1;
                EndIndex[2] = zsize -1;
                EndIndex[3] = tsize -1;
                EndContinuousIndex[0] = (xsize-0.5)/scale + Originx;
                EndContinuousIndex[1] = (ysize-0.5)/scale + Originy;
                EndContinuousIndex[2] = (zsize-0.5)/scale + Originz;
                EndContinuousIndex[3] = tsize-0.5;

                bool IsInsideBuffer = true;

                for ( unsigned int j = 0; j < ImageDimension; j++ )
                {
                    /* Test for negative of a positive so we can catch NaN's. */
                    if ( ! (index[j] >= StartContinuousIndex[j] && index[j] < EndContinuousIndex[j] ) )
                    {
                        IsInsideBuffer = false;
                    }
                }

                if (!IsInsideBuffer)
                {
                    index[0] = (index[0]-Originx)*scale;
                    index[1] = (index[1]-Originy)*scale;
                    index[2] = (index[2]-Originz)*scale;

                    for (unsigned int dim = 0; dim < ImageDimension; ++dim)
                    {	
                        index[dim] = round(index[dim]);
                    }

                    for ( unsigned int j = 0; j < ImageDimension; j++ )
                    {
                        while(index[j] > EndIndex[j])
                        {
                            index[j] = EndIndex[j];
                        }
                        while(index[j] < StartIndex[j])
                        {
                            index[j] = StartIndex[j];
                        }
                    }
                }
                else
                {
                    index[0] = (index[0]-Originx)*scale;
                    index[1] = (index[1]-Originy)*scale;
                    index[2] = (index[2]-Originz)*scale;
                }

                for (unsigned int dim = 0; dim < ImageDimension; ++dim)
                {
                    baseIndex[dim] = floor(index[dim]);
                    distance[dim] = index[dim] - baseIndex[dim];
                }

                float velocityValue[3] = {0,0,0};
                float totalOverlap = 0;
                for ( unsigned int counter = 0; counter < m_Neighbors; ++counter )
                {
                    float overlap = 1.0;    // fraction overlap
                    unsigned int upper = counter;  // each bit indicates upper/lower neighbour
                    for (unsigned int dim = 0; dim < ImageDimension; ++dim)
                    {
                        neighIndex[dim] = baseIndex[dim];
                    }
                    for ( unsigned int dim = 0; dim < ImageDimension; ++dim )
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
                            // Take care of the case where the pixel is just
                            // in the outer lower boundary of the image grid.
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
                        velocityValue[0] += velocityFieldx[neighIndex[3]*xsize*ysize*zsize + neighIndex[2]*xsize*ysize +  neighIndex[1]*xsize + neighIndex[0]] * overlap;
                        velocityValue[1] += velocityFieldy[neighIndex[3]*xsize*ysize*zsize + neighIndex[2]*xsize*ysize +  neighIndex[1]*xsize + neighIndex[0]] * overlap;
                        velocityValue[2] += velocityFieldz[neighIndex[3]*xsize*ysize*zsize + neighIndex[2]*xsize*ysize +  neighIndex[1]*xsize + neighIndex[0]] * overlap;
                        totalOverlap += overlap;
                    }
                    if (totalOverlap == 1.0)
                    {
                        break;
                    }
                }
                displacement[0] = velocityValue[0] * DeltaTime;
                displacement[1] = velocityValue[1] * DeltaTime;
                displacement[2] = velocityValue[2] * DeltaTime;

            }
            currentSpatialPoint[0] += displacement[0];
            currentSpatialPoint[1] += displacement[1];
            currentSpatialPoint[2] += displacement[2];
            totalDisplacement[0] += displacement[0];
            totalDisplacement[1] += displacement[1];
            totalDisplacement[2] += displacement[2];
        }
        DisplaceFieldx[idz*xDsize*yDsize+idy*xDsize+idx] = totalDisplacement[0];
        DisplaceFieldy[idz*xDsize*yDsize+idy*xDsize+idx] = totalDisplacement[1];
        DisplaceFieldz[idz*xDsize*yDsize+idy*xDsize+idx] = totalDisplacement[2];
    }
}

std::vector<at::Tensor> intergrateVelocity_cuda(
    const at::Tensor velocityFieldx,
    const at::Tensor velocityFieldy,
    const at::Tensor velocityFieldz,
    float LowerTimeBound,
    int Originx,
    int Originy,
    int Originz,
    float DeltaTime,   //(m_UpperTimeBound-m_LowerTimeBound)/m_NumberOfIntegrationSteps;
    float TimeSpan,   //m_NumberOfTimeSteps-1
    float TimeOrigion,  //0
    int NumberOfIntegrationSteps,
    float scale)
{

    at::Tensor DisplaceFieldx = at::zeros({velocityFieldx.size(1), velocityFieldx.size(2), velocityFieldx.size(3)}, velocityFieldx.type());
    at::Tensor DisplaceFieldy = at::zeros({velocityFieldx.size(1), velocityFieldx.size(2), velocityFieldx.size(3)}, velocityFieldx.type());
    at::Tensor DisplaceFieldz = at::zeros({velocityFieldx.size(1), velocityFieldx.size(2), velocityFieldx.size(3)}, velocityFieldx.type());

    //std::cout<<"Dissize:  "<<DisplaceFieldx.size(0)<<" "<<DisplaceFieldx.size(1)<<" "<<DisplaceFieldx.size(2)<<std::endl;
    const int uint = 8;
    const dim3 grid((DisplaceFieldx.size(2)+uint-1)/uint,(DisplaceFieldx.size(1)+uint-1)/uint,(DisplaceFieldx.size(0)+uint-1)/uint);
    const dim3 block(uint,uint,uint);

    AT_DISPATCH_FLOATING_TYPES(velocityFieldx.type(), "velocity_integration", ([&] {
        linear_interpolate_4D<scalar_t><<<grid, block>>>(
            DisplaceFieldx.data<scalar_t>(),
            DisplaceFieldy.data<scalar_t>(),
            DisplaceFieldz.data<scalar_t>(),
            velocityFieldx.data<scalar_t>(),
            velocityFieldy.data<scalar_t>(),
            velocityFieldz.data<scalar_t>(),
            LowerTimeBound,
            DeltaTime,
            TimeSpan,
            TimeOrigion,
            NumberOfIntegrationSteps,
            velocityFieldx.size(3),
            velocityFieldx.size(2),
            velocityFieldx.size(3),
            velocityFieldx.size(2),
            velocityFieldx.size(1),
            velocityFieldx.size(0),
            Originx,
            Originy,
            Originz,
            scale);}));
    //std::cout<<velocityFieldx.size(0)<<" "<<velocityFieldx.size(1)<<" "<<velocityFieldx.size(2)<<" "<<velocityFieldx.size(3)<<std::endl;
    return {DisplaceFieldx, DisplaceFieldy, DisplaceFieldz};

}
