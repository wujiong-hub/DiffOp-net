#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include<torch/types.h>

template<typename Real>
inline __device__ Real
GetDisplacement(
	const Real* DisplaceField,
	int Dsizex,
	int Dsizey,
	int Dsizez,
	int xindex,
	int yindex,
	int zindex,
	int ImageSizex,
	int ImageSizey,
	int ImageSizez,
	float scale)
{
	double  DisplaceValue = 0;
	double index[3];

	if (xindex >= 0 && xindex <= ImageSizex - 1 && yindex >= 0 && yindex <= ImageSizey - 1 && zindex >= 0 && zindex <= ImageSizez - 1)
	{
		index[0] = xindex * scale;
		index[1] = yindex * scale;
		index[2] = zindex * scale;

		int xBas0, xBas1, yBas0, yBas1, zBas0, zBas1;
		float perc[8];
		float xCom, yCom, zCom;
		float xComi, yComi, zComi;
		float color[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
		float fTlocalx, fTlocaly, fTlocalz;

		fTlocalx = floor(index[0]); fTlocaly = floor(index[1]); fTlocalz = floor(index[2]);

		/* Determine the coordinates of the pixel(s) which will be come the current pixel */
		/* (using linear interpolation) */
		xBas0 = (int)fTlocalx; yBas0 = (int)fTlocaly; zBas0 = (int)fTlocalz;
		xBas1 = xBas0 + 1;      yBas1 = yBas0 + 1;      zBas1 = zBas0 + 1;

		/* Clamp to boundary */
		if (xBas0 < 0) { xBas0 = 0; if (xBas1 < 0) { xBas1 = 0; } }
		if (yBas0 < 0) { yBas0 = 0; if (yBas1 < 0) { yBas1 = 0; } }
		if (zBas0 < 0) { zBas0 = 0; if (zBas1 < 0) { zBas1 = 0; } }
		if (xBas1 > (Dsizex - 1)) { xBas1 = Dsizex - 1; if (xBas0 > (Dsizex - 1)) { xBas0 = Dsizex - 1; } }
		if (yBas1 > (Dsizey - 1)) { yBas1 = Dsizey - 1; if (yBas0 > (Dsizey - 1)) { yBas0 = Dsizey - 1; } }
		if (zBas1 > (Dsizez - 1)) { zBas1 = Dsizez - 1; if (zBas0 > (Dsizez - 1)) { zBas0 = Dsizez - 1; } }

		/*  Get intensities */
#define getcolor_mindex3_float( x, y, z, sizx, sizy, sizz, I) ( I[z*sizx*sizy+y*sizx+x] )
		color[0] = getcolor_mindex3_float(xBas0, yBas0, zBas0, Dsizex, Dsizey, Dsizez, DisplaceField);
		color[1] = getcolor_mindex3_float(xBas0, yBas0, zBas1, Dsizex, Dsizey, Dsizez, DisplaceField);
		color[2] = getcolor_mindex3_float(xBas0, yBas1, zBas0, Dsizex, Dsizey, Dsizez, DisplaceField);
		color[3] = getcolor_mindex3_float(xBas0, yBas1, zBas1, Dsizex, Dsizey, Dsizez, DisplaceField);
		color[4] = getcolor_mindex3_float(xBas1, yBas0, zBas0, Dsizex, Dsizey, Dsizez, DisplaceField);
		color[5] = getcolor_mindex3_float(xBas1, yBas0, zBas1, Dsizex, Dsizey, Dsizez, DisplaceField);
		color[6] = getcolor_mindex3_float(xBas1, yBas1, zBas0, Dsizex, Dsizey, Dsizez, DisplaceField);
		color[7] = getcolor_mindex3_float(xBas1, yBas1, zBas1, Dsizex, Dsizey, Dsizez, DisplaceField);

		/* Linear interpolation constants (percentages) */
		xCom = index[0] - fTlocalx;  yCom = index[1] - fTlocaly;   zCom = index[2] - fTlocalz;

		xComi = (1 - xCom); yComi = (1 - yCom); zComi = (1 - zCom);
		perc[0] = xComi * yComi; perc[1] = perc[0] * zCom; perc[0] = perc[0] * zComi;
		perc[2] = xComi * yCom;  perc[3] = perc[2] * zCom; perc[2] = perc[2] * zComi;
		perc[4] = xCom * yComi;  perc[5] = perc[4] * zCom; perc[4] = perc[4] * zComi;
		perc[6] = xCom * yCom;   perc[7] = perc[6] * zCom; perc[6] = perc[6] * zComi;

		/* Set the current pixel value */
		DisplaceValue = color[0] * perc[0] + color[1] * perc[1] + color[2] * perc[2] + color[3] * perc[3] + color[4] * perc[4] + color[5] * perc[5] + color[6] * perc[6] + color[7] * perc[7];
	}
	else
	{
		DisplaceValue = 0;
	}
	return DisplaceValue;
}


template<typename Real>
inline __device__ Real
ResamplePoint(
	const Real* InputImage,
	int ImageSizex,
	int ImageSizey,
	int ImageSizez,
	double xindex,
	double yindex,
	double zindex)
{
	double ImageValue;
	ImageValue = 0;
	int xBas0, xBas1, yBas0, yBas1, zBas0, zBas1;
	float perc[8];
	float xCom, yCom, zCom;
	float xComi, yComi, zComi;
	float color[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	float fTlocalx, fTlocaly, fTlocalz;

	fTlocalx = floor(xindex); fTlocaly = floor(yindex); fTlocalz = floor(zindex);

	/* Determine the coordinates of the pixel(s) which will be come the current pixel */
	/* (using linear interpolation) */
	xBas0 = (int)fTlocalx; yBas0 = (int)fTlocaly; zBas0 = (int)fTlocalz;
	xBas1 = xBas0 + 1;      yBas1 = yBas0 + 1;      zBas1 = zBas0 + 1;

	/* Clamp to boundary */
	if (xBas0 < 0) { xBas0 = 0; if (xBas1 < 0) { xBas1 = 0; } }
	if (yBas0 < 0) { yBas0 = 0; if (yBas1 < 0) { yBas1 = 0; } }
	if (zBas0 < 0) { zBas0 = 0; if (zBas1 < 0) { zBas1 = 0; } }
	if (xBas1 > (ImageSizex - 1)) { xBas1 = ImageSizex - 1; if (xBas0 > (ImageSizex - 1)) { xBas0 = ImageSizex - 1; } }
	if (yBas1 > (ImageSizey - 1)) { yBas1 = ImageSizey - 1; if (yBas0 > (ImageSizey - 1)) { yBas0 = ImageSizey - 1; } }
	if (zBas1 > (ImageSizez - 1)) { zBas1 = ImageSizez - 1; if (zBas0 > (ImageSizez - 1)) { zBas0 = ImageSizez - 1; } }

	/*  Get intensities */
#define getcolor_mindex3_float( x, y, z, sizx, sizy, sizz, I) ( I[z*sizx*sizy+y*sizx+x] )
	color[0] = getcolor_mindex3_float(xBas0, yBas0, zBas0, ImageSizex, ImageSizey, ImageSizez, InputImage);
	color[1] = getcolor_mindex3_float(xBas0, yBas0, zBas1, ImageSizex, ImageSizey, ImageSizez, InputImage);
	color[2] = getcolor_mindex3_float(xBas0, yBas1, zBas0, ImageSizex, ImageSizey, ImageSizez, InputImage);
	color[3] = getcolor_mindex3_float(xBas0, yBas1, zBas1, ImageSizex, ImageSizey, ImageSizez, InputImage);
	color[4] = getcolor_mindex3_float(xBas1, yBas0, zBas0, ImageSizex, ImageSizey, ImageSizez, InputImage);
	color[5] = getcolor_mindex3_float(xBas1, yBas0, zBas1, ImageSizex, ImageSizey, ImageSizez, InputImage);
	color[6] = getcolor_mindex3_float(xBas1, yBas1, zBas0, ImageSizex, ImageSizey, ImageSizez, InputImage);
	color[7] = getcolor_mindex3_float(xBas1, yBas1, zBas1, ImageSizex, ImageSizey, ImageSizez, InputImage);

	/* Linear interpolation constants (percentages) */
	xCom = xindex - fTlocalx;  yCom = yindex - fTlocaly;   zCom = zindex - fTlocalz;

	xComi = (1 - xCom); yComi = (1 - yCom); zComi = (1 - zCom);
	perc[0] = xComi * yComi; perc[1] = perc[0] * zCom; perc[0] = perc[0] * zComi;
	perc[2] = xComi * yCom;  perc[3] = perc[2] * zCom; perc[2] = perc[2] * zComi;
	perc[4] = xCom * yComi;  perc[5] = perc[4] * zCom; perc[4] = perc[4] * zComi;
	perc[6] = xCom * yCom;   perc[7] = perc[6] * zCom; perc[6] = perc[6] * zComi;

	/* Set the current pixel value */
	ImageValue = color[0] * perc[0] + color[1] * perc[1] + color[2] * perc[2] + color[3] * perc[3] + color[4] * perc[4] + color[5] * perc[5] + color[6] * perc[6] + color[7] * perc[7];

	return ImageValue;
}

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
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned idy = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned idz = blockIdx.z * blockDim.z + threadIdx.z;
	if (idx < ImageSizex && idy < ImageSizey && idz < ImageSizez)
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

		if (xDirection[0] < 0) xDirection[0] = idx;
		if (yDirection[0] < 0) yDirection[0] = idy;
		if (zDirection[0] < 0) zDirection[0] = idz;
		if (xDirection[1] > ImageSizex - 1) xDirection[1] = idx;
		if (yDirection[1] > ImageSizey - 1) yDirection[1] = idy;
		if (zDirection[1] > ImageSizez - 1) zDirection[1] = idz;

		OutputImagex[idx + idy * ImageSizex + idz * ImageSizex * ImageSizey] = (InputImage[xDirection[1] + idy * ImageSizex + idz * ImageSizex * ImageSizey]
			- InputImage[xDirection[0] + idy * ImageSizex + idz * ImageSizex * ImageSizey]) / 2;
		OutputImagey[idx + idy * ImageSizex + idz * ImageSizex * ImageSizey] = (InputImage[idx + yDirection[1] * ImageSizex + idz * ImageSizex * ImageSizey]
			- InputImage[idx + yDirection[0] * ImageSizex + idz * ImageSizex * ImageSizey]) / 2;
		OutputImagez[idx + idy * ImageSizex + idz * ImageSizex * ImageSizey] = (InputImage[idx + idy * ImageSizex + zDirection[1] * ImageSizex * ImageSizey]
			- InputImage[idx + idy * ImageSizex + zDirection[0] * ImageSizex * ImageSizey]) / 2;
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
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned idy = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned idz = blockIdx.z * blockDim.z + threadIdx.z;
	double Detxx, Detxy, Detxz, Detyx, Detyy, Detyz, Detzx, Detzy, Detzz;
	if (idx < ImageSizex && idy < ImageSizey && idz < ImageSizez)
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

		if (xDirection[0] < 0) xDirection[0] = idx;
		if (yDirection[0] < 0) yDirection[0] = idy;
		if (zDirection[0] < 0) zDirection[0] = idz;
		if (xDirection[1] > ImageSizex - 1) xDirection[1] = idx;
		if (yDirection[1] > ImageSizey - 1) yDirection[1] = idy;
		if (zDirection[1] > ImageSizez - 1) zDirection[1] = idz;

		Detxx = double(DisplaceFieldx[xDirection[1] + idy * ImageSizex + idz * ImageSizex * ImageSizey] - DisplaceFieldx[xDirection[0] + idy * ImageSizex + idz * ImageSizex * ImageSizey]) / 8 + 1;
		Detyx = double(DisplaceFieldx[idx + yDirection[1] * ImageSizex + idz * ImageSizex * ImageSizey] - DisplaceFieldx[idx + yDirection[0] * ImageSizex + idz * ImageSizex * ImageSizey]) / 8;
		Detzx = double(DisplaceFieldx[idx + idy * ImageSizex + zDirection[1] * ImageSizex * ImageSizey] - DisplaceFieldx[idx + idy * ImageSizex + zDirection[0] * ImageSizex * ImageSizey]) / 8;
		Detxy = double(DisplaceFieldy[xDirection[1] + idy * ImageSizex + idz * ImageSizex * ImageSizey] - DisplaceFieldy[xDirection[0] + idy * ImageSizex + idz * ImageSizex * ImageSizey]) / 8;
		Detyy = double(DisplaceFieldy[idx + yDirection[1] * ImageSizex + idz * ImageSizex * ImageSizey] - DisplaceFieldy[idx + yDirection[0] * ImageSizex + idz * ImageSizex * ImageSizey]) / 8 + 1;
		Detzy = double(DisplaceFieldy[idx + idy * ImageSizex + zDirection[1] * ImageSizex * ImageSizey] - DisplaceFieldy[idx + idy * ImageSizex + zDirection[0] * ImageSizex * ImageSizey]) / 8;
		Detxz = double(DisplaceFieldz[xDirection[1] + idy * ImageSizex + idz * ImageSizex * ImageSizey] - DisplaceFieldz[xDirection[0] + idy * ImageSizex + idz * ImageSizex * ImageSizey]) / 8;
		Detyz = double(DisplaceFieldz[idx + yDirection[1] * ImageSizex + idz * ImageSizex * ImageSizey] - DisplaceFieldz[idx + yDirection[0] * ImageSizex + idz * ImageSizex * ImageSizey]) / 8;
		Detzz = double(DisplaceFieldz[idx + idy * ImageSizex + zDirection[1] * ImageSizex * ImageSizey] - DisplaceFieldz[idx + idy * ImageSizex + zDirection[0] * ImageSizex * ImageSizey]) / 8 + 1;

		Jacobian[idx + idy * ImageSizex + idz * ImageSizex * ImageSizey] = Detxx * Detyy * Detzz + Detxy * Detyz * Detzx + Detxz * Detzy * Detyx
			- Detxz * Detyy * Detzx - Detxy * Detzz * Detyx - Detzy * Detxx * Detyz;
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
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned idy = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned idz = blockIdx.z * blockDim.z + threadIdx.z;
	if (idx < ImageSizex && idy < ImageSizey && idz < ImageSizez)
	{
		resultx[idx + idy * ImageSizex + idz * ImageSizex * ImageSizey] = Jacobian[idx + idy * ImageSizex + idz * ImageSizex * ImageSizey] * MetricDerivativex[idx + idy * ImageSizex + idz * ImageSizex * ImageSizey] * constData;
		resulty[idx + idy * ImageSizex + idz * ImageSizex * ImageSizey] = Jacobian[idx + idy * ImageSizex + idz * ImageSizex * ImageSizey] * MetricDerivativey[idx + idy * ImageSizex + idz * ImageSizex * ImageSizey] * constData;
		resultz[idx + idy * ImageSizex + idz * ImageSizex * ImageSizey] = Jacobian[idx + idy * ImageSizex + idz * ImageSizex * ImageSizey] * MetricDerivativez[idx + idy * ImageSizex + idz * ImageSizex * ImageSizey] * constData;
	}
}

/*******************************************************************************************
 * 							The metirc of Cross Correlation
 *******************************************************************************************/
 template<typename Real>
 __global__ void GetCCMetric(
    Real* __restrict__  ANTsCCImage,
    Real* __restrict__  FixedImage,
    Real* __restrict__  MovingImage,
    int ImageSizex,
    int ImageSizey,
    int ImageSizez,
    int VirtualImageSizex,
    int VirtualImageSizey,
    int VirtualImageSizez,
    float scale,
    int Radius)
{
    unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned idy = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned idz = blockIdx.z*blockDim.z+threadIdx.z;

    if(idx < VirtualImageSizex && idy<VirtualImageSizey &&idz<VirtualImageSizez)
    {
        float count = 0;
        float sumFixed2 = 0;
        float sumMoving2 = 0;
        float sumFixed = 0;
        float sumMoving = 0;
        float sumFixedMoving = 0;
        float fixedMean = 0;
        float movingMean = 0;
        float sFixedFixed = 0;
        float sMovingMoving = 0;
        float sFixedMoving = 0;
        float localCC = 1;
        float sFixedFixed_sMovingMoving = 0;
        float fixedImageValue;
        float movingImageValue;

        float index[3];
        int Index[3];
        int StartVirtualIndex[3] = {0,0,0};
        int EndVirtualIndex[3] = {VirtualImageSizex-1, VirtualImageSizey-1, VirtualImageSizez-1};
        int i, j, k;
        int range[3];
        range[0] = idx;
        range[1] = idy;
        range[2] = idz;

        for(k=range[2]-Radius; k<=range[2]+Radius; k++)
        {
            for(j=range[1]-Radius; j<=range[1]+Radius; j++)
            {
                for(i=range[0]-Radius; i<=range[0]+Radius; i++)
                {
                    index[0] = i;
                    index[1] = j;
                    index[2] = k;

                    if(index[0]>=StartVirtualIndex[0]&&index[0]<=EndVirtualIndex[0]&&
                        index[1]>=StartVirtualIndex[1]&&index[1]<=EndVirtualIndex[1]&&
                        index[2]>=StartVirtualIndex[2]&&index[2]<=EndVirtualIndex[2])
                    {
                        index[0] = float(index[0]/scale);
                        index[1] = float(index[1]/scale);
                        index[2] = float(index[2]/scale);

                        if(index[0]>=0&&index[0]<=ImageSizex-1&&
                                index[1]>=0&&index[1]<=ImageSizey-1&&
                                index[2]>=0&&index[2]<=ImageSizez-1)
                        {
                            Index[0] = int(index[0]);
                            Index[1] = int(index[1]);
                            Index[2] = int(index[2]);
                            fixedImageValue = FixedImage[Index[0]+Index[1]*ImageSizex+ Index[2]*ImageSizex*ImageSizey];
                            movingImageValue = MovingImage[Index[0]+Index[1]*ImageSizex+ Index[2]*ImageSizex*ImageSizey];
                            sumFixed2 += pow(fixedImageValue,2);
                            sumMoving2 += pow(movingImageValue,2);
                            sumFixed += fixedImageValue;
                            sumMoving += movingImageValue;
                            sumFixedMoving += fixedImageValue*movingImageValue;
                            count++;
                        }
                    }
                }
            }
        }

        fixedMean  = sumFixed  / count;
        movingMean = sumMoving / count;
        sFixedFixed   = sumFixed2 - fixedMean * sumFixed - fixedMean * sumFixed + count * fixedMean * fixedMean;
        sMovingMoving = sumMoving2 - movingMean * sumMoving - movingMean * sumMoving + count * movingMean * movingMean;
        sFixedMoving  = sumFixedMoving - movingMean * sumFixed - fixedMean * sumMoving + count * movingMean * fixedMean;
        sFixedFixed_sMovingMoving = sFixedFixed * sMovingMoving;

        if ( fabs(sFixedFixed_sMovingMoving) > 2.22045e-16 )
            localCC = sFixedMoving * sFixedMoving / (sFixedFixed_sMovingMoving);

        ANTsCCImage[idx+idy*VirtualImageSizex + idz*VirtualImageSizex*VirtualImageSizey] = localCC;
    }
}

at::Tensor ccMetric_cuda(
    torch::Tensor MI,
    torch::Tensor FI,
    int Vsizex,
    int Vsizey,
    int Vsizez,
    float scale,
    int Radius)
{
    const int uint = 8;
	const dim3 grid((Vsizez+uint-1)/uint,(Vsizey+uint-1)/uint,(Vsizex+uint-1)/uint);
    const dim3 block(uint,uint,uint);
    
    at::Tensor CCImage = at::zeros({Vsizex, Vsizey, Vsizez}, MI.type());

	AT_DISPATCH_FLOATING_TYPES(MI.type(), "cc metric", ([&] {
        GetCCMetric<<<grid, block>>>(CCImage.data<scalar_t>(),
                                                    FI.data<scalar_t>(),  
                                                    MI.data<scalar_t>(),
                                                    MI.size(2),
                                                    MI.size(1),
                                                    MI.size(0),
                                                    Vsizez,
                                                    Vsizey,
                                                    Vsizex,
                                                    scale,
                                                    Radius);}));

    return -CCImage.sum();
}


/*****************************************************************************************
*                             the derivative of cc metric                                *
*****************************************************************************************/
template<typename Real>
__global__ void GetDerivativeCCMetric(    
    Real* __restrict__ Derivativex,
    Real* __restrict__ Derivativey,
    Real* __restrict__ Derivativez,
    Real* __restrict__ ForwardTransformedImage,
    Real* __restrict__ FixedTransformedImage,
    Real* __restrict__ TransformedGradientx,
    Real* __restrict__ TransformedGradienty,
    Real* __restrict__ TransformedGradientz,
    int ImageSizex,
    int ImageSizey,
    int ImageSizez,
    int VirtualImageSizex,
    int VirtualImageSizey,
    int VirtualImageSizez,
    float scale,
    int Radius)
{
    unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned idy = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned idz = blockIdx.z*blockDim.z+threadIdx.z;

    if(idx < VirtualImageSizex && idy<VirtualImageSizey &&idz<VirtualImageSizez)
    {
        float count = 0;
        float sumFixed2 = 0;
        float sumMoving2 = 0;
        float sumFixed = 0;
        float sumMoving = 0;
        float sumFixedMoving = 0;
        float fixedMean = 0;
        float movingMean = 0;
        float sFixedFixed = 0;
        float sMovingMoving = 0;
        float sFixedMoving = 0;
        float fixedA = 0;
        float movingA = 0;
        float sFixedFixed_sMovingMoving = 0;
        float fixedImageValue;
        float movingImageValue;
        float derivWRTImage[3];
        float index[3];
        int Index[3];
        int EndVirtualIndex[3];
        int StartVirtualIndex[3] = {0,0,0};
        EndVirtualIndex[0] = VirtualImageSizex - 1;
        EndVirtualIndex[1] = VirtualImageSizey - 1;
        EndVirtualIndex[2] = VirtualImageSizez - 1;
        int i, j, k;
        float movingImageGradient[3];
        int range[3];

        range[0] = idx;
        range[1] = idy;
        range[2] = idz;

        for(k=range[2]-Radius; k<=range[2]+Radius; k++)
        {
            for(j=range[1]-Radius; j<=range[1]+Radius; j++)
            {
                for(i=range[0]-Radius; i<=range[0]+Radius; i++)
                {
                    index[0] = i;
                    index[1] = j;
                    index[2] = k;

                    if(index[0]>=StartVirtualIndex[0]&&index[0]<=EndVirtualIndex[0]&&
                    index[1]>=StartVirtualIndex[1]&&index[1]<=EndVirtualIndex[1]&&
                    index[2]>=StartVirtualIndex[2]&&index[2]<=EndVirtualIndex[2])
                    {

                        index[0] = float(index[0]/scale);
                        index[1] = float(index[1]/scale);
                        index[2] = float(index[2]/scale);

                        if(index[0]>=0&&index[0]<=ImageSizex-1&&
                        index[1]>=0&&index[1]<=ImageSizey-1&&
                        index[2]>=0&&index[2]<=ImageSizez-1)
                        {
                            Index[0] = int(index[0]);
                            Index[1] = int(index[1]);
                            Index[2] = int(index[2]);

                            fixedImageValue = FixedTransformedImage[Index[0]+Index[1]*ImageSizex+ Index[2]*ImageSizex*ImageSizey];
                            movingImageValue = ForwardTransformedImage[Index[0]+Index[1]*ImageSizex+ Index[2]*ImageSizex*ImageSizey];

                            sumFixed2 += pow(fixedImageValue,2);
                            sumMoving2 += pow(movingImageValue,2);
                            sumFixed += fixedImageValue;
                            sumMoving += movingImageValue;
                            sumFixedMoving += fixedImageValue*movingImageValue;
                            count++;
                        }
                    }
                }
            }
        }

        fixedMean  = sumFixed  / count;
        movingMean = sumMoving / count;
        sFixedFixed   = sumFixed2 - fixedMean * sumFixed - fixedMean * sumFixed + count * fixedMean * fixedMean;
        sMovingMoving = sumMoving2 - movingMean * sumMoving - movingMean * sumMoving + count * movingMean * movingMean;
        sFixedMoving  = sumFixedMoving - movingMean * sumFixed - fixedMean * sumMoving + count * movingMean * fixedMean;
        sFixedFixed_sMovingMoving = sFixedFixed * sMovingMoving;

        index[0] = float(idx/scale);
        index[1] = float(idy/scale);
        index[2] = float(idz/scale);


        if(index[0]>=0&&index[0]<=ImageSizex-1&&
        index[1]>=0&&index[1]<=ImageSizey-1&&
        index[2]>=0&&index[2]<=ImageSizez-1)
        {
        Index[0] = int(index[0]);
        Index[1] = int(index[1]);
        Index[2] = int(index[2]);
        fixedImageValue = FixedTransformedImage[Index[0]+Index[1]*ImageSizex+ Index[2]*ImageSizex*ImageSizey];
        movingImageValue = ForwardTransformedImage[Index[0]+Index[1]*ImageSizex+ Index[2]*ImageSizex*ImageSizey];
        movingImageGradient[0] = TransformedGradientx[Index[0]+Index[1]*ImageSizex+ Index[2]*ImageSizex*ImageSizey];
        movingImageGradient[1] = TransformedGradienty[Index[0]+Index[1]*ImageSizex+ Index[2]*ImageSizex*ImageSizey];
        movingImageGradient[2] = TransformedGradientz[Index[0]+Index[1]*ImageSizex+ Index[2]*ImageSizex*ImageSizey];


        fixedA        = fixedImageValue  - fixedMean;
        movingA       = movingImageValue - movingMean;

        Derivativex[idx+idy*VirtualImageSizex+idz*VirtualImageSizex*VirtualImageSizey]=fixedA;
        Derivativey[idx+idy*VirtualImageSizex+idz*VirtualImageSizex*VirtualImageSizey]=fixedA;
        Derivativez[idx+idy*VirtualImageSizex+idz*VirtualImageSizex*VirtualImageSizey]=fixedA;

        if ( ! (sFixedFixed >  2.22045e-16 && sMovingMoving > 2.22045e-16 ) )
        {
            Derivativex[idx+idy*VirtualImageSizex+idz*VirtualImageSizex*VirtualImageSizey]=0;
            Derivativey[idx+idy*VirtualImageSizex+idz*VirtualImageSizex*VirtualImageSizey]=0;
            Derivativez[idx+idy*VirtualImageSizex+idz*VirtualImageSizex*VirtualImageSizey]=0;
        }
        else
        {
            derivWRTImage[0] = 2.0 * sFixedMoving / (sFixedFixed_sMovingMoving) * (fixedA - sFixedMoving / sMovingMoving * movingA) * movingImageGradient[0];
            derivWRTImage[1] = 2.0 * sFixedMoving / (sFixedFixed_sMovingMoving) * (fixedA - sFixedMoving / sMovingMoving * movingA) * movingImageGradient[1];
            derivWRTImage[2] = 2.0 * sFixedMoving / (sFixedFixed_sMovingMoving) * (fixedA - sFixedMoving / sMovingMoving * movingA) * movingImageGradient[2];
            Derivativex[idx+idy*VirtualImageSizex+idz*VirtualImageSizex*VirtualImageSizey]=derivWRTImage[0];
            Derivativey[idx+idy*VirtualImageSizex+idz*VirtualImageSizex*VirtualImageSizey]=derivWRTImage[1];
            Derivativez[idx+idy*VirtualImageSizex+idz*VirtualImageSizex*VirtualImageSizey]=derivWRTImage[2];
        }

        }
        else
        {
            Derivativex[idx+idy*VirtualImageSizex+idz*VirtualImageSizex*VirtualImageSizey]=0;
            Derivativey[idx+idy*VirtualImageSizex+idz*VirtualImageSizex*VirtualImageSizey]=0;
            Derivativez[idx+idy*VirtualImageSizex+idz*VirtualImageSizex*VirtualImageSizey]=0;
        }

    }
}

template<typename Real>
__global__ void CCTransformImage(
	Real* __restrict__ ForwardTransformedImage,
	Real* __restrict__ FixedTransformedImage,
	Real* __restrict__ TranGradientx,
	Real* __restrict__ TranGradienty,
	Real* __restrict__ TranGradientz,
	Real* __restrict__ ForwardImage,
	Real* __restrict__ FixedImage,
	Real* __restrict__ gradientImagex,
	Real* __restrict__ gradientImagey,
	Real* __restrict__ gradientImagez,
	Real* __restrict__ DisplaceFieldx,
	Real* __restrict__ DisplaceFieldy,
	Real* __restrict__ DisplaceFieldz,
	int ImageSizex,
	int ImageSizey,
	int ImageSizez,
	int xsize,
	int ysize,
	int zsize,
	float scale)
{
#define MovingImageDimension 3
	unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned idy = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned idz = blockIdx.z*blockDim.z + threadIdx.z;
	float xindex, yindex, zindex;
	float transformed[3];
	float gradient[3];
	float ForwardTransformedImageValue;
	float FixedTransformedImageValue;
	if (idx < ImageSizex && idy<ImageSizey &&idz<ImageSizez)
	{
		xindex = float(idx);
		yindex = float(idy);
		zindex = float(idz);
		transformed[0] = xindex + GetDisplacement(DisplaceFieldx, xsize, ysize, zsize, xindex, yindex, zindex, ImageSizex, ImageSizey, ImageSizez, scale);
		transformed[1] = yindex + GetDisplacement(DisplaceFieldy, xsize, ysize, zsize, xindex, yindex, zindex, ImageSizex, ImageSizey, ImageSizez, scale);
		transformed[2] = zindex + GetDisplacement(DisplaceFieldz, xsize, ysize, zsize, xindex, yindex, zindex, ImageSizex, ImageSizey, ImageSizez, scale);

		ForwardTransformedImageValue = ResamplePoint(ForwardImage, ImageSizex, ImageSizey, ImageSizez, transformed[0], transformed[1], transformed[2]);
		FixedTransformedImageValue = ResamplePoint(FixedImage, ImageSizex, ImageSizey, ImageSizez, transformed[0], transformed[1], transformed[2]);
		gradient[0] = ResamplePoint(gradientImagex, ImageSizex, ImageSizey, ImageSizez, transformed[0], transformed[1], transformed[2]);
		gradient[1] = ResamplePoint(gradientImagey, ImageSizex, ImageSizey, ImageSizez, transformed[0], transformed[1], transformed[2]);
		gradient[2] = ResamplePoint(gradientImagez, ImageSizex, ImageSizey, ImageSizez, transformed[0], transformed[1], transformed[2]);

		ForwardTransformedImage[idx + idy*ImageSizex + idz*ImageSizex*ImageSizey] = ForwardTransformedImageValue;
		FixedTransformedImage[idx + idy*ImageSizex + idz*ImageSizex*ImageSizey] = FixedTransformedImageValue;
		TranGradientx[idx + idy*ImageSizex + idz*ImageSizex*ImageSizey] = gradient[0];
		TranGradienty[idx + idy*ImageSizex + idz*ImageSizex*ImageSizey] = gradient[1];
		TranGradientz[idx + idy*ImageSizex + idz*ImageSizex*ImageSizey] = gradient[2];
	}

}


std::vector<at::Tensor>   ccMetricDerivative_cuda(
    torch::Tensor MI,
    torch::Tensor FI,
    torch::Tensor DFx,
    torch::Tensor DFy,
    torch::Tensor DFz,
    float scale,
    float constData,
    int Radius)
{
	const int uint = 4;
	const dim3 grid((MI.size(2) + uint - 1) / uint, (MI.size(1) + uint - 1) / uint, (MI.size(0) + uint - 1) / uint);
	const dim3 block(uint, uint, uint);
    at::Tensor gMI_x = at::zeros_like(MI);
    at::Tensor gMI_y = at::zeros_like(MI);
    at::Tensor gMI_z = at::zeros_like(MI);

    
	AT_DISPATCH_FLOATING_TYPES(MI.type(), "gradient_of_moving_image", ([&] {
        Gradient<scalar_t><<<grid, block>>>(
            gMI_x.data<scalar_t>(),
            gMI_y.data<scalar_t>(),
            gMI_z.data<scalar_t>(),
            MI.data<scalar_t>(),
            MI.size(2),
            MI.size(1),
            MI.size(0));}));
            
    
    at::Tensor transf_MI = at::zeros_like(MI);
    at::Tensor transf_FI = at::zeros_like(MI);
    at::Tensor sampled_gx = at::zeros_like(MI);
    at::Tensor sampled_gy = at::zeros_like(MI);
    at::Tensor sampled_gz = at::zeros_like(MI);

	AT_DISPATCH_FLOATING_TYPES(MI.type(), "derivative_of_cc_part1", ([&] {
    CCTransformImage<scalar_t><< <grid, block >> >(
                                            transf_MI.data<scalar_t>(),
                                            transf_FI.data<scalar_t>(),
                                            sampled_gx.data<scalar_t>(),
                                            sampled_gy.data<scalar_t>(),
                                            sampled_gz.data<scalar_t>(),
                                            MI.data<scalar_t>(),
                                            FI.data<scalar_t>(),
                                            gMI_x.data<scalar_t>(),
                                            gMI_y.data<scalar_t>(),
                                            gMI_z.data<scalar_t>(),
                                            DFx.data<scalar_t>(),
                                            DFy.data<scalar_t>(),
                                            DFz.data<scalar_t>(),
                                            MI.size(2),
                                            MI.size(1),
                                            MI.size(0),
                                            DFx.size(2),
                                            DFx.size(1),
                                            DFx.size(0),
                                            scale);}));

    const dim3 gridvirtual((DFx.size(2) + uint - 1) / uint, (DFx.size(1) + uint - 1) / uint, (DFx.size(0) + uint - 1) / uint);
    const dim3 blockvirtual(uint, uint, uint);

    at::Tensor cc_derivex1 = at::zeros_like(DFx);
    at::Tensor cc_derivey1 = at::zeros_like(DFx);
    at::Tensor cc_derivez1 = at::zeros_like(DFx);

    AT_DISPATCH_FLOATING_TYPES(MI.type(), "derivative_of_cc_part2", ([&] {
        GetDerivativeCCMetric<scalar_t><< <gridvirtual, blockvirtual >> >(
                                cc_derivex1.data<scalar_t>(),
                                cc_derivey1.data<scalar_t>(),
                                cc_derivez1.data<scalar_t>(),
                                transf_MI.data<scalar_t>(),
                                transf_FI.data<scalar_t>(),
                                sampled_gx.data<scalar_t>(),
                                sampled_gy.data<scalar_t>(),
                                sampled_gz.data<scalar_t>(),
                                MI.size(2),
                                MI.size(1),
                                MI.size(0),            
                                DFx.size(2),
                                DFx.size(1),
                                DFx.size(0),
                                scale,
                                Radius);}));

    

    at::Tensor Jacobain = at::zeros_like(DFx);

    AT_DISPATCH_FLOATING_TYPES(MI.type(), "derivative_of_cc_Jacobain", ([&] {
        ComputJacobian <scalar_t><< <gridvirtual, blockvirtual >> >(
            Jacobain.data<scalar_t>(),
            DFx.data<scalar_t>(),
            DFy.data<scalar_t>(),
            DFz.data<scalar_t>(),
            DFx.size(2),
            DFx.size(1),
            DFx.size(0));}));
    
    at::Tensor cc_derivex = at::zeros_like(DFx);
    at::Tensor cc_derivey = at::zeros_like(DFx);
    at::Tensor cc_derivez = at::zeros_like(DFx);

    AT_DISPATCH_FLOATING_TYPES(MI.type(), "derivative_of_cc_Jacobain", ([&] {
        MultiplyJacobianAndConst <scalar_t><< <gridvirtual, blockvirtual >> >(
            cc_derivex.data<scalar_t>(),
            cc_derivey.data<scalar_t>(),
            cc_derivez.data<scalar_t>(),
            Jacobain.data<scalar_t>(),
            cc_derivex1.data<scalar_t>(),
            cc_derivey1.data<scalar_t>(),
            cc_derivez1.data<scalar_t>(),
            DFx.size(2),
            DFx.size(1),
            DFx.size(0),
            constData);}));
    
    /*
    cc_derivex = at::unsqueeze(cc_derivex, 0);
    cc_derivey = at::unsqueeze(cc_derivey, 0);
    cc_derivez = at::unsqueeze(cc_derivez, 0); 
    return at::cat({cc_derivex, cc_derivey, cc_derivez}, 0);
    */
    return {cc_derivex, cc_derivey, cc_derivez};
}

