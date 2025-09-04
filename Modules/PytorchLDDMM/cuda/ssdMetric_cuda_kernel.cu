//#include <torch/extension.h>
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
 * 							The metirc of Sum of Squared Difference
 *******************************************************************************************/

 template<typename Real>
 __global__ void SumOfSquaredMetric(
	Real* __restrict__  SumOfSquaredImage,
	Real* __restrict__  FixedImage,
	Real* __restrict__  MovingImage,
	int ImageSizex,
	int ImageSizey,
	int ImageSizez,
	int Vsizex,
	int Vsizey,
	int Vsizez,
	float scale)
{
	unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned idy = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned idz = blockIdx.z*blockDim.z + threadIdx.z;
	int index[3];
	double difference;
	if (idx < Vsizex && idy<Vsizey &&idz<Vsizez)
	{
		index[0] = int(float(idx / scale));
		index[1] = int(float(idy / scale));
		index[2] = int(float(idz / scale));
		if (index[0] >= 0 && index[0] <= ImageSizex - 1 &&
			index[1] >= 0 && index[1] <= ImageSizey - 1 &&
			index[2] >= 0 && index[2] <= ImageSizez - 1)
		{
			difference = FixedImage[index[0] + index[1] * ImageSizex + index[2] * ImageSizex*ImageSizey] - MovingImage[index[0] + index[1] * ImageSizex + index[2] * ImageSizex*ImageSizey];
			SumOfSquaredImage[idx + idy*Vsizex + idz*Vsizex*Vsizey] = difference * difference;
		}
		else
		{
			SumOfSquaredImage[idx + idy*Vsizex + idz*Vsizex*Vsizey] = 0;
		}
	}

}

at::Tensor ssdMetric_cuda(
	torch::Tensor  MI,
    torch::Tensor  FI,
    int Vsizex,
    int Vsizey,
    int Vsizez,
    float scale)
{
	const int uint = 8;
	const dim3 grid((Vsizez+uint-1)/uint,(Vsizey+uint-1)/uint,(Vsizex+uint-1)/uint);
	const dim3 block(uint,uint,uint);

    at::Tensor DifferenceImage = at::zeros_like(MI);

    AT_DISPATCH_FLOATING_TYPES(MI.type(), "ssd metric", ([&] {
        SumOfSquaredMetric<scalar_t><<<grid, block>>>(DifferenceImage.data<scalar_t>(),
                                                      FI.data<scalar_t>(),  
                                                      MI.data<scalar_t>(),
                                                      MI.size(2),
                                                      MI.size(1),
                                                      MI.size(0),
                                                      Vsizez,
                                                      Vsizey,
                                                      Vsizex,
                                                      scale);}));
	return DifferenceImage.sum();
}



/*******************************************************************************************
 * 							The Derivative of Sum of Squared Difference metric
 *******************************************************************************************/
 template<typename Real>
 __global__ void SSDTransformImage(
	Real* __restrict__  ForwardTransformedImage,
	Real* __restrict__  FixedTransformedImage,
	Real* __restrict__  TranGradientx,
	Real* __restrict__  TranGradienty,
	Real* __restrict__  TranGradientz,
	const Real* __restrict__  ForwardImage,
	const Real* __restrict__  FixedImage,
	const Real* __restrict__  gradientImagex,
	const Real* __restrict__  gradientImagey,
	const Real* __restrict__  gradientImagez,
	const Real* __restrict__  DisplaceFieldx,
	const Real* __restrict__  DisplaceFieldy,
	const Real* __restrict__  DisplaceFieldz,
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
	double transformed[3];
	double gradient[3];
	double ForwardTransformedImageValue;
	double FixedTransformedImageValue;
	if (idx < xsize && idy<ysize &&idz<zsize)
	{
		xindex = float(idx) / scale;
		yindex = float(idy) / scale;
		zindex = float(idz) / scale;
		transformed[0] = xindex + GetDisplacement(DisplaceFieldx, xsize, ysize, zsize, xindex, yindex, zindex, ImageSizex, ImageSizey, ImageSizez, scale);
		transformed[1] = yindex + GetDisplacement(DisplaceFieldy, xsize, ysize, zsize, xindex, yindex, zindex, ImageSizex, ImageSizey, ImageSizez, scale);
		transformed[2] = zindex + GetDisplacement(DisplaceFieldz, xsize, ysize, zsize, xindex, yindex, zindex, ImageSizex, ImageSizey, ImageSizez, scale);

		ForwardTransformedImageValue = ResamplePoint(ForwardImage, ImageSizex, ImageSizey, ImageSizez, transformed[0], transformed[1], transformed[2]);
		FixedTransformedImageValue = ResamplePoint(FixedImage, ImageSizex, ImageSizey, ImageSizez, transformed[0], transformed[1], transformed[2]);
		gradient[0] = ResamplePoint(gradientImagex, ImageSizex, ImageSizey, ImageSizez, transformed[0], transformed[1], transformed[2]);
		gradient[1] = ResamplePoint(gradientImagey, ImageSizex, ImageSizey, ImageSizez, transformed[0], transformed[1], transformed[2]);
		gradient[2] = ResamplePoint(gradientImagez, ImageSizex, ImageSizey, ImageSizez, transformed[0], transformed[1], transformed[2]);

		ForwardTransformedImage[idx + idy*xsize + idz*xsize*ysize] = ForwardTransformedImageValue;
		FixedTransformedImage[idx + idy*xsize + idz*xsize*ysize] = FixedTransformedImageValue;
		TranGradientx[idx + idy*xsize + idz*xsize*ysize] = gradient[0];
		TranGradienty[idx + idy*xsize + idz*xsize*ysize] = gradient[1];
		TranGradientz[idx + idy*xsize + idz*xsize*ysize] = gradient[2];
	}

}

template<typename Real>
__global__ void GetDerivativeMeansquareMetric(
	Real* __restrict__  Derivativex,
	Real* __restrict__  Derivativey,
	Real* __restrict__  Derivativez,
	Real* __restrict__  ForwardTransformedImage,
	Real* __restrict__  FixedTransformedImage,
	Real* __restrict__  GradientTransformedx,
	Real* __restrict__  GradientTransformedy,
    Real* __restrict__  GradientTransformedz,
	int virtualImageSizex,
	int virtualImageSizey,
	int virtualImageSizez)
{
	unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned idy = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned idz = blockIdx.z*blockDim.z + threadIdx.z;
	if (idx < virtualImageSizex && idy<virtualImageSizey &&idz<virtualImageSizez)
	{
		Derivativex[idx + idy*virtualImageSizex + idz*virtualImageSizex*virtualImageSizey] = 2 * double(FixedTransformedImage[idx + idy*virtualImageSizex + idz*virtualImageSizex*virtualImageSizey] -
			ForwardTransformedImage[idx + idy*virtualImageSizex + idz*virtualImageSizex*virtualImageSizey])
			*GradientTransformedx[idx + idy*virtualImageSizex + idz*virtualImageSizex*virtualImageSizey];

		Derivativey[idx + idy*virtualImageSizex + idz*virtualImageSizex*virtualImageSizey] = 2 * double(FixedTransformedImage[idx + idy*virtualImageSizex + idz*virtualImageSizex*virtualImageSizey] -
			ForwardTransformedImage[idx + idy*virtualImageSizex + idz*virtualImageSizex*virtualImageSizey])
			*GradientTransformedy[idx + idy*virtualImageSizex + idz*virtualImageSizex*virtualImageSizey];

		Derivativez[idx + idy*virtualImageSizex + idz*virtualImageSizex*virtualImageSizey] = 2 * double(FixedTransformedImage[idx + idy*virtualImageSizex + idz*virtualImageSizex*virtualImageSizey] -
			ForwardTransformedImage[idx + idy*virtualImageSizex + idz*virtualImageSizex*virtualImageSizey])
			*GradientTransformedz[idx + idy*virtualImageSizex + idz*virtualImageSizex*virtualImageSizey];
	}
}


std::vector<at::Tensor> ssdMetricDerivative_cuda(
	torch::Tensor MI,
	torch::Tensor FI,
	torch::Tensor DFx,
	torch::Tensor DFy,
	torch::Tensor DFz,
	float scale,
	float constData)
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
    //std::cout<<"============================="<<std::endl;
    //std::cout<<gMI_x.size(0)<<" "<<gMI_x.size(1)<<" "<<gMI_x.size(2)<<std::endl;
	const dim3 gridvirtual((DFx.size(2) + uint - 1) / uint, (DFx.size(1) + uint - 1) / uint, (DFx.size(0) + uint - 1) / uint);
	const dim3 blockvirtual(uint, uint, uint);
    
    at::Tensor transf_MI = at::zeros_like(DFx);
    at::Tensor transf_FI = at::zeros_like(DFx);
    at::Tensor sampled_gx = at::zeros_like(DFx);
    at::Tensor sampled_gy = at::zeros_like(DFx);
    at::Tensor sampled_gz = at::zeros_like(DFx);

    AT_DISPATCH_FLOATING_TYPES(MI.type(), "derivative_of_ssd_part1", ([&] {
	SSDTransformImage<scalar_t><< <gridvirtual, blockvirtual >> >(
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
	
	//std::cout<<"============================="<<std::endl;
	//std::cout<<transf_MI.size(0)<<" "<<transf_MI.size(1)<<" "<<transf_MI.size(2)<<std::endl;

    at::Tensor ssd_derivex1 = at::zeros_like(DFx);
    at::Tensor ssd_derivey1 = at::zeros_like(DFx);
    at::Tensor ssd_derivez1 = at::zeros_like(DFx);

    AT_DISPATCH_FLOATING_TYPES(MI.type(), "derivative_of_ssd_part2", ([&] {
        GetDerivativeMeansquareMetric<scalar_t><< <gridvirtual, blockvirtual >> >(
            ssd_derivex1.data<scalar_t>(),
            ssd_derivey1.data<scalar_t>(),
            ssd_derivez1.data<scalar_t>(),
            transf_MI.data<scalar_t>(),
            transf_FI.data<scalar_t>(),
            sampled_gx.data<scalar_t>(),
            sampled_gy.data<scalar_t>(),
            sampled_gz.data<scalar_t>(),
            DFx.size(2),
            DFx.size(1),
            DFx.size(0));}));

    at::Tensor Jacobain = at::zeros_like(DFx);

    AT_DISPATCH_FLOATING_TYPES(MI.type(), "derivative_of_ssd_Jacobain", ([&] {
        ComputJacobian <scalar_t><< <gridvirtual, blockvirtual >> >(
            Jacobain.data<scalar_t>(),
            DFx.data<scalar_t>(),
            DFy.data<scalar_t>(),
            DFz.data<scalar_t>(),
            DFx.size(2),
            DFx.size(1),
            DFx.size(0));}));
    
    at::Tensor ssd_derivex = at::zeros_like(DFx);
    at::Tensor ssd_derivey = at::zeros_like(DFx);
    at::Tensor ssd_derivez = at::zeros_like(DFx);

    AT_DISPATCH_FLOATING_TYPES(MI.type(), "derivative_of_ssd_Jacobain", ([&] {
	    MultiplyJacobianAndConst <scalar_t><< <gridvirtual, blockvirtual >> >(
            ssd_derivex.data<scalar_t>(),
            ssd_derivey.data<scalar_t>(),
            ssd_derivez.data<scalar_t>(),
            Jacobain.data<scalar_t>(),
            ssd_derivex1.data<scalar_t>(),
            ssd_derivey1.data<scalar_t>(),
            ssd_derivez1.data<scalar_t>(),
            DFx.size(2),
            DFx.size(1),
            DFx.size(0),
            constData);}));
    
    return {ssd_derivex, ssd_derivey, ssd_derivez};
}
