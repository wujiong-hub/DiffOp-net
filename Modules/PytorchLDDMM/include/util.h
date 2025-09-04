//#include<torch/extension.h>
#include<torch/types.h>

template<typename Real>
inline __device__ Real 
GetDisplacement(
	const Real *DisplaceField,
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
		if (xBas0<0) { xBas0 = 0; if (xBas1<0) { xBas1 = 0; } }
		if (yBas0<0) { yBas0 = 0; if (yBas1<0) { yBas1 = 0; } }
		if (zBas0<0) { zBas0 = 0; if (zBas1<0) { zBas1 = 0; } }
		if (xBas1>(Dsizex - 1)) { xBas1 = Dsizex - 1; if (xBas0>(Dsizex - 1)) { xBas0 = Dsizex - 1; } }
		if (yBas1>(Dsizey - 1)) { yBas1 = Dsizey - 1; if (yBas0>(Dsizey - 1)) { yBas0 = Dsizey - 1; } }
		if (zBas1>(Dsizez - 1)) { zBas1 = Dsizez - 1; if (zBas0>(Dsizez - 1)) { zBas0 = Dsizez - 1; } }

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
	const Real *InputImage,
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
	if (xBas0<0) { xBas0 = 0; if (xBas1<0) { xBas1 = 0; } }
	if (yBas0<0) { yBas0 = 0; if (yBas1<0) { yBas1 = 0; } }
	if (zBas0<0) { zBas0 = 0; if (zBas1<0) { zBas1 = 0; } }
	if (xBas1>(ImageSizex - 1)) { xBas1 = ImageSizex - 1; if (xBas0>(ImageSizex - 1)) { xBas0 = ImageSizex - 1; } }
	if (yBas1>(ImageSizey - 1)) { yBas1 = ImageSizey - 1; if (yBas0>(ImageSizey - 1)) { yBas0 = ImageSizey - 1; } }
	if (zBas1>(ImageSizez - 1)) { zBas1 = ImageSizez - 1; if (zBas0>(ImageSizez - 1)) { zBas0 = ImageSizez - 1; } }

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

