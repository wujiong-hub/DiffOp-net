import SimpleITK as sitk
import nibabel as ni
import numpy as np

def imgRead(path):
    """
    Alias for sitk.ReadImage
    """
    inImg = sitk.ReadImage(path)
    inDimension = inImg.GetDimension()
    inImg.SetDirection(sitk.AffineTransform(inDimension).GetMatrix())
    inImg.SetOrigin([0]*inDimension)

    return inImg

def imgApplyField(img, field, useNearest=False, size=[], spacing=[], defaultValue=0):
    """
    img \circ field
    """
    field = sitk.Cast(field, sitk.sitkVectorFloat64)
    # Set interpolator
    interpolator = [sitk.sitkLinear, sitk.sitkNearestNeighbor][useNearest]

    # Set transform field
    transform = sitk.DisplacementFieldTransform(img.GetDimension())
    transform.SetInterpolator(sitk.sitkLinear)
    transform.SetDisplacementField(field)

    # Set size
    if size == []:
        size = img.GetSize()
    else:
        if len(size) != img.GetDimension(): raise Exception("size must have length {0}.".format(img.GetDimension()))

    # Set Spacing
    if spacing == []:
        spacing = img.GetSpacing()
    else:
        if len(spacing) != img.GetDimension(): raise Exception(
            "spacing must have length {0}.".format(img.GetDimension()))

    # Apply displacement transform
    return sitk.Resample(img, size, transform, interpolator, [0] * img.GetDimension(), spacing, img.GetDirection(),
                         defaultValue)

def getField(fieldx, fieldy, fieldz, targetImageDir, scale):
    """
    :param fieldx: the x component of field
    :param fieldy: the y component of field
    :param fieldz: the z component of field
    :param targetImageDir: the target image used for get some useful information
    :return: composed field
    """
    targetImage = sitk.ReadImage(targetImageDir)
    phixImg = sitk.GetImageFromArray(fieldx)
    phiyImg = sitk.GetImageFromArray(fieldy)
    phizImg = sitk.GetImageFromArray(fieldz)
    phixImg.SetOrigin(targetImage.GetOrigin())
    phiyImg.SetOrigin(targetImage.GetOrigin())
    phizImg.SetOrigin(targetImage.GetOrigin())

    orgSpacing = targetImage.GetSpacing()
    Spacing = (orgSpacing[0]/scale, orgSpacing[1]/scale, orgSpacing[2]/scale)
    phixImg.SetSpacing(Spacing)
    phiyImg.SetSpacing(Spacing)
    phizImg.SetSpacing(Spacing)
    phixImg.SetDirection(targetImage.GetDirection())
    phiyImg.SetDirection(targetImage.GetDirection())
    phizImg.SetDirection(targetImage.GetDirection())
    compose = sitk.ComposeImageFilter()
    field = compose.Execute(phixImg, phiyImg, phizImg)
    field.SetDirection(targetImage.GetDirection())
    return field

dimension = 3
vectorComponentType = sitk.sitkFloat32
vectorType = sitk.sitkVectorFloat32
affine = sitk.AffineTransform(dimension)
identityAffine = list(affine.GetParameters())
identityDirection = list(affine.GetMatrix())
zeroOrigin = [0]*dimension
zeroIndex = [0]*dimension


def fieldApplyField(inField, field, imgDir):
    """ outField = inField \circ field """
    inDimension = inField.GetDimension()
    inField.SetDirection(sitk.AffineTransform(inDimension).GetMatrix())
    inField.SetOrigin([0]*inDimension)

    #inDimension = field.GetDimension()
    field.SetDirection(sitk.AffineTransform(inDimension).GetMatrix())
    field.SetOrigin([0]*inDimension)

    inField = sitk.Cast(inField, sitk.sitkVectorFloat64)
    field = sitk.Cast(field, sitk.sitkVectorFloat64)

    size = list(inField.GetSize())
    spacing = list(inField.GetSpacing())

    # Create transform for input field
    inTransform = sitk.DisplacementFieldTransform(dimension)
    inTransform.SetDisplacementField(inField)
    inTransform.SetInterpolator(sitk.sitkLinear)

    # Create transform for field
    transform = sitk.DisplacementFieldTransform(dimension)
    transform.SetDisplacementField(field)
    transform.SetInterpolator(sitk.sitkLinear)

    # Combine thransforms
    outTransform = sitk.Transform()
    outTransform.AddTransform(transform)
    outTransform.AddTransform(inTransform)


    # Get output displacement field
    combineField = sitk.TransformToDisplacementFieldFilter().Execute(outTransform, vectorType, size, zeroOrigin, spacing, identityDirection)
    combineField.SetDirection(sitk.ReadImage(imgDir).GetDirection())
    return combineField

def getTransformedImage(templateImageDir, field, targetImageDir, useNearest=False):
    """
    :param templateImageDir
    :param fieldDir
    :param targetImageDir
    :return: transformed tempalte image
    """
    templateImage = imgRead(templateImageDir)
    field.SetDirection(sitk.AffineTransform(field.GetDimension()).GetMatrix())
    field.SetOrigin([0]*field.GetDimension())
    targetImage = sitk.ReadImage(targetImageDir)
    transformedImage = imgApplyField(templateImage, field, useNearest=useNearest, size=targetImage.GetSize(), spacing=targetImage.GetSpacing())
    transformedImage.SetOrigin(targetImage.GetOrigin())
    transformedImage.SetSpacing(targetImage.GetSpacing())
    transformedImage.SetDirection(targetImage.GetDirection())

    return transformedImage

