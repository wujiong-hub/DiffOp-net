#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

std::vector<at::Tensor> KernelInitialize_cuda(
    at::Tensor VFx,
    double alpha,
    double gamma);

std::vector<at::Tensor> imageGradient_cuda(
    at::Tensor I);

at::Tensor imageApplyField_cuda(
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
    float scale);

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
    float scale);

at::Tensor ssdMetric_cuda(
	  torch::Tensor MI,
    torch::Tensor FI,
    int Vsizex,
    int Vsizey,
    int Vsizez,
    float scale);

at::Tensor ccMetric_cuda(
    torch::Tensor MI,
	torch::Tensor FI,
    int Vsizex,
    int Vsizey,
    int Vsizez,
    float scale,
    int Radius);

std::vector<at::Tensor> ssdMetricDerivative_cuda(
	torch::Tensor MI,
	torch::Tensor FI,
	torch::Tensor DFx,
	torch::Tensor DFy,
	torch::Tensor DFz,
	float scale,
	float constData);

std::vector<at::Tensor>   ccMetricDerivative_cuda(
    torch::Tensor MI,
    torch::Tensor FI,
    torch::Tensor DFx,
    torch::Tensor DFy,
    torch::Tensor DFz,
    float scale,
    float constData,
    int Radius);

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
    float scale);


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
bool check_contiguous(std::vector<at::Tensor> tens) {
    for (const auto& x : tens) {
        if (!x.is_contiguous())
            return false;
    }
    return true;
}
bool check_inputs(std::vector<at::Tensor> tens) {
    for (const auto& x : tens) {
        if (!x.is_contiguous() || !x.is_cuda())
            return false;
    }
    return true;
}


std::vector<at::Tensor> imageGradient(
    at::Tensor I) {
  CHECK_INPUT(I);
  return imageGradient_cuda(I);
}

std::vector<at::Tensor> KernelInitialize(
    at::Tensor VFx,
    double alpha,
    double gamma){
      CHECK_INPUT(VFx);
      return KernelInitialize_cuda(
        VFx, 
        alpha,
        gamma);
}

at::Tensor imageApplyField(
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
    float scale){
        CHECK_INPUT(MI);
        CHECK_INPUT(DFx);
        CHECK_INPUT(DFy);
        CHECK_INPUT(DFz);
        return imageApplyField_cuda(
              MI,
              DFx,
              DFy,
              DFz,
              ForwardImageOriginx,
              ForwardImageOriginy,
              ForwardImageOriginz,
              MovingImageOriginx,
              MovingImageOriginy,
              MovingImageOriginz,
              scale);

}

at::Tensor labelApplyField(
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
    float scale){
        CHECK_INPUT(MI);
        CHECK_INPUT(DFx);
        CHECK_INPUT(DFy);
        CHECK_INPUT(DFz);
        return labelApplyField_cuda(
              MI,
              DFx,
              DFy,
              DFz,
              ForwardImageOriginx,
              ForwardImageOriginy,
              ForwardImageOriginz,
              MovingImageOriginx,
              MovingImageOriginy,
              MovingImageOriginz,
              scale);

}

at::Tensor ssdMetric(
    torch::Tensor MI,
    torch::Tensor FI,
    int Vsizex,
    int Vsizey,
    int Vsizez,
    float scale){
        CHECK_INPUT(MI);
        CHECK_INPUT(FI);
        return ssdMetric_cuda(
          MI,
          FI,
          Vsizex,
          Vsizey,
          Vsizez,
          scale);
}

at::Tensor ccMetric(
    torch::Tensor MI,
    torch::Tensor FI,
    int Vsizex,
    int Vsizey,
    int Vsizez,
    float scale,
    int Radius){
        CHECK_INPUT(MI);
        CHECK_INPUT(FI);
        return ccMetric_cuda(
          MI,
          FI,
          Vsizex,
          Vsizey,
          Vsizez,
          scale,
          Radius);
}

std::vector<at::Tensor> ssdMetricDerivative(
	torch::Tensor MI,
	torch::Tensor FI,
	torch::Tensor DFx,
	torch::Tensor DFy,
	torch::Tensor DFz,
	float scale,
	float constData){
    CHECK_INPUT(MI);
    CHECK_INPUT(FI);
    return ssdMetricDerivative_cuda(
                MI,
                FI,
                DFx,
                DFy,
                DFz,
                scale,
                constData);
  }

std::vector<at::Tensor> ccMetricDerivative(
	torch::Tensor MI,
	torch::Tensor FI,
	torch::Tensor DFx,
	torch::Tensor DFy,
	torch::Tensor DFz,
	float scale,
	float constData,
    float radius){
    CHECK_INPUT(MI);
    CHECK_INPUT(FI);
    return ccMetricDerivative_cuda(
                MI,
                FI,
                DFx,
                DFy,
                DFz,
                scale,
                constData,
                radius);
  }


std::vector<at::Tensor> intergrateVelocity(
    const at::Tensor velocityFieldx,
    const at::Tensor velocityFieldy,
    const at::Tensor velocityFieldz,
    float LowerTimeBound,
    int Originx,
    int Originy,
    int Originz,
    float DeltaTime,   
    float TimeSpan,   
    float TimeOrigion,  
    int NumberOfIntegrationSteps,
    float scale){
    CHECK_INPUT(velocityFieldx);
    CHECK_INPUT(velocityFieldy);   
    CHECK_INPUT(velocityFieldz);
    return intergrateVelocity_cuda(
        velocityFieldx,
        velocityFieldy,
        velocityFieldz,
        LowerTimeBound,
        Originx,
        Originy,
        Originz,
        DeltaTime,   
        TimeSpan,   
        TimeOrigion,  
        NumberOfIntegrationSteps,
        scale);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("KernelInitialize", &KernelInitialize, "kernel initialization (cpu)");
  m.def("imageGradient", &imageGradient, "image gradient (CUDA)");
  m.def("imageApplyField", &imageApplyField, "image apply field (CUDA)");
  m.def("labelApplyField", &labelApplyField, "label apply field (CUDA)");
  m.def("ssdMetric", &ssdMetric, "ssd metric (CUDA)");
  m.def("ccMetric", &ccMetric, "cc metric (CUDA)");
  m.def("ssdMetricDerivative", &ssdMetricDerivative, "derivative of ssd metrix (CUDA)");
  m.def("ccMetricDerivative", &ccMetricDerivative, "derivative of cc metrix (CUDA)");
  m.def("intergrateVelocity", &intergrateVelocity, "integration of time-vary velocity field (CUDA)");
}
