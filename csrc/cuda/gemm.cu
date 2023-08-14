#include <iostream>
#include <vector>
#include <cublasLt.h>

#include <stdio.h>
#define BM_CUBLAS_ASSERT(status) do{cublasStatus_t v = (status);if (v != CUBLAS_STATUS_SUCCESS) std::cout << "CUBLAS Error: " #status, __FILE__, __LINE__, __PRETTY_FUNCTION__, cublasGetErrorString(v);}while(0)


class CublasLtHandle {
public:
    static CublasLtHandle& getInstance() {
        static CublasLtHandle instance; // 唯一实例
        return instance;
    }

    cublasLtHandle_t getHandle() {
        return handle;
    }

private:
    CublasLtHandle() {
        cublasLtCreate(&handle);
    }

    ~CublasLtHandle() {
        cublasLtDestroy(handle);
    }

    cublasLtHandle_t handle;
};

template<typename T>
class Linear{
    uint32_t _in_features, _out_features;
    const T* _weight;
    const T* _bias;
    cudaDataType _dtype;

    cublasLtMatmulDesc_t _desc;
    cublasLtMatrixLayout_t _weight_layout;
    
public:
    Linear(const uint32_t in_features, const uint32_t out_features, const T* weight, const T*bias, const int dtype)
        : _in_features(in_features),
        _out_features(out_features),
        _weight(weight),
        _bias(bias) {
        if(dtype == 0){ //fp16
            _dtype = CUDA_R_16F;
            cublasLtMatmulDescCreate(&_desc, CUBLAS_COMPUTE_16F, CUDA_R_16F); 
        }else if(dtype == 1) { //fp32
            _dtype = CUDA_R_32F;
            cublasLtMatmulDescCreate(&_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        }
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        cublasLtMatmulDescSetAttribute(_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
        cublasLtMatrixLayoutCreate(&_weight_layout, _dtype, _out_features, _in_features, _out_features);
        auto status = cublasLtMatmulDescSetAttribute(
                _desc,
                CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                &_bias,
                sizeof(_bias));
    }
    ~Linear(){
        cublasLtMatmulDescDestroy(_desc);
        cublasLtMatrixLayoutDestroy(_weight_layout);
    }
    void forward(const int batch, const T* x, T* out, cudaStream_t stream){
        cublasLtMatrixLayout_t input_layout, out_layout;
        cublasLtMatrixLayoutCreate(&input_layout, _dtype, _in_features, batch, _in_features);
        cublasLtMatrixLayoutCreate(&out_layout, _dtype, _out_features, batch, _out_features);

        CublasLtHandle& handleInstance = CublasLtHandle::getInstance();
        cublasLtHandle_t cublas_handle = handleInstance.getHandle();

        if (_dtype == CUDA_R_32F){
            float alpha = 1, beta = 0;
            cublasLtMatmul(
                    cublas_handle,
                    _desc,
                    &alpha,
                    _weight, _weight_layout,
                    x, input_layout,
                    &beta,
                    out, out_layout,
                    out, out_layout, 
                    0,
                    NULL,
                    0,
                    stream);
        }else{
            half alpha = __float2half(1), beta = __float2half(0);
            cublasLtMatmul(
                    cublas_handle,
                    _desc,
                    &alpha,
                    _weight, _weight_layout,
                    x, input_layout,
                    &beta,
                    out, out_layout,
                    out, out_layout, 
                    NULL,
                    NULL,
                    0,
                    stream);
        }
        cublasLtMatrixLayoutDestroy(input_layout);
        cublasLtMatrixLayoutDestroy(out_layout);
    }
};

void linear_launcher(std::uintptr_t x, 
        std::uintptr_t weight,
        std::uintptr_t bias,
        std::uintptr_t out,
        const int batch,
        const int in_features,
        const int out_features,
        std::uintptr_t stream){
    auto* x_ptr = reinterpret_cast<half*>(x);
    auto* weight_ptr = reinterpret_cast<half*>(weight);
    auto* bias_ptr = reinterpret_cast<half*>(bias);
    auto* out_ptr = reinterpret_cast<half*>(out);
    Linear<half> linear(in_features, out_features, weight_ptr, bias_ptr, 0);
    auto curr_stream = reinterpret_cast<cudaStream_t>(stream);
    linear.forward(batch, x_ptr, out_ptr, curr_stream);
}



//int main(){
//    const int in_features = 4;
//    const int out_features = 4;
//    const int batch = 4;
//    float *weight, *in, *out;
//    cudaMalloc((void**)&weight, in_features * out_features * sizeof(float)); 
//    cudaMalloc((void**)&in, batch * in_features * sizeof(float)); 
//    cudaMalloc((void**)&out, batch * out_features * sizeof(float)); 
//    std::vector<float> h_weight(in_features * out_features);
//    std::vector<float> h_in(batch * in_features);
//    for(int i = 0; i < h_weight.size(); i++){
//        h_weight[i] = 1;
//    }
//    for(int i = 0; i < h_in.size(); i++){
//        h_in[i] = i;
//    }
//    cudaMemcpy(weight, h_weight.data(), h_weight.size() * sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemcpy(in, h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice);
//    Linear<float> linear(in_features, out_features, weight, NULL, 1);
//    linear.forward(batch, in, out, 0);
//    std::vector<float> h_out(batch * out_features);
//    cudaMemcpy(h_out.data(), out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost);
//    for(int i = 0; i < h_out.size(); i++){
//        printf("%f ", h_out[i]);
//    }
//    printf("\n");
//    return 0;
//}
