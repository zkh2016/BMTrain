#include <iostream>
#include <vector>
#include <cublasLt.h>

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

template<typename T, bool backward=false>
void gemm(const T* A, const T* B, const T* Bias, 
        const int M, const int K, const int N,
        const bool trans_a, const bool trans_b,
        T* C,
        cudaStream_t stream){
    cudaDataType dtype;
    cublasLtMatmulDesc_t desc;
    if (std::is_same<T, float>::value){
        dtype = CUDA_R_32F;
        cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, dtype); 
    }else{
        dtype = CUDA_R_16F;
        cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_16F, dtype); 
    }

    cublasLtMatrixLayout_t a_layout, b_layout, c_layout;
    if (Bias != nullptr){
        if (backward){
            cublasLtEpilogue_t fused_db = CUBLASLT_EPILOGUE_BGRADB;
            cublasLtMatmulDescSetAttribute(
                    desc,
                    CUBLASLT_MATMUL_DESC_EPILOGUE,
                    &fused_db,
                    sizeof(fused_db));
        }else{
            cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
            cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
        }
        cublasLtMatmulDescSetAttribute(
                desc,
                CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                &Bias,
                sizeof(Bias));
    }
    cublasLtMatrixLayoutCreate(&a_layout, dtype, K, M, K);
    cublasOperation_t transa = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(
                  desc, CUBLASLT_MATMUL_DESC_TRANSB, &transa, sizeof(transa));
    cublasLtMatmulDescSetAttribute(
                  desc, CUBLASLT_MATMUL_DESC_TRANSA, &transb, sizeof(transb));
    if(trans_a){
        cublasLtMatrixLayoutCreate(&a_layout, dtype, M, K, M);
    }else{
        cublasLtMatrixLayoutCreate(&a_layout, dtype, K, M, K);
    }

    if(trans_b){
        cublasLtMatrixLayoutCreate(&b_layout, dtype, K, N, K);
    }else{
        cublasLtMatrixLayoutCreate(&b_layout, dtype, N, K, N);
    }
    cublasLtMatrixLayoutCreate(&c_layout, dtype, N, M, N);

    CublasLtHandle& handleInstance = CublasLtHandle::getInstance();
    cublasLtHandle_t cublas_handle = handleInstance.getHandle();

    if (dtype == CUDA_R_32F){
        float alpha = 1, beta = 0;
        cublasLtMatmul(
                cublas_handle,
                desc,
                &alpha,
                B, b_layout,
                A, a_layout,
                &beta,
                C, c_layout,
                C, c_layout, 
                0,
                NULL,
                0,
                stream);
    }else{
        half alpha = __float2half(1), beta = __float2half(0);
        cublasLtMatmul(
                cublas_handle,
                desc,
                &alpha,
                B, b_layout,
                A, a_layout,
                &beta,
                C, c_layout,
                C, c_layout, 
                NULL,
                NULL,
                0,
                stream);
    }
    cublasLtMatrixLayoutDestroy(a_layout);
    cublasLtMatrixLayoutDestroy(b_layout);
    cublasLtMatrixLayoutDestroy(c_layout);
    cublasLtMatmulDescDestroy(desc);
}

void linear_launcher(std::uintptr_t x, 
        std::uintptr_t weight,
        std::uintptr_t bias,
        std::uintptr_t out,
        const int batch,
        const int in_features,
        const int out_features,
        const bool trans_a,
        const bool trans_b,
        std::uintptr_t stream){
    auto* x_ptr = reinterpret_cast<half*>(x);
    auto* weight_ptr = reinterpret_cast<half*>(weight);
    auto* bias_ptr = reinterpret_cast<half*>(bias);
    auto* out_ptr = reinterpret_cast<half*>(out);
    auto curr_stream = reinterpret_cast<cudaStream_t>(stream);
    gemm<half>(x_ptr, weight_ptr, bias_ptr, 
            batch, in_features, out_features, 
            trans_a, trans_b, 
            out_ptr, curr_stream);
}

void linear_backward_launcher(std::uintptr_t x, 
        std::uintptr_t weight,
        std::uintptr_t bias,
        std::uintptr_t out,
        std::uintptr_t dout,
        std::uintptr_t dx,
        std::uintptr_t dweight,
        std::uintptr_t dbias,
        const int batch,
        const int in_features,
        const int out_features,
        const bool trans_a,
        const bool trans_b,
        std::uintptr_t stream){
    auto* x_ptr = reinterpret_cast<half*>(x);
    auto* weight_ptr = reinterpret_cast<half*>(weight);
    auto* bias_ptr = reinterpret_cast<half*>(bias);
    auto* out_ptr = reinterpret_cast<half*>(out);
    auto* dout_ptr = reinterpret_cast<half*>(dout);
    auto* dx_ptr = reinterpret_cast<half*>(dx);
    auto* dweight_ptr = reinterpret_cast<half*>(dweight);
    auto* dbias_ptr = reinterpret_cast<half*>(dbias);
    const int m = batch;
    const int k = in_features;
    const int n = out_features;
    auto curr_stream = reinterpret_cast<cudaStream_t>(stream);
    //weight(n, k) in forward
    //dx(m, k) = dout(m, n) * weight(n, k)
    gemm<half, true>(dout_ptr, weight_ptr, nullptr, 
            m, n, k, 
            false, false, 
            dx_ptr, curr_stream);
    //dweight(n, k) = Trans(dout(m, n)) * x(m,k) 
    //fused dbias
    gemm<half, true>(dout_ptr, x_ptr, dbias_ptr, 
            m, n, k, 
            true, false, 
            dweight_ptr, curr_stream);
}
