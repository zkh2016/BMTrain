#include<pybind11/pybind11.h>
#include "nccl.hpp"
#include "adam_cpu.hpp"

void has_nan_inf_launcher(int32_t n,std::uintptr_t g_fp16,std::uintptr_t mid,std::uintptr_t out,std::uintptr_t stream);

void cross_entropy_backward_launcher(
    int32_t m, int32_t n,
    std::uintptr_t grad_output,
    std::uintptr_t target,
    std::uintptr_t softmax,
    std::uintptr_t grad_input,
    int32_t ignore_index,
    std::uintptr_t stream
);
void cross_entropy_backward_inplace_launcher(
    int32_t m, int32_t n,
    std::uintptr_t grad_output,
    std::uintptr_t target,
    std::uintptr_t x,
    int32_t ignore_index,
    std::uintptr_t stream
);
 void cross_entropy_forward_inplace_launcher(
    int32_t m, int32_t n,
    std::uintptr_t x,
    std::uintptr_t target,
    std::uintptr_t output,
    int32_t ignore_index,
    std::uintptr_t stream
);
void cross_entropy_forward_launcher(
    int32_t m, int32_t n,
    std::uintptr_t input,
    std::uintptr_t target,
    std::uintptr_t softmax,
    std::uintptr_t output,
    int32_t ignore_index,
    std::uintptr_t stream
);
void adam_launcher(
    int n,
    std::uintptr_t param_fp32,
    std::uintptr_t param_fp16,
    std::uintptr_t g_fp16,
    std::uintptr_t m_fp16,
    std::uintptr_t v_fp32,
    float beta1, float beta2,
    float eps, float lr,
    float scale,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    uintptr_t stream
);

void linear_launcher(std::uintptr_t x, 
        std::uintptr_t weight,
        std::uintptr_t bias,
        std::uintptr_t out,
        const int batch,
        const int in_features,
        const int out_features,
        const bool trans_a,
        const bool trans_b,
        std::uintptr_t stream
);

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
        std::uintptr_t stream);
