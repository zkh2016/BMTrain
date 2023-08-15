#include <iostream>
#include <vector>
#include <algorithm>
#include <mutex>
#include <unordered_map>
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

class GemmEpilogueAlgoCache {
	public:
		size_t workspace_size = static_cast<size_t>(4) * 1024 * 1024;
		void* workspace;

		static GemmEpilogueAlgoCache& Instance() {
			static GemmEpilogueAlgoCache instance(30);
			return instance;
		}

		GemmEpilogueAlgoCache(GemmEpilogueAlgoCache const&) = delete;
		void operator=(GemmEpilogueAlgoCache const&) = delete;

		cublasLtMatmulAlgo_t* GetGemmAlgo(cublasLtHandle_t lt_handle,
				cublasLtMatmulDesc_t op_desc,
				cublasLtMatrixLayout_t a_desc,
				cublasLtMatrixLayout_t b_desc,
				cublasLtMatrixLayout_t c_desc,
				const void* alpha,
				const void* beta,
				const void* a,
				const void* b,
				void* c,
				cudaStream_t stream){
				//void* workspace,
				//const size_t workspace_size) {
			if (search_times_ <= 0) return nullptr;

			int64_t seed = 0;
			std::hash<int64_t> hash_fn;

			HashMatmulDesc_(op_desc, &seed, hash_fn);
			HashMatrixLayoutDesc_(a_desc, &seed, hash_fn);
			HashMatrixLayoutDesc_(b_desc, &seed, hash_fn);
			HashMatrixLayoutDesc_(c_desc, &seed, hash_fn);

			cublasLtMatmulAlgo_t ret;
			{
				std::lock_guard<std::mutex> lock(cache_mutex_);
				auto it = map_.find(seed);
				if (it != map_.end()) {
					return &(it->second);
				}
			}

			cublasLtMatmulPreference_t preference;
			(
					cublasLtMatmulPreferenceCreate(&preference));
			(
					cublasLtMatmulPreferenceSetAttribute(
						preference,
						CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
						&workspace_size,
						sizeof(workspace_size)));

			int returned_results = 0;
			std::vector<cublasLtMatmulHeuristicResult_t> heuristic_results(
					requested_algo_count_);
            auto heur_status = cublasLtMatmulAlgoGetHeuristic(lt_handle,
                    op_desc,
                    a_desc,
                    b_desc,
                    c_desc,
                    c_desc,
                    preference,
                    requested_algo_count_,
                    heuristic_results.data(),
                    &returned_results);


			(
					cublasLtMatmulPreferenceDestroy(preference));

			int best_algo_idx = -1;
			float best_algo_time = 0;

			// Run 100 times for warmup
			int warmup_algo_idx = 0;
			for (int t = 0; t < 100; t++) {
				cublasStatus_t status =
					cublasLtMatmul(lt_handle,
							op_desc,
							alpha,
							a,
							a_desc,
							b,
							b_desc,
							beta,
							c,
							c_desc,
							c,
							c_desc,
							&heuristic_results[warmup_algo_idx].algo,
							workspace,
							workspace_size,
							stream);
				if (status != CUBLAS_STATUS_SUCCESS) {
					t = -1;
					warmup_algo_idx += 1;
					if (warmup_algo_idx == requested_algo_count_) {
						std::cout << "No GEMM epilogue algorithm support!" << status << std::endl;
					}
				}
			}

			cudaEvent_t start_event, stop_event;
			(cudaEventCreate(&start_event));
			(cudaEventCreate(&stop_event));

			for (int algo_idx = 0; algo_idx < returned_results; ++algo_idx) {
				float curr_time = 0;
				for (int check_idx = 0; check_idx < search_times_; check_idx++) {
					float time = 0;
					(cudaEventRecord(start_event, stream));

					cublasStatus_t status =
						cublasLtMatmul(lt_handle,
								op_desc,
								alpha,
								a,
								a_desc,
								b,
								b_desc,
								beta,
								c,
								c_desc,
								c,
								c_desc,
								&heuristic_results[algo_idx].algo,
								workspace,
								workspace_size,
								stream);

					(cudaEventRecord(stop_event, stream));
					(cudaEventSynchronize(stop_event));
					(
							cudaEventElapsedTime(&time, start_event, stop_event));
					curr_time += time;
					if (status != CUBLAS_STATUS_SUCCESS) {
						curr_time = 3.40282e+038;  // Max Value of float
						break;
					}
				}

				curr_time = curr_time / search_times_;
				if (curr_time < best_algo_time || algo_idx == 0) {
					best_algo_idx = algo_idx;
					best_algo_time = curr_time;
				}
			}

			(cudaEventDestroy(start_event));
			(cudaEventDestroy(stop_event));

			if (best_algo_idx == -1) {
				std::cout << "No GEMM epilogue algorithm support!\n";
			}

			ret = heuristic_results[best_algo_idx].algo;

			std::lock_guard<std::mutex> lock(cache_mutex_);
			auto& algo_in_map = map_[seed];
			algo_in_map = ret;
			return &algo_in_map;
		}

	private:
		explicit GemmEpilogueAlgoCache(int search_times)
			: search_times_(search_times) {
				map_.clear();
				cudaError_t status = cudaMalloc(&workspace, workspace_size);
				std::cout << "create GemmEpilogueAlgoCache " << status << std::endl;
		}
		std::unordered_map<int64_t, cublasLtMatmulAlgo_t> map_;
		int search_times_;
		const int requested_algo_count_ = 10;
		std::mutex cache_mutex_;

		void HashMatmulDesc_(cublasLtMatmulDesc_t desc,
				int64_t* seed,
				const std::hash<int64_t>& hash_fn) {
			size_t size_to_write;
			int trans_a, trans_b;
			uint32_t epilogue;

			(cublasLtMatmulDescGetAttribute(
						desc,
						CUBLASLT_MATMUL_DESC_TRANSA,
						&trans_a,
						sizeof(trans_a),
						&size_to_write));
			HashValue_(seed, hash_fn, static_cast<int64_t>(trans_a));

			(cublasLtMatmulDescGetAttribute(
						desc,
						CUBLASLT_MATMUL_DESC_TRANSB,
						&trans_b,
						sizeof(trans_b),
						&size_to_write));
			HashValue_(seed, hash_fn, static_cast<int64_t>(trans_b));

			(cublasLtMatmulDescGetAttribute(
						desc,
						CUBLASLT_MATMUL_DESC_EPILOGUE,
						&epilogue,
						sizeof(epilogue),
						&size_to_write));
			HashValue_(seed, hash_fn, static_cast<int64_t>(epilogue));
		}

		void HashMatrixLayoutDesc_(cublasLtMatrixLayout_t desc,
				int64_t* seed,
				const std::hash<int64_t>& hash_fn) {
			size_t size_to_write;
			uint32_t dtype;
			int32_t batch;
			uint64_t row, col;
			int64_t ld, batch_offset;

			(cublasLtMatrixLayoutGetAttribute(
						desc,
						CUBLASLT_MATRIX_LAYOUT_TYPE,
						&dtype,
						sizeof(dtype),
						&size_to_write));
			HashValue_(seed, hash_fn, static_cast<int64_t>(dtype));

			(cublasLtMatrixLayoutGetAttribute(
						desc,
						CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
						&batch,
						sizeof(batch),
						&size_to_write));
			HashValue_(seed, hash_fn, static_cast<int64_t>(batch));

			(cublasLtMatrixLayoutGetAttribute(
						desc, CUBLASLT_MATRIX_LAYOUT_ROWS, &row, sizeof(row), &size_to_write));
			HashValue_(seed, hash_fn, static_cast<int64_t>(row));

			(cublasLtMatrixLayoutGetAttribute(
						desc, CUBLASLT_MATRIX_LAYOUT_COLS, &col, sizeof(col), &size_to_write));
			HashValue_(seed, hash_fn, static_cast<int64_t>(col));

			(cublasLtMatrixLayoutGetAttribute(
						desc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld), &size_to_write));
			HashValue_(seed, hash_fn, static_cast<int64_t>(ld));

			(cublasLtMatrixLayoutGetAttribute(
						desc,
						CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
						&batch_offset,
						sizeof(batch_offset),
						&size_to_write));
			HashValue_(seed, hash_fn, static_cast<int64_t>(batch_offset));
		}

		void HashValue_(int64_t* seed,
				const std::hash<int64_t>& hash_fn,
				int64_t value) {
			*seed ^= hash_fn(value) + 0x9e3779b9 + (*seed << 6) + (*seed >> 2);
		}
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
		auto algo = GemmEpilogueAlgoCache::Instance().GetGemmAlgo(cublas_handle,
				desc,
				b_layout,
				a_layout,
				c_layout,
				&alpha,
				&beta,
			 	B,	
				A,
				C,
				stream);

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
            n, m, k, 
            true, false, 
            dweight_ptr, curr_stream);
}
