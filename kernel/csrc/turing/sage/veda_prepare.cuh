// Veda preparation is deliberately separate from the attention CTA: no
// additional attention registers/shared memory, and no architecture > SM75.
#include <torch/types.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Export.h>
#include <cublas_v2.h>

// Match PyTorch's exported declaration without CUDAContext.h, which also
// requires unrelated cuSPARSE/cuSOLVER development headers on Windows.
// Keep using PyTorch's handle so its stream and workspace management apply.
namespace at::cuda {
TORCH_CUDA_CPP_API cublasHandle_t getCurrentCUDABlasHandle();
}

namespace veda_prepare {
template<class T>
__global__ void scatter_tiles(T* out,const T* value,const int64_t* scatter,
    const int64_t* heads,const int32_t* counts,int full_heads,
    int64_t vs0,int64_t vs1) {
  int tile=blockIdx.x,h=blockIdx.y,d=threadIdx.x;
  int64_t target_head=heads[h];
  for(int r=0;r<counts[tile];++r) {
    int slot=tile*128+r;
    int64_t target=scatter[slot];
    out[(target*full_heads+target_head)*128+d]=value[int64_t(slot)*vs0+h*vs1+d];
  }
}

template<class T>
__global__ void gather_pool(const T* q, const T* k, const T* v,
    const int64_t* indices, const int64_t* heads, const int32_t* counts,
    T* oq, T* ok, T* ov, float* fq, float* fk,
    int hs, int tiles, int video_tiles, int64_t qs0, int64_t qs1,
    int64_t ks0, int64_t ks1, int64_t vs0, int64_t vs1) {
  int tile=blockIdx.x, h=blockIdx.y, d=threadIdx.x;
  int count=counts[tile]; int64_t source_head=heads[h];
  float sq=0, sk=0, lq=INFINITY, lk=INFINITY, uq=-INFINITY, uk=-INFINITY;
  for(int r=0;r<128;++r) {
    int slot=tile*128+r; int64_t src=indices[slot];
    T a=T(0), b=T(0), c=T(0);
    if(r<count) {
      a=q[src*qs0+source_head*qs1+d];
      b=k[src*ks0+source_head*ks1+d];
      c=v[src*vs0+source_head*vs1+d];
      float af=float(a), bf=float(b);
      sq+=af; sk+=bf; lq=fminf(lq,af); lk=fminf(lk,bf);
      uq=fmaxf(uq,af); uk=fmaxf(uk,bf);
    }
    // Physical HND storage lets Sage consume V without a full-size transpose
    // copy. Return an NHD view to preserve the public preparation contract.
    int64_t offset=(int64_t(h)*tiles*128+slot)*128+d;
    oq[offset]=a; ok[offset]=b; ov[offset]=c;
  }
  if(tile<video_tiles) {
    int64_t offset=(int64_t(h)*video_tiles+tile)*384+d;
    fq[offset]=count?sq/count:0; fk[offset]=count?sk/count:0;
    fq[offset+128]=count?uq:0; fk[offset+128]=count?uk:0;
    fq[offset+256]=count?lq:0; fk[offset+256]=count?lk:0;
  }
}

__global__ void prepare_scores(const float* scores, const int32_t* counts,
    float* out, int64_t total, int queries, int columns, int start, int count, int row_start) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=total) return;
  int c=i%count, r=(i/count)%queries;
  int col=start+c;
  // Force the diagonal after masking, matching the reference implementation.
  out[i]=(row_start+r==col)?INFINITY:
      ((counts[col]>0)?scores[(i/count)*columns+col]:-INFINITY);
}

__global__ void finish_selection(const float* best, const int64_t* selected,
    const int32_t* counts, int64_t* indices, bool* keep, int64_t total,
    int queries, int high, int start, int row_start, int low, double fraction) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=total) return;
  int row=row_start+(i/high)%queries, rank=i%high;
  // Separate FP64 multiplies intentionally match the Bresenham quota pattern.
  int extra=floor(double(row+1)*fraction)>floor(double(row)*fraction);
  int64_t col=selected[i]+start;
  indices[i]=col;
  keep[i]=rank<low+extra && best[i]>-INFINITY && counts[col]>0;
}

__global__ void projection_epilogue(const int32_t* acc, const float* xs,
    const float* ws, const __nv_bfloat16* residual, __nv_bfloat16* out,
    int rows, int n, int residual_stride) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  // y indexes heads; x indexes rows/channels of a head.
  if(i>=int64_t(rows)*n) return;
  int h=blockIdx.y, r=i/n, d=i%n;
  int64_t offset=int64_t(h)*rows*n+i;
  float f=float(acc[offset])*xs[h*rows+r]*ws[h*n+d];
  // Match existing BF16 GEMM output followed by BF16 residual addition.
  out[offset]=__float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(f))+
      __bfloat162float(residual[(int64_t(h)*rows+r)*residual_stride+d]));
}

__global__ void routes(const int64_t* indices, const bool* keep,
    const int32_t* counts, int32_t* out, int selected, int queries,
    int video_tiles, int tiles, int words) {
  int row=blockIdx.x;
  auto* bits=reinterpret_cast<unsigned int*>(out+int64_t(row)*words);
  for(int w=threadIdx.x;w<words;w+=blockDim.x) bits[w]=0;
  __syncthreads();
  for(int j=threadIdx.x;j<selected;j+=blockDim.x) {
    int64_t p=int64_t(row)*selected+j;
    if(keep[p]) {
      int t=indices[p]; int n=counts[t];
      if(n>0) atomicOr(bits+(2*t)/32,1u<<((2*t)%32));
      if(n>64) atomicOr(bits+(2*t+1)/32,1u<<((2*t+1)%32));
    }
  }
  for(int t=video_tiles+threadIdx.x;t<tiles;t+=blockDim.x) {
    int n=counts[t];
    if(n>0) atomicOr(bits+(2*t)/32,1u<<((2*t)%32));
    if(n>64) atomicOr(bits+(2*t+1)/32,1u<<((2*t+1)%32));
  }
}
} // namespace veda_prepare

void veda_scatter_tiles(torch::Tensor out,torch::Tensor value,torch::Tensor scatter,
    torch::Tensor heads,torch::Tensor counts) {
  TORCH_CHECK(out.is_cuda() && out.dim()==3 && out.is_contiguous() && out.size(2)==128 &&
    value.dim()==3 && value.size(2)==128 && value.stride(2)==1 &&
    value.scalar_type()==out.scalar_type() && value.size(0)==counts.numel()*128 &&
    value.size(1)==heads.numel() && scatter.numel()==value.size(0) &&
    heads.scalar_type()==torch::kInt64 && scatter.scalar_type()==torch::kInt64 &&
    counts.scalar_type()==torch::kInt32,"Veda scatter shapes/dtypes invalid");
  for(const auto& t:{value,scatter,heads,counts}) TORCH_CHECK(t.device()==out.device(),"Veda scatter device mismatch");
  for(const auto& t:{scatter,heads,counts}) TORCH_CHECK(t.dim()==1 && t.is_contiguous(),"Veda scatter indices invalid");
  c10::cuda::CUDAGuard guard(out.device());
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,out.scalar_type(),"veda_scatter_tiles",[&]{
    veda_prepare::scatter_tiles<scalar_t><<<dim3(counts.numel(),heads.numel()),128,0,c10::cuda::getCurrentCUDAStream()>>>(
      out.data_ptr<scalar_t>(),value.data_ptr<scalar_t>(),scatter.data_ptr<int64_t>(),heads.data_ptr<int64_t>(),
      counts.data_ptr<int32_t>(),out.size(1),value.stride(0),value.stride(1));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> veda_gather_pool(torch::Tensor q, torch::Tensor k,
    torch::Tensor v, torch::Tensor indices, torch::Tensor heads,
    torch::Tensor counts, int64_t video_tiles) {
  TORCH_CHECK(q.is_cuda() && q.dim()==3 && q.size(2)==128 && q.stride(2)==1,
              "Veda gather needs CUDA NHD D128 input");
  TORCH_CHECK(k.sizes()==q.sizes() && v.sizes()==q.sizes() &&
              k.scalar_type()==q.scalar_type() && v.scalar_type()==q.scalar_type() &&
              k.device()==q.device() && v.device()==q.device() &&
              k.stride(2)==1 && v.stride(2)==1,"Veda QKV mismatch");
  for(const auto& t:{indices,heads})
    TORCH_CHECK(t.device()==q.device() && t.scalar_type()==torch::kInt64 &&
                t.dim()==1 && t.is_contiguous(),"Veda layout must be contiguous CUDA int64");
  TORCH_CHECK(counts.device()==q.device() && counts.scalar_type()==torch::kInt32 &&
              counts.dim()==1 && counts.is_contiguous(),"Veda counts must be CUDA int32");
  TORCH_CHECK(indices.numel()==counts.numel()*128 && video_tiles>=0 &&
              video_tiles<=counts.numel() && heads.numel()>0,"Veda layout size mismatch");
  c10::cuda::CUDAGuard guard(q.device());
  auto oq=torch::empty({heads.numel(),indices.numel(),128},q.options());
  auto ok=torch::empty_like(oq),ov=torch::empty_like(oq);
  auto fq=torch::empty({heads.numel(),video_tiles,384},q.options().dtype(torch::kFloat32));
  auto fk=torch::empty_like(fq);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,q.scalar_type(),"veda_gather_pool",[&]{
    veda_prepare::gather_pool<scalar_t><<<dim3(counts.numel(),heads.numel()),128,0,c10::cuda::getCurrentCUDAStream()>>>(
      q.data_ptr<scalar_t>(),k.data_ptr<scalar_t>(),v.data_ptr<scalar_t>(),
      indices.data_ptr<int64_t>(),heads.data_ptr<int64_t>(),counts.data_ptr<int32_t>(),
      oq.data_ptr<scalar_t>(),ok.data_ptr<scalar_t>(),ov.data_ptr<scalar_t>(),
      fq.data_ptr<float>(),fk.data_ptr<float>(),heads.numel(),counts.numel(),video_tiles,
      q.stride(0),q.stride(1),k.stride(0),k.stride(1),v.stride(0),v.stride(1));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {oq.transpose(0,1),ok.transpose(0,1),ov.transpose(0,1),fq,fk};
}

torch::Tensor veda_prepare_scores(torch::Tensor scores, torch::Tensor counts,
    int64_t start, int64_t stop, int64_t row_start) {
  TORCH_CHECK(scores.is_cuda() && scores.dim()==3 && scores.is_contiguous() &&
    scores.scalar_type()==torch::kFloat32 && counts.scalar_type()==torch::kInt32 &&
    counts.device()==scores.device() && counts.is_contiguous() && counts.dim()==1,
    "Veda score preparation requires contiguous CUDA FP32 scores and int32 counts");
  TORCH_CHECK(start>=0 && stop>start && stop<=scores.size(2) && counts.numel()>=scores.size(2) &&
    row_start>=0 && row_start+scores.size(1)<=scores.size(2),"Veda score ranges invalid");
  c10::cuda::CUDAGuard guard(scores.device());
  auto out=torch::empty({scores.size(0),scores.size(1),stop-start},scores.options());
  if(out.numel()) veda_prepare::prepare_scores<<<(out.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
    scores.data_ptr<float>(),counts.data_ptr<int32_t>(),out.data_ptr<float>(),out.numel(),
    scores.size(1),scores.size(2),start,stop-start,row_start);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<torch::Tensor> veda_finish_selection(torch::Tensor best, torch::Tensor selected,
    torch::Tensor counts, int64_t start, int64_t row_start, int64_t low, double fraction) {
  TORCH_CHECK(best.is_cuda() && best.dim()==3 && best.scalar_type()==torch::kFloat32 &&
    selected.sizes()==best.sizes() && selected.scalar_type()==torch::kInt64 &&
    counts.dim()==1 && counts.scalar_type()==torch::kInt32 && low>=1 && low<=best.size(2) &&
    start>=0 && start<counts.numel() && row_start>=0 && fraction>=0 && fraction<=1,
    "Veda TopK finish inputs invalid");
  for(const auto& t:{best,selected,counts}) TORCH_CHECK(t.device()==best.device() && t.is_contiguous(),
    "Veda TopK finish device/stride mismatch");
  c10::cuda::CUDAGuard guard(best.device());
  auto indices=torch::empty_like(selected), keep=torch::empty(best.sizes(),best.options().dtype(torch::kBool));
  if(best.numel()) veda_prepare::finish_selection<<<(best.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
    best.data_ptr<float>(),selected.data_ptr<int64_t>(),counts.data_ptr<int32_t>(),
    indices.data_ptr<int64_t>(),keep.data_ptr<bool>(),best.numel(),best.size(1),best.size(2),
    start,row_start,low,fraction);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {indices,keep};
}

torch::Tensor veda_projection_int8(torch::Tensor x, torch::Tensor w,
    torch::Tensor xs, torch::Tensor ws, torch::Tensor residual) {
  TORCH_CHECK(x.is_cuda() && x.dim()==3 && w.dim()==3 && x.is_contiguous() &&
    w.is_contiguous() && x.scalar_type()==torch::kInt8 && w.scalar_type()==torch::kInt8,
    "Veda predictor expects contiguous batched INT8 operands");
  int h=x.size(0),m=x.size(1),k=x.size(2),n=w.size(1);
  TORCH_CHECK(h>0 && m>0 && n==128 && k%4==0 && w.size(0)==h && w.size(2)==k,
    "Veda predictor dimensions mismatch");
  for(const auto& t:{w,xs,ws,residual}) TORCH_CHECK(t.device()==x.device() && t.is_contiguous(),"Veda predictor device/strides mismatch");
  TORCH_CHECK(xs.scalar_type()==torch::kFloat32 && ws.scalar_type()==torch::kFloat32 &&
    xs.numel()==h*m && ws.numel()==h*n && residual.scalar_type()==torch::kBFloat16 &&
    residual.dim()==3 && residual.size(0)==h && residual.size(1)==m && residual.size(2)>=n,
    "Veda predictor scales/residual mismatch");
  c10::cuda::CUDAGuard guard(x.device());
  auto acc=torch::empty({h,m,n},x.options().dtype(torch::kInt32));
  auto out=torch::empty({h,m,n},x.options().dtype(torch::kBFloat16));
  auto handle=at::cuda::getCurrentCUDABlasHandle();
  int32_t alpha=1,beta=0;
  // Column-major W^T [N,K] times X^T [K,M] gives row-major [M,N].
  auto status=cublasGemmStridedBatchedEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,
    &alpha,w.data_ptr<int8_t>(),CUDA_R_8I,k,int64_t(n)*k,
    x.data_ptr<int8_t>(),CUDA_R_8I,k,int64_t(m)*k,&beta,
    acc.data_ptr<int32_t>(),CUDA_R_32I,n,int64_t(m)*n,h,
    CUBLAS_COMPUTE_32I,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  TORCH_CHECK(status==CUBLAS_STATUS_SUCCESS,"Veda batched INT8 GEMM failed: ",int(status));
  veda_prepare::projection_epilogue<<<dim3((m*n+255)/256,h),256,0,c10::cuda::getCurrentCUDAStream()>>>(
    acc.data_ptr<int32_t>(),xs.data_ptr<float>(),ws.data_ptr<float>(),
    reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr()),
    reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),m,n,residual.size(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor veda_pack_routes(torch::Tensor indices, torch::Tensor keep,
    torch::Tensor counts,int64_t video_tiles) {
  TORCH_CHECK(indices.is_cuda() && indices.dim()==3 && indices.scalar_type()==torch::kInt64 &&
    keep.sizes()==indices.sizes() && keep.scalar_type()==torch::kBool &&
    counts.dim()==1 && counts.scalar_type()==torch::kInt32,
    "Veda route inputs invalid");
  for(const auto& t:{indices,keep,counts}) TORCH_CHECK(t.device()==indices.device() && t.is_contiguous(),"Veda routes device/strides mismatch");
  TORCH_CHECK(video_tiles>=0 && video_tiles<=counts.numel(),"Veda video tile count invalid");
  c10::cuda::CUDAGuard guard(indices.device());
  int words=(counts.numel()*2+31)/32;
  auto out=torch::empty({1,indices.size(0),indices.size(1),words},indices.options().dtype(torch::kInt32));
  if(indices.size(0)*indices.size(1)>0)
    veda_prepare::routes<<<indices.size(0)*indices.size(1),128,0,c10::cuda::getCurrentCUDAStream()>>>(
      indices.data_ptr<int64_t>(),keep.data_ptr<bool>(),counts.data_ptr<int32_t>(),out.data_ptr<int32_t>(),
      indices.size(2),indices.size(1),video_tiles,counts.numel(),words);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
