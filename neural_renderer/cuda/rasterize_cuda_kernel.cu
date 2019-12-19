#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace {
template <typename scalar_t>
__global__ void mask_foreground_forward_cuda_kernel(
    const int32_t* face_index,
    const scalar_t* data_in,
    scalar_t* data_out,
    int face_index_size,
    int dim){
        const int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (face_index_size <= i){
            return;
        }

        if (0 <= face_index[i]) {
            float* p1 = (float*)&data_in[i * dim];
            float* p2 = (float*)&data_out[i * dim];
            for (int j = 0; j < dim; j++) {
                *p2++ = *p1++;
            }
        }
    }

template <typename scalar_t>
__global__ void mask_foreground_backward_cuda_kernel(
    const int32_t* face_index,
    const scalar_t* grad_in,
    scalar_t* grad_out,
    int face_index_size,
    int dim){
        const int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (face_index_size <= i){
            return;
        }

        if (0 <= face_index[i]) {
            float* p1 = (float*)&grad_in[i * dim];
            float* p2 = (float*)&grad_out[i * dim];
            for (int j = 0; j < dim; j++) {
                *p1++ = *p2++;
            }
        }
    }


template <typename scalar_t>
__global__ void face_index_map_forward_safe_cuda_kernel(
    const scalar_t* faces,
    int32_t* face_index,
    int face_index_size,
    int num_faces,
    int image_size,
    float near,
    float far,
    int draw_backside,
    float eps
    ){
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i => face_index_size){
            return;
        }

        const int is = image_size;
        const int nf = num_faces;
        const int bn = i / (is * is);
        const int pn = i % (is * is);
        const int yi = pn / is;
        const int xi = pn % is;
        const float yp = (2. * yi + 1 - is) / is;
        const float xp = (2. * xi + 1 - is) / is;

        scalar_t* face = (scalar_t*)&faces[bn * nf * 9];
        float depth_min = far;
        int face_index_min = -1;
        for (int fn = 0; fn < nf; fn++) {
            /* go to next face */
            const float x0 = *face++;
            const float y0 = *face++;
            const float z0 = *face++;
            const float x1 = *face++;
            const float y1 = *face++;
            const float z1 = *face++;
            const float x2 = *face++;
            const float y2 = *face++;
            const float z2 = *face++;

            if (xp < x0 && xp < x1 && xp < x2) continue;
            if (x0 < xp && x1 < xp && x2 < xp) continue;
            if (yp < y0 && yp < y1 && yp < y2) continue;
            if (y0 < yp && y1 < yp && y2 < yp) continue;

            /* return if backside */
            if (!draw_backside) {
                if ((y2 - y0) * (x1 - x0) > (y1 - y0) * (x2 - x0)) {
                    continue;
                }
            }

            /* check in or out */
            float c1 = (yp - y0) * (x1 - x0) - (y1 - y0) * (xp - x0);
            float c2 = (yp - y1) * (x2 - x1) - (y2 - y1) * (xp - x1);
            if (c1 * c2 < 0) {
                continue;
            }

            float c3 = (yp - y2) * (x0 - x2) - (y0 - y2) * (xp - x2);
            if (c2 * c3 < 0) {
                continue;
            }

            float det = x2 * (y0 - y1) + x0 * (y1 - y2) + x1 * (y2 - y0);
            if (abs(det) < 0.00000001) {
                continue;
            }

            /* */
            if (depth_min < z0 && depth_min < z1 && depth_min < z2) {
                continue;
            }

            /* compute w */
            float w[3];
            w[0] = yp * (x2 - x1) + xp * (y1 - y2) + (x1 * y2 - x2 * y1);
            w[1] = yp * (x0 - x2) + xp * (y2 - y0) + (x2 * y0 - x0 * y2);
            w[2] = yp * (x1 - x0) + xp * (y0 - y1) + (x0 * y1 - x1 * y0);
            const float w_sum = w[0] + w[1] + w[2];
            w[0] /= w_sum;
            w[1] /= w_sum;
            w[2] /= w_sum;

            /* compute 1 / zp = sum(w / z) */
            const float zp = 1. / (w[0] / z0 + w[1] / z1 + w[2] / z2);
            if (zp <= near || far <= zp) {
                continue;
            }

            /* check z-buffer */
            if (zp <= depth_min) {
                depth_min = zp;
                face_index_min = fn;
            }
        }

        /* set to global memory */
        face_index[i] = face_index_min;
    }


template <typename scalar_t>
__global__ void face_index_map_forward_unsafe_cuda_kernel(
    const scalar_t* faces,
    const int32_t* face_index_map,
    const scalar_t* depth_map,
    const int32_t* lock,
    int num_faces,
    int image_size,
    float near,
    float far,
    int draw_backside,
    float eps
    ){
        const int i = blockIdx.x * blockDim.x + threadIdx.x;

        const int is = image_size;
        // const int fn = i % num_faces;
        const int bn = i / num_faces;

        float* face = (float*)&faces[i * 9];
        const float x0 = *face++;
        const float y0 = *face++;
        const float z0 = *face++;
        const float x1 = *face++;
        const float y1 = *face++;
        const float z1 = *face++;
        const float x2 = *face++;
        const float y2 = *face++;
        const float z2 = *face++;
        const float xp_min = min(x0, min(x1, x2));
        const float xp_max = max(x0, max(x1, x2));
        const float yp_min = min(y0, min(y1, y2));
        const float yp_max = max(y0, max(y1, y2));
        const int xi_min = ceil((xp_min * is + is - 1) / 2.);
        const int xi_max = floor((xp_max * is + is - 1) / 2.);
        const int yi_min = ceil((yp_min * is + is - 1) / 2.);
        const int yi_max = floor((yp_max * is + is - 1) / 2.);
        for (int xi = xi_min; xi <= xi_max; xi++) {
            for (int yi = yi_min; yi <= yi_max; yi++) {
                const int pi = bn * is * is + yi * is + xi;
                const float yp = (2. * yi + 1 - is) / is;
                const float xp = (2. * xi + 1 - is) / is;

                if (xp < x0 && xp < x1 && xp < x2) continue;
                if (x0 < xp && x1 < xp && x2 < xp) continue;
                if (yp < y0 && yp < y1 && yp < y2) continue;
                if (y0 < yp && y1 < yp && y2 < yp) continue;

                /* return if backside */
                if (!draw_backside) {
                    if ((y2 - y0) * (x1 - x0) > (y1 - y0) * (x2 - x0)) continue;
                }

                /* check in or out */
                float c1 = (yp - y0) * (x1 - x0) - (y1 - y0) * (xp - x0);
                float c2 = (yp - y1) * (x2 - x1) - (y2 - y1) * (xp - x1);
                if (c1 * c2 < 0) continue;
                float c3 = (yp - y2) * (x0 - x2) - (y0 - y2) * (xp - x2);
                if (c2 * c3 < 0) continue;

                float det = x2 * (y0 - y1) + x0 * (y1 - y2) + x1 * (y2 - y0);
                if (abs(det) < 0.00000001) continue;

                /* compute w */
                float w[3];
                w[0] = yp * (x2 - x1) + xp * (y1 - y2) + (x1 * y2 - x2 * y1);
                w[1] = yp * (x0 - x2) + xp * (y2 - y0) + (x2 * y0 - x0 * y2);
                w[2] = yp * (x1 - x0) + xp * (y0 - y1) + (x0 * y1 - x1 * y0);
                const float w_sum = w[0] + w[1] + w[2];
                w[0] /= w_sum;
                w[1] /= w_sum;
                w[2] /= w_sum;

                /* compute 1 / zp = sum(w / z) */
                const float zp = 1. / (w[0] / z0 + w[1] / z1 + w[2] / z2);
                if (zp <= near || far <= zp) continue;

                unsigned int* l = (unsigned int*)&lock[pi];
                while (atomicCAS(l, 0 , 1) != 0);
                /* check z-buffer */
                /*float depth_min = depth_map[pi];
                if (zp <= depth_min) {
                    atomicExch((float*)&depth_map[pi], zp);
                    atomicExch((int*)&face_index_map[pi], fn);
                } */
                atomicExch(l, 0);
            }
        }
    }

template <typename scalar_t>
__global__ void compute_weight_map_cuda_kernel(
    const scalar_t* faces,
    const int32_t* face_index_map,
    const scalar_t* weight_map,
    int face_index_size,
    int num_faces,
    int image_size
    ){
        const int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i => face_index_size){
            return;
        }

        const int fi = face_index_map[i];
        if (fi < 0) {
            return;
        }

        const int is = image_size;
        const int nf = num_faces;
        const int bn = i / (is * is);
        const int pn = i % (is * is);
        const int yi = pn / is;
        const int xi = pn % is;
        const float yp = (2. * yi + 1 - is) / is;
        const float xp = (2. * xi + 1 - is) / is;

        float* face = (float*)&faces[(bn * nf + fi) * 9];
        float x0 = *face++;
        float y0 = *face++;
        float z0 = *face++;
        float x1 = *face++;
        float y1 = *face++;
        float z1 = *face++;
        float x2 = *face++;
        float y2 = *face++;
        float z2 = *face++;

        /* compute w */
        float w[3];
        w[0] = yp * (x2 - x1) + xp * (y1 - y2) + (x1 * y2 - x2 * y1);
        w[1] = yp * (x0 - x2) + xp * (y2 - y0) + (x2 * y0 - x0 * y2);
        w[2] = yp * (x1 - x0) + xp * (y0 - y1) + (x0 * y1 - x1 * y0);
        float w_sum = w[0] + w[1] + w[2];
        if (w_sum < 0) {
            w[0] *= -1;
            w[1] *= -1;
            w[2] *= -1;
        }
        w[0] = max(w[0], 0.);
        w[1] = max(w[1], 0.);
        w[2] = max(w[2], 0.);
        w_sum = w[0] + w[1] + w[2];
        float* wm = (float*)&weight_map[i * 3];
        for (int j = 0; j < 3; j++) {
            w[j] /= w_sum;
            w[j] = max(min(w[j], 1.), 0.);
            wm[j] = w[j];
        }

}
}



at::Tensor mask_foreground_forward_cuda(
        at::Tensor face_index,
        at::Tensor data_in,
        at::Tensor data_out,
        int dim) {

    const int face_index_size = face_index_map.reshape(-1,).size(0)
    const int threads = 1024;
    const dim3 blocks ((face_index_size - 1) / threads +1);

    AT_DISPATCH_FLOATING_TYPES(data_in.type(), "mask_foreground_forward_cuda", ([&] {
      mask_foreground_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
          face_index.data<int32_t>(),
          data_in.data<scalar_t>(),
          data_out.data<scalar_t>(),
          face_index_size,
          dim);
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error in mask_foreground_forward_cuda: %s\n", cudaGetErrorString(err));
    return data_out;
}

at::Tensor mask_foreground_backward_cuda(
        at::Tensor face_index,
        at::Tensor grad_in,
        at::Tensor grad_out,
        int dim) {

    const int threads = 1024;
    const dim3 blocks ((grad_in.size(0) / 3 - 1) / threads + 1);

    AT_DISPATCH_FLOATING_TYPES(grad_in.type(), "mask_foreground_backward_cuda", ([&] {
      mask_foreground_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
          face_index.data<int32_t>(),
          grad_in.data<scalar_t>(),
          grad_out.data<scalar_t>(),
          dim);
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error in mask_foreground_backward_cuda: %s\n", cudaGetErrorString(err));
    return grad_in;
}


at::Tensor face_index_map_forward_safe_cuda(
        at::Tensor faces, at::Tensor face_index, int num_faces,
        int image_size, float near, float far, int draw_backside,
        float eps){

    const int threads = 1024;
    const int face_index_size = face_index.size(0);
    const dim3 blocks ((face_index_size - 1) / threads +1);


    AT_DISPATCH_FLOATING_TYPES(faces.type(), "face_index_map_forward_safe_cuda", ([&] {
      face_index_map_forward_safe_cuda_kernel<scalar_t><<<blocks, threads>>>(
          faces.data<scalar_t>(),
          face_index.data<int32_t>(),
          face_index_size,
          num_faces,
          image_size,
          near,
          far,
          draw_backside,
          eps);
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error in face_index_map_forward_safe_cuda: %s\n", cudaGetErrorString(err));
    return face_index;
}

at::Tensor face_index_map_forward_unsafe_cuda(
        at::Tensor faces, at::Tensor face_index_map, at::Tensor depth_map,
        at::Tensor lock, int num_faces, int image_size, float near, float far,
        int draw_backside, float eps){
    const int threads = 1024;
    const dim3 blocks ((faces.size(0) / 3 - 1) / threads + 1);

    AT_DISPATCH_FLOATING_TYPES(faces.type(), "face_index_map_forward_unsafe_cuda", ([&] {
      face_index_map_forward_unsafe_cuda_kernel<scalar_t><<<blocks, threads>>>(
          faces.data<scalar_t>(),
          face_index_map.data<int32_t>(),
          depth_map.data<scalar_t>(),
          lock.data<int32_t>(),
          num_faces,
          image_size,
          near,
          far,
          draw_backside,
          eps);
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error in face_index_map_forward_unsafe_cuda: %s\n", cudaGetErrorString(err));
    return face_index_map;
}


at::Tensor compute_weight_map_cuda(
        at::Tensor faces, at::Tensor face_index_map, at::Tensor weight_map,
        int num_faces, int image_size){
            const int threads = 1024;
    const int face_index_size = face_index_map.size(0);
    const dim3 blocks ((face_index_size - 1) / threads +1);

    AT_DISPATCH_FLOATING_TYPES(faces.type(), "compute_weight_map_cuda", ([&] {
      compute_weight_map_cuda_kernel<scalar_t><<<blocks, threads>>>(
          faces.data<scalar_t>(),
          face_index_map.data<int32_t>(),
          weight_map.data<scalar_t>(),
          face_index_size,
          num_faces,
          image_size);
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error in compute_weight_map_cuda: %s\n", cudaGetErrorString(err));
    return face_index_map;


}