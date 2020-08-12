#include <torch/extension.h>
#include <vector>


#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/// CUDA kernel definitions
at::Tensor mask_foreground_forward_cuda(
        at::Tensor face_index, at::Tensor data_in,
        at::Tensor data_out, int dim);

at::Tensor mask_foreground_backward_cuda(
        at::Tensor face_index, at::Tensor grad_in,
        at::Tensor grad_out, int dim);

at::Tensor face_index_map_forward_safe_cuda(
        at::Tensor faces, at::Tensor face_index, int num_faces,
        int image_size, float near, float far, int draw_backside,
        float eps, float depth_min_delta);

at::Tensor face_index_map_forward_unsafe_cuda(
        at::Tensor faces, at::Tensor face_index_map, at::Tensor depth_map,
        at::Tensor lock, int num_faces, int image_size, float near, float far,
        int draw_backside, float eps);


at::Tensor compute_weight_map_cuda(
        at::Tensor faces, at::Tensor face_index_map, at::Tensor weight_map,
        int num_faces, int image_size);


// Wrapper implementations
at::Tensor mask_foreground_forward(
        at::Tensor face_index, at::Tensor data_in, at::Tensor data_out, int dim) {

    CHECK_INPUT(face_index);
    CHECK_INPUT(data_in);
    CHECK_INPUT(data_out);

    return mask_foreground_forward_cuda(face_index, data_in, data_out, dim);
}

at::Tensor mask_foreground_backward(
        at::Tensor face_index, at::Tensor data_in, at::Tensor data_out, int dim) {

    CHECK_INPUT(face_index);
    CHECK_INPUT(data_in);
    CHECK_INPUT(data_out);

    return mask_foreground_backward_cuda(face_index, data_in, data_out, dim);
}

at::Tensor face_index_map_forward_safe(
        at::Tensor faces, at::Tensor face_index, int num_faces,
        int image_size, float near, float far, int draw_backside,
        float eps, float depth_min_delta) {

    CHECK_INPUT(faces);
    CHECK_INPUT(face_index);

    return face_index_map_forward_safe_cuda(faces, face_index, num_faces, image_size, near,
                                            far, draw_backside, eps, depth_min_delta);
}

at::Tensor face_index_map_forward_unsafe(
        at::Tensor faces, at::Tensor face_index_map, at::Tensor depth_map,
        at::Tensor lock, int num_faces, int image_size, float near, float far,
        int draw_backside, float eps) {

    CHECK_INPUT(faces);
    CHECK_INPUT(face_index_map);
    CHECK_INPUT(depth_map);
    CHECK_INPUT(lock);

    return face_index_map_forward_unsafe_cuda(faces, face_index_map, depth_map, lock, num_faces,
                                              image_size, near, far, draw_backside, eps);
}

at::Tensor compute_weight_map_c(
        at::Tensor faces, at::Tensor face_index_map, at::Tensor weight_map,
        int num_faces, int image_size) {

    CHECK_INPUT(faces);
    CHECK_INPUT(face_index_map);
    CHECK_INPUT(weight_map);

    return compute_weight_map_cuda(faces, face_index_map, weight_map, num_faces, image_size);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mask_foreground_forward", &mask_foreground_forward, "MASK_FOREGROUND_FORWARD (CUDA)");
    m.def("mask_foreground_backward", &mask_foreground_backward, "MASK_FOREGROUND_BACKWARD (CUDA)");
    m.def("face_index_map_forward_safe", &face_index_map_forward_safe, "FACE_INDEX_MAP_FORWARD_SAFE (CUDA)");
    m.def("face_index_map_forward_unsafe", &face_index_map_forward_unsafe, "FACE_INDEX_MAP_FORWARD_UNSAFE (CUDA)");
    m.def("compute_weight_map_c", &compute_weight_map_c, "COMPUTE_WEIGHT_MAP_C (CUDA)");
}


