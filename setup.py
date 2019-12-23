import setuptools
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

ext_modules = [
    CUDAExtension('neural_renderer_torch.cuda.rasterize_cuda', [
        'neural_renderer_torch/cuda/rasterize_cuda.cpp',
        'neural_renderer_torch/cuda/rasterize_cuda_kernel.cu',
    ])
]
setuptools.setup(
    description='A 3D mesh renderer for neural networks',
    author='Hiroharu Kato',
    author_email='hiroharu.kato.1989.10.13@gmail.com',
    url='http://hiroharu-kato.com/projects_en/neural_renderer.html',
    license='MIT License',
    name='neural_renderer_torch',
    test_suite='tests_torch',
    packages=['neural_renderer_torch', 'neural_renderer_torch.cuda'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    install_requires=[],
)
