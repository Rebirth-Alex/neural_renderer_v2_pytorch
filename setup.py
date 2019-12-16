import setuptools
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

ext_modules = [
    CUDAExtension('neural_renderer.cuda.rasterize_cuda', [
        'neural_renderer/cuda/rasterize_cuda.cpp',
        'neural_renderer/cuda/rasterize_cuda_kernel.cu',
    ])
]
setuptools.setup(
    description='A 3D mesh renderer for neural networks',
    author='Hiroharu Kato',
    author_email='hiroharu.kato.1989.10.13@gmail.com',
    url='http://hiroharu-kato.com/projects_en/neural_renderer.html',
    license='MIT License',
    name='neural_renderer',
    test_suite='tests',
    packages=['neural_renderer', 'neural_renderer.cuda'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    install_requires=[],
)
