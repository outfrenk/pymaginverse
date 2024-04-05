# import os
from setuptools.dist import Distribution
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize

# os.environ['CC'] = 'clang'
# os.environ['LDSHARED'] = 'clang -shared'

extensions = []
cmdclass = {'build_ext': build_ext}
dist = Distribution(attrs=dict(
            cmdclass=dict(build_ext=cmdclass['build_ext']),
            ext_modules=cythonize(extensions,
                                  language_level=3,
                                  ),
        )
)
build_ext_cmd = dist.get_command_obj('build_ext')
build_ext_cmd.ensure_finalized()
build_ext_cmd.inplace = 1
build_ext_cmd.run()

# https://stackoverflow.com/questions/63679315/how-to-use-cython-with-poetry
# import os
#
# # See if Cython is installed
# try:
#     from Cython.Build import cythonize
# # Do nothing if Cython is not available
# except ImportError:
#     # Got to provide this function. Otherwise, poetry will fail
#     def build(setup_kwargs):
#         pass
# # Cython is installed. Compile
# else:
#     from setuptools import Extension
#     from setuptools.dist import Distribution
#     from setuptools.command.build_ext import build_ext
#
#     # This function will be executed in setup.py:
#     def build(setup_kwargs):
#         # The file you want to compile
#         extensions = [
#             "banded_tools/build_banded.pyx"
#         ]
#
#         # gcc arguments hack: enable optimizations
#         os.environ['CFLAGS'] = '-O3'
#
#         # Build
#         setup_kwargs.update({
#             'ext_modules': cythonize(
#                 extensions,
#                 language_level=3,
#             ),
#             'cmdclass': {'build_ext': build_ext}
#         })