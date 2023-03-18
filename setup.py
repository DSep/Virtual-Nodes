from setuptools import setup, find_packages


packages = find_packages()
# Ensure that we don't pollute the global namespace.
for p in packages:
    assert p == 'virtual_nodes' or p.startswith('virtual_nodes.')

setup(name='drl-implementation',
      version='1.0.0',
      description='Graph augmentation with virtual nodes',
      url='https://github.com/DSep/virtual-nodes',
      author='sd974 and jw2323',
      author_email='author@cam.ac.uk',
      packages=packages,
      package_dir={'virtual_nodes': 'virtual_nodes'},
      package_data={'virtual_nodes': [
          # 'examples/*.md',
      ]},
      classifiers=[
          "Programming Language :: Python :: 3",
          "Operating System :: OS Independent",
      ])
