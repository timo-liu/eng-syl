from setuptools import setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='eng-syl',
      version='3.0.5',
      description='English word syllabifier and extended syllable analysis tool',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/ellipse-liu/eng-syl',
      author='Timothy-Liu',
      author_email='timothys.new.email@gmail.com',
      license='MIT',
      packages=['eng_syl'],
      package_data={'eng_syl': ['*.pkl', '*.h5']},
      install_requires=[
          'tensorflow',
          'numpy',
      ],
      keywords=['Syllable', 'NLP', 'psycholinguistics'],
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Topic :: Software Development :: Build Tools',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.8',
      ],
      zip_safe=False)
