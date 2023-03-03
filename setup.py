from setuptools import setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='eng-syl',
      version='0.1.2',
      description='English word syllabifier',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/ellipse-liu/eng-syl',
      author='ellipse-liu',
      author_email='timothys.new.email@gmail.comm',
      license='MIT',
      packages=['eng_syl'],
      package_data={'eng_syl': ['e2i.pkl', 'syllabler_best_weights.h5']},
      install_requires=[
          'tensorflow',
          'numpy',
      ],
      keywords=['Syllable', 'NLP', 'psycholinguistics'],
      classifiers=[
          'Development Status :: 3 - Alpha',
          # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
          'Intended Audience :: Developers',  # Define that your audience are developers
          'Topic :: Software Development :: Build Tools',
          'License :: OSI Approved :: MIT License',  # Again, pick a license
          'Programming Language :: Python :: 3.8',
      ],
      zip_safe=False)
