<!-- Source : Best-README-template on : https://github.com/othneildrew/Best-README-Template -->

<h3 align="center">Root Detection On 2D+T Images</h3>

  <p align="center">
    Project designed to use Deep learning in images of growing Arabidopsis Thaliana roots to recognize which pixel is part of root, and what ispart of background
    or noise
    <br />

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


<!-- About the project -->
## About the project

The project, done in the context of an internship at CIRAD, allow the user to train a model based on 2D+T image of roots growing. Those images should be TIFF images containing different slices representing each times a different period of the plant's growth. You can then use this project to train a model showing which pixels belong to a root, and which ones are background.
<br/><br/>
You also just use the pre-compiled models to just apply a mask on your 2D+t image in order to get a 2D image showing the growth of your root using grayscale.
<br/><br/>
If you need an older commit of this repository, you can consult Wilgawox/basic_Deep_cirad

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

Requirements:
A Python 3 environment with python libraries:
- tensorflow (or tensorflow-gpu if you have a gpu) with all dependancies (https://www.tensorflow.org/install). 
- yaml
<br/>
If you want to train the neural network, you will need tensorflow GPU, with all dependancies (https://www.tensorflow.org/install). 
<br/>
Or you can install it with `pip install tensorflow`
<br/>

### Installation

You can use this package using ""TODO : add Git command to write it locally""


  
## Preparing your dataset

**Input data:** 
  - 2D + t image sequence acquired with an imaging automaton presenting a growing root system in a petri dish. Data should be organized in an input directory, one subdirectory per dish, with name patterning such as img_XX.tif, in order that ImageJ detect the image sequence autonomously.
  - to fill

**Output data:** (to fill)
  
  
<!-- Usage -->
## Usage 

Training: first, create the tiles used for training. Make your own config file by copying the template (paths.yml) into my_config.yml, then
<br/>
```
python make_data.py create_files --config my_config.yml
```

Running a trained model  
To fill
  
<!-- Roadmap -->
## Roadmap

27/06 : The base functionnalities are up and working<br />
10/07 : Base commands to visualise result and lauch test are available. Some tests are online.
15/07 : The project is efficient and can be more precise than some others methods<br />
29/07 : End of the project<br />

<!-- Contact -->
## Contact

If you encounter any issue with this project, please submit an issue with Github !
