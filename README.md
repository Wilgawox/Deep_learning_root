<!-- Source : Best-README-template on : https://github.com/othneildrew/Best-README-Template -->

<h3 align="center">Root Detection On 2D+T Images</h3>

  <p align="center">
    Project designed to use Deep learning in images of growing Arabidopsis Thaliana roots to recognize which pixel is part of root, and what is part of background
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
- Tensorboard
<br/>
You can use this command to install tensorflow GPU with all dependancies or go at [tensorflow website](https://www.tensorflow.org/install) : 
<br/>
Or you can install it with `pip install tensorflow`
<br/>

### Installation

You can use this package using ```TODO : add Git command to write it locally```


  
## Preparing your dataset

**Input data:** 
  - 2D + t image sequence acquired with an imaging automaton presenting a growing root system in a petri dish. Data should be organized in an input directory, one subdirectory per dish, with name patterning such as img_XX.tif, in order that ImageJ detect the image sequence autonomously.
  - 

**Output data:** (to fill)
  
  
<!-- Usage -->
## Usage 

**Creating local files :**
To create the local files you will need to execute a CNN, you will need to copy the template YAML file (paths.yml), and create a custom version, named my_config.yml, with your personnal values. Then you can go to the Deep_learning_root repository, and type : 
<br/>
```
python make_data.py create_files --config my_config.yml
```
It will create a data repository in you local folder named data/ with tiles created from your 2D+t images names : "ML1_input_img0000X.timeX.numX.npy", or from the associated masks : "ML1_result_img0000X.timeX.numX.npy".

**Training a model : **
To create a model, you will need the same YAML file used to create the local files. You can then go to the repository and type : 
```
python cnn_dataset.py CNN_dataset --config my_config.yml --name model_name
```
You will then find your model_name.h5 file in your logs/ folder in the repository, along with imagesshowing that model applied to test images.

**Running a trained model :**
You  can run the test test_model_2.py by replacing the tests/model/test_model.h5 file by your own h5 file, and then going to the /tests and typing : 
```
python test_model_2
```tests.
The resulting image of this test will be saved in

<!-- Roadmap -->
## Roadmap

27/06 : The base functionnalities are up and working<br />
10/07 : Base commands to visualise result and lauch test are available. Some tests are online.
15/07 : The project is efficient and can be more precise than some others methods<br />
29/07 : End of the project<br />

<!-- Contact -->
## Contact

If you encounter any issue with this project, please submit an issue with Github !
