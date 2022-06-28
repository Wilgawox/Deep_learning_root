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

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

You need to have a Python 3 environment, anaconda is recommended.
<br/>
If you want to train the neural network, you will need tensorflow GPU, with all dependancies (https://www.tensorflow.org/install). 
<br/>
Or you can install it with `pip install tensorflow`
<br/>
This code is also writted for windows and tested as such, so there is no guaranties for it to work outside Windows.

### Installation

You can use this package using ""TODO : add Git command to write it locally""


<!-- Usage -->
## Usage 

WIP

<!-- Roadmap -->
## Roadmap

27/06 : The base functionnalities are up and working<br />
15/07 : The project is efficient and can be more precise than some others methods<br />
29/07 : End of the project<br />

<!-- Contact -->
## Contact

If you encounter any issue with this project, please submit an issue with Github !
