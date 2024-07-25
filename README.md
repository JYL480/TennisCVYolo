<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/JYL480/TennisCVYolo">
  </a>

<h3 align="center">Computer Vision Tennis Analysis</h3>

  <p align="center">
    Utilised YOLO and CNN models to extract key points on the players and tennis court
    <br />
    <br />
  </p>
</div>


<!-- ABOUT THE PROJECT -->
## About The Project

![Alt Text](https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif)

### Machine Learning Computer Vision

I made use of Pytorch's torchvision package and trained a pre-trained efficientnet_b2 model with a small dataset of food pictures (Steak, Sushi, and Pizza). I used Google's GPU from Colab to train my model!
Below are my loss and accuracy curves. Both show minor over and underfitting

![image](https://github.com/JYL480/FoodClassificationFastAPI/assets/106604224/26d9f472-2846-4317-bce8-36fd7b9b003b)

### FastAPI & Docker

I utilized FastAPI to create an API for me to POST a photo and to predict what is in the image. The API is created locally.

#### Docker Setup

To facilitate deployment and scaling, the API was containerized using Docker. This allows for greater flexibility and consistency across different environments.

#### FastAPI Backend

The FastAPI application is containerized to handle the machine learning inference. The Docker configuration ensures that the API can be easily deployed and scaled.


<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* ![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
* ![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
* ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
<!-- GETTING STARTED -->
## Getting Started

Follow these instructions to set up your project locally using Docker.

### Prerequisites

Make sure you have the following software installed:

* **Docker**
  * [Install Docker](https://docs.docker.com/get-docker/)

* **Docker Compose**
  * Docker Compose is included with Docker Desktop for Windows and Mac. For Linux, follow the [Install Docker Compose](https://docs.docker.com/compose/install/) guide.

Build and run the Docker containers using Docker Compose:

  ```sh
  docker compose up
  ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
