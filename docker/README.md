# APP Basic 

## Description

This application is to illustrate the functionalities and capabilities of language models. The models used in the app are those available in the Spacy package for the Spanish language.

The available language models are as follows:

* [es_core_news_sm](https://spacy.io/models/es#es_core_news_sm) 
* [es_core_news_md](https://spacy.io/models/es#es_core_news_md)
* [es_core_news_lg](https://spacy.io/models/es#es_core_news_lg)

They are added as sample texts from Programas Presupuestales.

### Using Docker Containers

Create local container to reproduce environment, follow the instructions below to use docker.

> To create the container, you need to locate the terminal where the `Dockerfile` file is located.
~~~bash    
    cd docker
    sudo docker build -t  pnud_textmining .
~~~

Run Locally

~~~bash
    sudo docker run -i pnud_textminig
~~~

The application will open in the browser.

> To stop the application you need to open another terminal, follow the instructions below to stop the application.

The following command shows the containers that are running:

~~~bash
sudo docker ps -a
~~~

Use the id corresponding to the name container *pnud_textminig*


~~~bash
sudo docker stop IMAGE_ID_to_pnud_textminig
~~~

If you want to delete the docker image, run the following command with the image id

~~~bash
sudo docker rmi IMAGE_ID_to_pnud_textminig
~~~


## Running Locally

Activate the environment to be able to run the application locally.

~~~bash
conda activate textmining-env
~~~

> In a terminal locate the folder `docker/app_basic`

~~~bash
cd docker/app_basic
streamlit run app.py
~~~

> To stop the application in the terminal run `Ctrl + C`



