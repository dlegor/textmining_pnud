# Using Docker
### Using Docker

Create local container for debug purposes

> To enable terminal usage of a docker container, remove CMD line of the `Dockerfile`
    
    cd docker
    sudo docker build -t  pnud_textmining .

Run Locally

    sudo docker run -i pnud_textminig
