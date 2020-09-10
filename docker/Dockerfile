FROM continuumio/miniconda3:latest

LABEL maintainer "d.legorreta.anguiano@gmail.com"

##Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN export DEBIAN_FRONTEND=noninteractive
RUN apt-get update --fix-missing\
    && apt-get upgrade -y


RUN conda update -n base -c defaults conda
RUN conda clean --all

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY app_basic ./

COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml

RUN echo "source activate textmining-env" > ~/.bashrc
ENV PATH /opt/conda/envs/textmining-env/bin:$PATH
RUN python -m spacy download es_core_news_sm
RUN python -m spacy download es_core_news_md
RUN python -m spacy download es_core_news_lg
RUN echo "Make sured streamlit is installed:"
RUN python -c "import streamlit;print('Streamlit Version:{}'.format(streamlit.__version__))"
# Expose port 
EXPOSE 8501

# cmd to launch app when container is run
CMD streamlit run app.py
#CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app
# streamlit-specific commands for config
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'

RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'
