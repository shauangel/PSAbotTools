# Start from the Python image, automatically retrieved from Docker Hub
FROM python:3.9

# work dir application's code
RUN mkdir /usr/src/app
WORKDIR /usr/src/app
COPY . /usr/src/app/

# priviledge settings
RUN useradd -m docker
RUN chown -R docker:docker /usr/src/app
RUN groupadd -f docker
USER docker


# Install our dependencies, by running a command in the container
RUN pip install --upgrade pip
RUN pip install gunicorn
RUN pip install -r /usr/src/app/requirements.txt
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download en_core_web_md
RUN python -m spacy download en_core_web_lg
# RUN /usr/src/app/lang_mod.sh

# Have containers serve the application with gunicorn by default
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "-w", "4", "toolbox:app", "--log-level", "debug"]