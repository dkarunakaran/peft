FROM jupyter/minimal-notebook

RUN pip install --upgrade pip

RUN pip install tensorflow==2.12.0

RUN pip install numpy

RUN pip install pandas

RUN pip install keras-nlp -q

RUN pip install tensorflow-datasets

RUN pip install jaxlib



