FROM public.ecr.aws/lambda/python:3.8

RUN pip install keras-image-helper
COPY 