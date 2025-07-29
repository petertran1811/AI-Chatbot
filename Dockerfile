FROM 803614452193.dkr.ecr.ap-southeast-1.amazonaws.com/ai-core-mini-chatbot:0.1.0_base
WORKDIR /app

# COPY code to /app
COPY . .

WORKDIR /app

ENV PYHTONUNBUFFERED=1
ENV NUM_WORKER=1

RUN apt update && apt install -y build-essential

RUN chmod 777 ./run_service.sh
#RUN pip uninstall diffusers -y

CMD [ "sh", "-c" ,"./run_service.sh"]