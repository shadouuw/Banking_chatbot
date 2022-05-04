# <font color = berry >Banking Chatbot </font>




## Objective
1. Data Preparation 
2. Data Modeling and Evaluation
3. Deployment




## Overview
Jawebni is  an Intelligent chatbot that can find the best answer to any request
## Dataset Information
The dataset is stored in mongodb database and contain three essential columns question , answer and explication 

## Technologies Used



![alt text](images/python1.png "Logo Title Text 1")
###Python 3.9.5

![alt text](images/mongodb.png "Logo Title Text 1")
###MongoDB 
![alt text](images/docker.png "Logo Title Text 1")
###Docker
![alt text](images/flask.jpg "Logo Title Text 1")
###Flask 2.0

#How to set up the project? 

you need to have a docker environment installed than execute both these commands : 

```docker
docker build -t chatbot_final .    
```
```docker
docker run --name  chatbot_final   -p 8000:8000 chatbot_final     
```