# Disaster Response Pipeline Project
This is a project in the course of the Udacity data science nanodegree. The purpose is to take social messages and tweets and try to predict what is required after a disaster. 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db` 
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

## Caveats
1. On my local computer training the model ran out of memory (including SWAP space), resulting in the training to be terminated by a 'Interupted by signal 9: SIGKILL'. That is why I needed to increase
the SWAP size of my computer. In order to change the SWAP size on your computer, use the following commands

   Turn off all running swap processes: 
   ```
   sudo swapoff -a
   ```
   Resize SWAP (replace the 64G with any value you like)
   ```
   sudo fallocate -l 64G /swapfile
   ```
   CHMOD swap
   ```
   sudo chmod 600 /swapfile
   ```
   Make file usable as swap
   ```
   sudo mkswap /swapfile
   ```
   Activate the SWAP file
   ```
   sudo swapon /swapfile
   ```
   
   Another option there is to reduce memory consumption is to limit the amount of messages to be used during training.
   For this we can use the 'message_limit' parameter
   