# Mushroom Classification modeling and deployment

This is a mushroom classification project, where model predicts whether mushroom is edible or not based on different characteristics of the mushroom. The dataset was downloaded from UCI ML repository (it was uploaded in 2021) and is not in the Azure ecosystem. The project consisted of two different uproaches to the task: AutoML and HyperDrive. In AutoML a bunch of different models were trained, whereas in HyperDrive only logistic regression hyperparameters were tuned. The best model was then deployed in ACI. 

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview
The dataset consisted of different mushroom characteristics, which are classified into edible or poisonous. The dataset was downloaded from UCI ML repository and consists of around 60K rows. Classes are balanced. The dataset consists of 19 features, where 3 of them numerical and the rest is categorical. However, some features were with more Nulls then actual values, hence, I dropped them. There were other features with Null, but all of them were categorical features, so I filled them with category 'Other'. 

### Task
This is a binary classification task, where classes in dataset are balanced.

### Access
Initially, I downloaded the dataset from UCI ML website. Then I uploaded it into Azure Workspace by registering it via uploading file. In the notebooks, I accessed the dataset by providing its name to Dataset.get_by_name() function. 

## Automated ML

As AutoMLConfig, I set iteration_timeout_minutes for 10 minutes in order to exclude events of long model training time per run. I also set experiment_timeout_minutes to 20 minutes, so that the experiment will not take too much time. I also enables early stopping in order to avoid time wasted on training models with no significant improvement. The task was binary classification and classes were balanced, therefore, I set primary metric as accuracy. Lastly, I set featurization to auto, to hendle numerical and categorical values automatically, and cross validation to 5 (since it is standard). 


### Results

AutoML was able to achieve almost perfect accuracy result (around 0.998 score). The best model was StackEnsemble, which is a meta-learning algorithm that optimizes combination of predictions of base ML algorithms. It consisted of 5 different decision tree based models (4 Gradient Boosters and 1 Random Forest). Each base algorithm consisted of own hyperparameters, which I will not go in detail. 

!['Proof'](https://github.com/bekiichone/nd00333-capstone/blob/master/starter_file/screenshots/AutoML%20run%20details.PNG)
AutoML Run Details

!['Proof'](https://github.com/bekiichone/nd00333-capstone/blob/master/starter_file/screenshots/AutoML%20best%20model.PNG)
AutoML best model 

## Hyperparameter Tuning

Since AutoML best model consisted of ensemble of tree based models, I wanted to compare it with linear classifcation model. Maybe simpler model is enough for the classification task? This was the question I was trying to answer. Therefore, I chose to train Logistic Regression model. As of hyperparameter space, the parameters consisted of Regularization Coefficient (C) and Maximum Iterations (max_iters). For each parameter I provided 5 values, hence, hyperparameter space size is 25. 


### Results

The best model achieved accuracy at around 0.82, which was much lower compared to AutoML best model. Does it prove that linear model is not enough for the task? Not really. Because I have only tried to tune only two set of parameters, where the best model had C = 10 and max_iter = 200. 

!['Proof'](https://github.com/bekiichone/nd00333-capstone/blob/master/starter_file/screenshots/hyperdrive%20run%20details.PNG)
HyperDrive Run Details

!['Proof'](https://github.com/bekiichone/nd00333-capstone/blob/master/starter_file/screenshots/hyperdrive%20best%20model.PNG)
HyperDrive best model 

## Model Deployment
The task was to deploy best model, therefore, I deployed AutoML best model. Below is the screenshot of active endpoint. 

!['Proof'](https://github.com/bekiichone/nd00333-capstone/blob/master/starter_file/screenshots/AutoML%20deploy%20endpoint.PNG)

Prior to deployment, I have registered the model in the workspace. Next, one have to provide entry script to InferengeConfig. You can write custom one, however, the easiest way is to download the scoring script from the best run. Then it is necessary to create an Environment class from yaml file (which can be downloaded form workspace). This is necessary because deployment may cause error and it is refered to explicitely providing environment to InferenceConfig. Next step is to set up Webservice configuration (in our case it is ACI). Finally, we can deploy the model by providing workspace, model, inference and webservice configurations. After a couple of minutes the model should be deployed successfully. 

In order to send a request to the endpoint we can take prepared script from Deploy page -> Consume script. There we have data provided as dictionary, which should be dumped as json file. Below is an example of such dictionary. 

data = {
    "Inputs": {
        "data":
        [
            {
                "cap-color": "o",
                "cap-diameter": 16.60,
                "cap-shape": "x",
                "cap-surface": "g",
                "does-bruise-or-bleed": False,
                "gill-attachment": "e",
                "gill-color": "w",
                "gill-spacing": None,
                "habitat": "d",
                "has-ring": True,
                "ring-type": "g",
                "season": "u",
                "stem-color": "w",
                "stem-height": 17.99,
                "stem-root": "s",
                "stem-surface": "y",
                "stem-width": 18.19,
                "veil-color": "w",
                "veil-type": "u"
            },
        ]
    },
    "GlobalParameters": {
        "method": "predict"
    }
}

Then we have to construct a request by providing endpoint url, header and body, where body is the json dump of test features provided for prediction. Then we send request via urlopen and receive a responce via read() function. The request should give back result something like: b'{"Results": ["p"]}', which shows that our provided mushroom is classified as poisonous (p). 

## Furute Improvement Suggestions

### Azure ML model

Because it achieved almost perfect accuracy, I am not sure that any idea will bring significant improvement, though, one may try to manually featurize the dataset in order to catch last edge cases in classification. 

### Hyperparameter Tuning Model

One may also try increase hyperparameter space by providing more parameters or parameter values. Another important step is featurization. In my case, I did not worked on feature generation and filled None values with one value. One may also do thorough exploratory data analysis and create more features. Lastly, I did OneHot encoding on categorical features. However, probobly other encoding techniques will produce better result as well as dimensionality reduction algorithms. Due to OneHot encoding the feature space increased drastically, which prone to dimensionality curse. Hence, PCA or any other dimensionality reduction algorithm may be a good idea to use. 


## Screen Recording
https://youtu.be/ajo98WoerRo

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
