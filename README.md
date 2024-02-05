<img width="472" alt="image" src="https://github.com/budhewarvijay0407/Transaction-description-classification/assets/70945959/cad9a171-7fd9-4d3c-8502-8eed766fc80f">
The primary goal here is to generate the labels first and then train a supervised model using these 
generated data to classify the types of transaction. 
We have been given a non-labelled data of bank transactions, these data have various levels of 
transactions in it and it may happen that we will get completely new input data in future. The goal is 
to minimize or (completely get rid of expensive models like OpenAI to get the final prediction) and 
use comprehensive supervised classification model to predict the old as well as completely new kind 
of input data into its respective class 
1. Process of Labelling the dataset:   
we have almost 5000, samples (raw) after cleaning the provided input dataset.   These data samples 
are not labelled and its quite hard to tag them with label manually, here we have introduced the first 
block of the solution: use of OpenAI to label the raw dataset  
As you can see that using prompts, we have passed the raw input (cleaned) data into the OpenAI 
LLM and asked them to given them the label as per its description. Below is the snapshot of the 
labels provided by openai for some of the complex description: 
2.Validate the labels by OpenAI:  
In this simple step, we manually picked random samples from the OpenAI and validated the labels, 
the temperature and Top-p hyperparameters were set to minimum and the prompts were optimally 
designed therefore we found that almost each description was labelled correctly  
Below are the total labels we used: 
1.Personal Expense: Education, Entertainment, Travel etc 
2.Online Purchase: Online shopping, Subscriptions, mandates etc 
3.Food Bill: Restaurant bill, Online Food Bill etc 
4. Card Payment: transaction where the details of “what” transaction are not defined but card details 
are mentioned 
5. Bank to Bank /Deposits - Personal transaction: Transaction done to other banks or person 
Once we have validated the transaction type on selected data samples, we are now ready to develop 
machine learning model  
3.Model Generation: 
Although I have designed 2 different ways to predict the type of transaction, the best way comes out 
as the 2nd way of “Use of Embeddings with PCA”, so in this document I would be explaining the 
working of this model. 
Flowchart of “Embedding model with PCA.” 
The input data is first converted into embedding vector (Numerical representation of the description) 
, This embedding or vector is of dimension 768*1 , to compress this size and reduce the dimension 
we used PCA , the usage of PCA was subjected to one constrain : “we should retain at least 80% of 
the 768 dimension’s information”
