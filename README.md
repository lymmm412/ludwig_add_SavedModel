# ludwig_add_SavedModel
The official site is [here](https://uber.github.io/ludwig) and the official github repo is [here](https://github.com/uber/ludwig). 

Inspired by the code provided by [ifokeev](https://github.com/ifokeev) in this [issue](https://github.com/uber/ludwig/issues/329), I add 2 functions to save the model as SavedModel format and use freeze_graph.py to freeze the model into a .pb file, which can be used in the model deployment.\

The dataset I use is [Titanic from Kaggle](https://www.kaggle.com/c/titanic).

**You should install Ludwig from source code instead of pip install.**\
**I'm new to tensorflow, so if there is something wrong, please tell me, thank you!**
# Save the tensorflow model as SavedModel format
Clone the official ludwig project first. I just modified ```/ludwig/models/model.py``` 
I added the following 2 functions between the ```restore``` func and ```load``` func:
```
  # ----------------------------------------------- begin -------------------------------------------------#

    def get_tensors(self, model_definition, model):
        input_tensors = {}
        for input_feature in model_definition['input_features']:
            input_tensors[input_feature['name']] = getattr(model, input_feature['name'])

        output_tensors = {}
        for output_feature in model_definition['output_features']:
            output_tensors[output_feature['name']] = getattr(model, output_feature['name'])

        return input_tensors, output_tensors

    def saved_model(self,model_dir):
        model, model_definition = load_model_and_definition(model_dir)
        input_tensors, output_tensors = self.get_tensors(
            model_definition,
            model,
        )
        saved_model_path = os.path.join(model_dir, 'saved_model')
        print('the saved model path: ',saved_model_path)
        builder = tf.saved_model.builder.SavedModelBuilder(saved_model_path)
        with self.session as sess:
            builder.add_meta_graph_and_variables(
                sess,
                [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'predict': tf.saved_model.predict_signature_def(input_tensors, output_tensors)
                },
                strip_default_attrs=True,
                saver=self.saver,
            )
        builder.save()
        # ---------------------------------------------- end --------------------------------------------------#
```
And I added the following lines in model.py/train func to all these two functions, and it should be added to where the training loop in the ```train``` has been done.
```
# ===== save as saved model (lym) ===== #
        dir_list = os.listdir('results')
        dir_list.sort(key= lambda fn:os.path.getatime('results/'+fn))
        model_dir =os.path.join('results',dir_list[-1])+'/model' # the model_dir is inside the current experiment_run_num/model folder
        self.saved_model(model_dir)
 ```
 The structure of SavedModel is like:
 ```
 ---saved_model
    ---variables
        ---variables.data-xxxx-of-xxxx
        ---variables.index
    ---saved_model.pb
 ```
**If my description is not clear to you, you can check these two files I uploaded and search for the function ```saved_model```**

## How to get your output_node_names
Because the output_node_names is a required parameter for freeze_graph.py, we need to find it before training.
I tried to find the output node from the visualized graph but I fail. I tried to use graph.read() but I got some errors I fail to solve. Therefore, I used the simplest way: print out the output node when training! \
1.find the ```batch_evaluation``` func in model.py
2. print out the output nodes\
After doing so, you can see the all the output node names. They are in the format of XXX/XXX/XXX:/0(:/0 is not included) However, I find that only one output node name is right so I just ues that one to freeze my graph.

## Start your training
Because we have modified the source code, you should install and build Ludwig from source code. You can follow the guide in the official website. Then train with your dataset with the following command: (you can see more examples in the official site)
```
ludwig train --data_train_csv gender_submission.csv --data_test_csv test.csv --model_definition_file model_definition.yaml
```
## Freeze the SavedModel into a .pb file
Since python has provided the ```freeze_graph.py``` (If you cannot find it, you can download it from my repo. I'm using python 3.6), you can simply using this command to freeze the graph:
```
python freeze_graph.py --input_saved_model_dir=/yourpath/saved_model --output_node_names=Survived/measures_Survived/correct_predictions_Survived --output_graph=model.pb
```
Now the ```model.pb``` can be used for the model deployment.

