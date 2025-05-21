## Installation
To install the required packages, run the following command:
```bash
pip install -r requirements.txt
```
## Structure
This project is structured in a way that separates the models, datasets, and transformations into their respective directories. 
```bash
├── root/
│   ├── Models
│   ├── Datasets/
│       ├── archive/
│           ├── animals/
│               ├── animals/
│   ├── Transformations/
│       ├── Compresison/
│           ├── compression.py
|           ├── resnet_test_result.txt
|           ├── yolov_test_result.txt
│       ├── GaussianNoise/
│   ├── main.py
│   ├── requirements.txt
│   ├── README.md
```
## Usage

### Create a new transformation

To save time, I have created a script that allows you to create a new transformation without having to manually create the folder and files. You can run the following command in the terminal:

```bash
python3 create_transformation.py <transformation_name>
```

Substitute `<transformation_name>` with the name of the transformation you want to create. This will create a new folder in the "Transformations" directory with the name of the transformation and a template file for the transformation.

Similarly, you can delete a transformation by running the following command:

```bash
python3 delete_transformation.py <transformation_name>
```

There will be a prompt asking for confirmation before the transformation is deleted.

### Run the main script

The role of the "main.py" script is to run the evaluation of all existing transformations. It will create the result files in .txt format in each corresponding transformation folder, you can see the evaluation metrics at the end of this file. The confusion matrix can be also found there (See the structure).

```bash
python main.py
```

### Run the combinations test

The role of the "Combinations.py" script is to run the evaluation of combinations of transformations. It will create the result files in .txt format in each corresponding combination folder inside Transformations/Combinations, you can see the evaluation metrics at the end of this file.

If you want to test all the good combos (manually added) run
```bash
python combinations.py
```

If you want to test a specific combo, you should add parameters when running
```bash
python combinations.py -t gaussianNoise cartoon
```
Remember to replace the embedded one with what you are looking for. You can see all the functions possible in valid_transformations inside the file

### See a demo

If you want to see what exactly each transformation does, you can run the "visual_test.py" script by the command:

```bash
python visual_test.py -t gaussianNoise cartoon
```
Remember to replace the embedded one with what you are looking for. You can see all the functions possible in valid_transformations inside the file

The pictures will come out to the screen, but if nothing happens, just add: 
```bash
plt.savefig('demo.png')
```
to the end of the script. The picture will be saved in the root directory.
