# TransUNet - Liver and Tumor Segmentation

This folder holds code for [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/pdf/2102.04306.pdf) and adapted to run it in LITS dataset.

The parts that where changed to be able to run it in Polyaxon and LITS data set there is a commet that says `# CHANGE`
The code is able to run it locally or in polyaxon. However, is recommend it to run it in polyaxon for better performance.

## Usage

### 1. Download Google pre-trained ViT models
* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): R50-ViT-B_16, ViT-B_16, ViT-L_16...
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz &&
mkdir ../model/vit_checkpoint/imagenet21k &&
mv {MODEL_NAME}.npz ../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```

### 2. Prepare data

Please go to ["./datasets/README.md"](datasets/README.md) for details, the pre-processed is in NAS folders "Natalia AP".

### 3. Parameter Configuration

In the file ["./config/config_dncnn.yaml"](config_dncnn.yaml) it is possible to set if the code is run in Polyaxon (`on_polyaxon: True`) or Locally `on_polyaxon: False`.

In addition, it is possible to set:
- `lits_dataset`: Folder address where is located the LITS dataset (locally or Polyaxon)
- `synapse_dataset`: Folder address where is located Synapse dataset (locally or Polyaxon)
- `pretrained_model`: Folder address where is located the pretained model
- `gpus`: GPU to use
- `num_workers`: Number of workers to train the model

### 4. Environment

Please prepare an environment with python=3.7 and cuda 11 to be able to run it in polyaxon, and the requirements are located in "requirements.txt" for the dependencies.


### 5. Train/Test

- Run the polyaxon file "polyaxonfile.yaml". To run the training comment line 32 of the file and uncomment line 31 of the file, and set or change the different hyperparameters.
    - `--vit_name`: pretained model used.
    - `--dataset`: Dataset to train
    - `--base_lr`: Learning rather
    - `--max_epochs`: max_epochs
    - `--img_size`: size of the image
    - `--batch_size`: batch size

```bash
cmd: CUDA_VISIBLE_DEVICES=0 python -u train.py --dataset LITS --vit_name R50-ViT-B_16 --base_lr 0.01 --max_epochs 15 --img_size 256 --batch_size 20
```
- Run the polyaxon file "polyaxonfile.yaml". It supports testing for both 2D images and 3D volumes. 
- To run the Testing comment line 31 of the file and uncomment line 32 of the file, and set or change the different hyperparameters.
    - `--vit_name`: pretained model used.
    - `--dataset`: Dataset to train
    - `--base_lr`: Learning rather
    - `--max_epochs`: max_epochs
    - `--img_size`: size of the image
    - `--batch_size`: batch size
    - `--model_time`: Folder where the model is save and the time where it was created
    - `--is_savenii`: Add if you want to save the resulting images,prediction and ground truth of your testing data.

```bash
cmd: python test.py --dataset LITS --vit_name R50-ViT-B_16 --base_lr 0.01 --max_epochs 15 --img_size 256 --batch_size 20 --model_time 20230321_07_32_54 --is_savenii
```
The output and the saved files are located in NAS cluster in the folder

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing(SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thank you to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README
Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
