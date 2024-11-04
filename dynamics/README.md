## Getting started

### To build environments
#build docker image
docker build -t grits_dynamics -f dynamics_image .

#build docker container
```
docker run --gpus all -v /home/hcis-s22/benyang/scoop-env/dynamics:/workspace/dynamics \
			-v /media/hcis-s22/data/new_dataset/dataset:/workspace/dataset \
	--name training_flow --shm-size="8g" -it dynamics
```
## Set parameters
```
Please set in config file
```

```
## Run
```
python mini_main.py



