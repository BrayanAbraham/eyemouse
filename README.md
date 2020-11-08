# EYE MOUSE FOR MOTOR IMPAIRED

An Eye mouse which uses eye and head movement to control the mouse of a computer.

## INSTRUCTIONS

1. Install `virtualenv`

```shell
python -m pip install virtualenv
```

2. Create a Virtual Environment

```shell
python -m venv env
```

3. Activate Virtual Environment

```shell
source env/bin/activate
```

4. Install cmake & wheel

```shell
python -m pip install cmake wheel
```

5. Install Requirements

```shell
python -m pip install -r requirements.txt
```

6. Download `shape_predictor_68_face_landmarks.dat` from [here](https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat) to the root folder

7. Run program

```shell
python eyeMouse.py
```

8. Instructions to use the mouse
   - select between modes : close both eyes
   - select mouse mode : move eye to left
   - select scroll mode : move eye to right
   - Move mouse : Move head to the direction the mouse is to be moved
