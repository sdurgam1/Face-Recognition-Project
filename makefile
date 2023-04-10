install_lib:
	sudo apt-get install libboost-all-dev
install_update:
	sudo apt-get update
install_cmake:
	sudo apt-get install cmake
install_req:
	pip install --ignore-installed -r requirements.txt
install:
	sudo apt-get install libboost-all-dev
	sudo apt-get install cmake
	pip install --ignore-installed -r requirements.txt

run:
	python3 final_facedetection.py
run_hog_full:
	python3 final_facedetection.py hog create train
run_hog_train:
	python3 final_facedetection.py hog no train
run_hog_predict:
	python3 final_facedetection.py hog no predict

run_face_full:
	python3 final_facedetection.py normal create train
run_face_train:
	python3 final_facedetection.py normal no train
run_face_predict:
	python3 final_facedetection.py normal no predict
