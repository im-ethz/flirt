# calibrate_matrix

cam_to_plane_0117.py파일에서 cam_to_minimap_whole 함수 import 하셔서 npy_file_path에 .npy 파일들이 들어있는 폴더 경로를 입력하고 cam_point_minimap
= cam_to_minimap_whole(cam_id, cam_point, npy_file_path)
와 같이 사용하시면 됩니다. 현재는 cam_id = 1, 5, 8, 10, 12, 13, 14, 17, 21에 대해 지원되고 나머지 경우엔 "i don`t cover this cam_id"라고 에러메시지 출력하도록
해놨습니다.

# cam_map_mapping3.py

This is old version of GUI for calibration.

### Usage

python cam_map_mapping3.py \ <br />
  --cam image_to_be_calibrated_1_path \ <br />
  --cam image_to_be_calibrated_2_path \ <br />
  ...  <br />
  --cam image_to_be_calibrated_n_path \  <br />
  --map image_of_map_path


# cam_map_mapping.py

This is new version of GUI for calibration. What's difference?


- [x] Show 3 images in one screen. 2 images for calibration on the left, 1 image for floor plan on the right.
- [x] When creating a new point that has no unique id, ask the user whether it is the point on ground or not.
- [x] Classify points on the ground and the ones not on the ground with different colors. 
- [x] Activate "Calibrate 2 cam" button when the conditions are met.
- [x] Calibrate 2 camera when "Calibrate 2 cam" button is clicked.
- [x] Show 2 camera calibration results on new window.
- [x] Create the button that saves 2 camera calibration parameters when the button is clicked.
- [x] Create "Calibrate 2 cam 완료하기" button that completes the 2 cam calibrations.
- [x] Change images on screens for calibration by GUI. 
- [ ] Classify cameras that are calibrated and the ones that are not calibrated. 
- [ ] When calibrating additional cameras, place a calibrated image on the left and the image to be calibrated on the right.
- [ ] Calibrate additional camera by "Calibrate additional cam" button.
- [ ] Show results for the newly calibrated camera on new window.
- [ ] Calibrate all cameras by the "Calibrate all" button and show the result.
- [ ] Return world coordinates of the selected points on images when calibration is completed.

### Usage

python cam_map_mapping.py \ <br /> 
  --image-dir-path image_dir_path \ <br /> 
  --image-name image_to_be_calibrated_1_file_name image_to_be_calibrated_2_file_name ... 
  image_to_be_calibrated_n_file_name image_of_map_path

e.g.

python cam_map_mapping.py --image-dir-path /Users/minyong/PycharmProjects/calibrate_matrix/calibration_example_data/ --image-name 1.png 2.png 3.png 4.png 5.png 6.png 7.png 8.png minimap.png

# Reference for calibration algorithm
https://www.notion.so/Camera-Calibration-217dbe08d022438f99a76183e3439456