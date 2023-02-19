# CAEVAD
This is an official PyTorch implementation for "A convolutional autoencoder approach for weakly-supervised anomaly video detection"
* State-of-the-art on ShanghaiTech Campus dataset
## Installation

Clone the repository.

```bash
git clone https://github.com/duchieuphan2k1/weakly-supervised-anomaly-video-detection.git
cd weakly-supervised-anomaly-video-detection
```
Download the VideoSwin feature of the ShanghaiTech Campus dataset by this link:
[shanghaitech-video-swin](https://drive.google.com/drive/folders/1PWTNbW4VNZJ9MWoAAz5DjCggKkqEKbDF?usp=sharing/).

Thanks to this [repo](https://github.com/kapildeshpande/Anomaly-Detection-in-Surveillance-Videos.git) for the extracted Video Swin Feature above.

Download our trained model by this link:
[best_proposed_model](https://drive.google.com/file/d/1-rvvLb6CtW_SNIvWaDx9UJ3qoUb886cF/view?usp=share_link/).
## Usage
### Testing
```python
python main.py --test 1 --modelpath [path_to_trained_model] --datafolder [your_data_folder]
```
* [path_to_trained_model]: the absolute path to the trained model, which can download by the link above
* [your_data_folder]: the absolute path to the data folder, which can download by the link shanghaitech-video-swin above

Thanks to [RTFM](https://github.com/tianyu0207/RTFM.git) for the starter code.
