chinese-dialects
------
This is for a competition hosted by iflytek (see challenge.xfyun.cn).
The aim is to classify short recordings into 10 different chinese dialects,
where the input is a 16-bit 16khz pcm file.

I used the same HCopy tool and config for feature extracting, as provided
by the official baseline, and simply use the off-the-shelf resnet34 for
further feature learning and predicting.

I also included another head for phoneme prediction, jointly trained by ctcloss.

From the technical aspect, I explored the use of field annotation to make a
unified interface so that all the data processing pipeline can be fit in one
single nn.Sequential module and fit easily.

The system ranked 6-th place on the final competition.
(~84% on testset, with the 1st place being 90%)

Some ideas for further improvement:
* explore other acoustic features such as plp (could be more robust)
* set a frequency threshold to remove high-freq noises
* use two-stage training instead of joint-training
(this way, the phoneme classifier could extract better features)
* add more loss for confusing pairs (e.g. Hakka vs Hokkien)
* reduce the number of layers in resnet
* more ensemble

To reproduce, first download the data from the aforementioned website,
then extract them under $data_dir, one dialect per sub-directory, then:
```
./TrainModel.sh $data_dir model.pkl
./TestModel.sh $data_dir model.pkl result.txt
```

You may also need to prepare the environment as stated in the Dockerfile.
The HTKToolkit's HCopy binary file and SeanNaren's warp-ctc library are
not provided, but can be easily downloaded (and compiled) from their websites.
