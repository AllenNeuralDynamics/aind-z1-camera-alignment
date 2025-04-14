# aind-z1-camera-alignment

This is the code for a code ocean computational capsule. 
It runs camera alignment between spectrally neighbouring channels collected at the same time on the Zeiss Z1 microscope. 

This is accomplished by running interest point detection on individual tiles and finding pairs of matched points between spectrally neighbouring channels (within each tile). 
A 2D affine transform is calculated with RANSAC to align these neighbouring channels. 
Based on the number of inlier matched points between tiles, a global weighted average for each pair of channels is computed and applied to all tiles in a channel. 

